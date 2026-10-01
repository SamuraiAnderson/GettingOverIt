"""
PPO Rollout Buffer — 存储采集数据 + GAE 优势计算 + minibatch 迭代。

每步存储模型输入窗口（dynamics, patches, action_history）和采集输出
（action, log_prob, value, reward, done）。每个 agent 使用独立 buffer，
采集完成后 per-agent 计算 GAE，再通过 merge() 合并用于训练。
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from typing import Generator

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _stack_optional(rows: list, idx, device):
    picked = [rows[i] for i in idx]
    if not picked or any(item is None for item in picked):
        return None
    return torch.from_numpy(np.stack(picked)).to(device)


@dataclass(frozen=True)
class PatchWindow:
    """References to immutable per-frame patches; materialize only a minibatch."""
    frames: tuple[np.ndarray, ...]
    context_len: int

    def __array__(self, dtype=None, copy=None):
        out = np.zeros((self.context_len, *self.frames[-1].shape), dtype=dtype or np.float32)
        out[-len(self.frames):] = self.frames
        return out

    def __getitem__(self, index):
        return np.asarray(self)[index]


@dataclass
class PPOBatch:
    """PPO 训练用的一个 minibatch。"""
    dynamics: torch.Tensor       # (B, T, state_dim)
    patches: torch.Tensor        # (B, T, C, H, H)
    act_history: torch.Tensor    # (B, T, 2)
    valid_mask: torch.Tensor     # (B, T)
    actions: torch.Tensor        # (B, 2)
    old_log_probs: torch.Tensor  # (B,)
    advantages: torch.Tensor     # (B,)
    returns: torch.Tensor        # (B,)
    old_values: torch.Tensor     # (B,)
    # 采集时该步的 OU 前一步归一化噪声 z_{t-1}，供条件式 log_prob 复算（PPO ratio 自洽）；
    # iid 模式恒为零向量，evaluate_actions 会在 noise_phi=0 时忽略。
    noise_prev: torch.Tensor     # (B, 2)
    raw_actions: torch.Tensor    # (B, 2), before tanh; never reconstruct saturated samples
    contour_tokens: torch.Tensor | None = None
    contour_mask: torch.Tensor | None = None
    relation_tokens: torch.Tensor | None = None
    relation_mask: torch.Tensor | None = None
    # 本步探索倍率。1 表示未延迟。更新必须用采集时的倍率，不能改回满噪声再算概率。
    noise_scale: torch.Tensor | None = None

    def sparse_kwargs(self) -> dict:
        if self.contour_tokens is None:
            return {}
        return {
            "contour_tokens": self.contour_tokens,
            "contour_mask": self.contour_mask,
            "relation_tokens": self.relation_tokens,
            "relation_mask": self.relation_mask,
        }


class PPORolloutBuffer:
    """
    PPO 数据缓冲区。

    调用顺序: add() × N → compute_gae() → get_batches()
    每轮采集结束后调用 reset() 清空。
    """

    def __init__(self, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self._dynamics: list[np.ndarray] = []
        self._patches: list[np.ndarray] = []
        self._act_histories: list[np.ndarray] = []
        self._valid_masks: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._log_probs: list[float] = []
        self._values: list[float] = []
        self._rewards: list[float] = []
        self._dones: list[bool] = []
        # 每步 OU 前一步归一化噪声 z_{t-1}，iid 模式下存零向量（无条件效应）
        self._noise_prev: list[np.ndarray] = []
        self._noise_scale: list[float] = []
        self._raw_actions: list[np.ndarray] = []
        self._contours: list = []
        self._cmasks: list = []
        self._relations: list = []
        self._rmasks: list = []
        self.policy_context: dict = {}

        self._advantages: np.ndarray | None = None
        self._returns: np.ndarray | None = None

    @property
    def size(self) -> int:
        return len(self._rewards)

    def add(
        self,
        dynamics_window: np.ndarray,
        patch_window: np.ndarray,
        act_history_window: np.ndarray,
        valid_mask_window: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        noise_prev: np.ndarray | None = None,
        *,
        raw_action: np.ndarray,
        patch_frames: tuple[np.ndarray, ...] | None = None,
        contour_window: np.ndarray | None = None,
        contour_mask_window: np.ndarray | None = None,
        relation_window: np.ndarray | None = None,
        relation_mask_window: np.ndarray | None = None,
        noise_scale: float = 1.0,
    ) -> None:
        """添加一个 timestep；NaN/Inf 中止本轮，避免跨缺失步骤计算 GAE。

        noise_prev: OU 采集时的 z_{t-1}；未提供时按 iid 模式补零向量（evaluate_actions 侧
        用 noise_phi=0 时自然忽略，两种口径一致）。
        """
        if noise_prev is None:
            noise_prev = np.zeros(action.shape, dtype=np.float32)
        noise_scale = float(noise_scale)
        if not np.isfinite(noise_scale) or noise_scale <= 0.0:
            raise ValueError("noise_scale must be finite and positive")
        if patch_frames is not None and (not patch_frames or len(patch_frames) > len(valid_mask_window)):
            raise ValueError("Patch frame history must fit the observation window")

        sparse_ok = all(
            arr is None or np.isfinite(arr).all()
            for arr in (contour_window, contour_mask_window, relation_window, relation_mask_window)
        )
        if not (
            np.isfinite(dynamics_window).all()
            and np.isfinite(patch_window).all()
            and np.isfinite(action).all()
            and np.isfinite(log_prob)
            and np.isfinite(value)
            and np.isfinite(reward)
            and np.isfinite(noise_prev).all()
            and np.isfinite(raw_action).all()
            and np.isfinite(act_history_window).all()
            and np.isfinite(valid_mask_window).all()
            and sparse_ok
        ):
            logger.warning(
                "Invalid timestep with NaN/Inf: "
                "dyn=%s patch=%s action=%s lp=%.4g v=%.4g r=%.4g noise=%s",
                np.isfinite(dynamics_window).all(),
                np.isfinite(patch_window).all(),
                np.isfinite(action).all(),
                log_prob, value, reward,
                np.isfinite(noise_prev).all(),
            )
            raise ValueError("Non-finite PPO transition; refusing to join discontinuous GAE history")

        self._dynamics.append(dynamics_window)
        self._patches.append(patch_window if patch_frames is None else
                             PatchWindow(patch_frames, len(valid_mask_window)))
        self._act_histories.append(act_history_window)
        self._valid_masks.append(valid_mask_window)
        self._actions.append(action)
        self._log_probs.append(log_prob)
        self._values.append(value)
        self._rewards.append(reward)
        self._dones.append(done)
        self._noise_prev.append(noise_prev.astype(np.float32))
        self._noise_scale.append(noise_scale)
        self._raw_actions.append(np.asarray(raw_action, dtype=np.float32).copy())
        self._contours.append(None if contour_window is None else np.asarray(contour_window, np.float32))
        self._cmasks.append(None if contour_mask_window is None else np.asarray(contour_mask_window, np.float32))
        self._relations.append(None if relation_window is None else np.asarray(relation_window, np.float32))
        self._rmasks.append(None if relation_mask_window is None else np.asarray(relation_mask_window, np.float32))

    def compute_gae(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """
        计算 GAE-lambda 优势估计和折扣回报。

        last_value: 最后一步之后的 V(s_{T})，用于 bootstrap。
        """
        n = self.size
        rewards = np.array(self._rewards, dtype=np.float64)
        values = np.array(self._values, dtype=np.float64)
        dones = np.array(self._dones, dtype=np.float64)

        advantages = np.zeros(n, dtype=np.float64)
        gae = 0.0
        next_value = float(last_value)

        for t in reversed(range(n)):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * non_terminal - values[t]
            gae = delta + gamma * gae_lambda * non_terminal * gae
            advantages[t] = gae
            next_value = values[t]

        self._advantages = advantages.astype(np.float32)
        self._returns = (advantages + values).astype(np.float32)

    def get_batches(
        self,
        batch_size: int,
        normalize_advantages: bool = True,
    ) -> Generator[PPOBatch, None, None]:
        """随机打乱后按 batch_size 分割，yield PPOBatch。"""
        assert self._advantages is not None, "先调用 compute_gae()"
        n = self.size
        indices = np.random.permutation(n)

        actions_arr = np.array(self._actions)
        log_probs_arr = np.array(self._log_probs, dtype=np.float32)
        values_arr = np.array(self._values, dtype=np.float32)
        noise_prev_arr = np.array(self._noise_prev, dtype=np.float32)
        noise_scale_arr = np.asarray(self._noise_scale, dtype=np.float32)
        raw_actions_arr = np.array(self._raw_actions, dtype=np.float32)

        adv = self._advantages.copy()
        if normalize_advantages and len(adv) > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            yield PPOBatch(
                dynamics=torch.from_numpy(np.stack([self._dynamics[i] for i in idx])).to(self.device),
                patches=torch.from_numpy(np.stack([self._patches[i] for i in idx])).to(self.device),
                act_history=torch.from_numpy(np.stack([self._act_histories[i] for i in idx])).to(self.device),
                valid_mask=torch.from_numpy(np.stack([self._valid_masks[i] for i in idx])).to(self.device),
                actions=torch.from_numpy(actions_arr[idx]).to(self.device),
                old_log_probs=torch.from_numpy(log_probs_arr[idx]).to(self.device),
                advantages=torch.from_numpy(adv[idx]).to(self.device),
                returns=torch.from_numpy(self._returns[idx]).to(self.device),
                old_values=torch.from_numpy(values_arr[idx]).to(self.device),
                noise_prev=torch.from_numpy(noise_prev_arr[idx]).to(self.device),
                raw_actions=torch.from_numpy(raw_actions_arr[idx]).to(self.device),
                noise_scale=torch.from_numpy(noise_scale_arr[idx]).to(self.device),
                contour_tokens=_stack_optional(self._contours, idx, self.device),
                contour_mask=_stack_optional(self._cmasks, idx, self.device),
                relation_tokens=_stack_optional(self._relations, idx, self.device),
                relation_mask=_stack_optional(self._rmasks, idx, self.device),
            )

    @classmethod
    def merge(cls, buffers: list[PPORolloutBuffer]) -> PPORolloutBuffer:
        """合并多个已完成 GAE 计算的 per-agent buffer 为一个训练用 buffer。"""
        merged = cls()
        if not buffers:
            merged._advantages = np.array([], dtype=np.float32)
            merged._returns = np.array([], dtype=np.float32)
            return merged
        for b in buffers:
            assert b._advantages is not None, "merge 前需先 compute_gae"
            if b.policy_context != buffers[0].policy_context:
                raise ValueError("Cannot merge different behavior-policy contexts")
            merged._dynamics.extend(b._dynamics)
            merged._patches.extend(b._patches)
            merged._act_histories.extend(b._act_histories)
            merged._valid_masks.extend(b._valid_masks)
            merged._actions.extend(b._actions)
            merged._log_probs.extend(b._log_probs)
            merged._values.extend(b._values)
            merged._rewards.extend(b._rewards)
            merged._dones.extend(b._dones)
            merged._noise_prev.extend(b._noise_prev)
            merged._noise_scale.extend(b._noise_scale)
            merged._raw_actions.extend(b._raw_actions)
            merged._contours.extend(b._contours)
            merged._cmasks.extend(b._cmasks)
            merged._relations.extend(b._relations)
            merged._rmasks.extend(b._rmasks)
        merged.policy_context = dict(buffers[0].policy_context)
        merged._advantages = np.concatenate([b._advantages for b in buffers])
        merged._returns = np.concatenate([b._returns for b in buffers])
        return merged

    def reset(self) -> None:
        """清空所有数据，准备下一轮采集。"""
        self._dynamics.clear()
        self._patches.clear()
        self._act_histories.clear()
        self._valid_masks.clear()
        self._actions.clear()
        self._log_probs.clear()
        self._values.clear()
        self._rewards.clear()
        self._dones.clear()
        self._noise_prev.clear()
        self._noise_scale.clear()
        self._raw_actions.clear()
        self._contours.clear()
        self._cmasks.clear()
        self._relations.clear()
        self._rmasks.clear()
        self.policy_context.clear()
        self._advantages = None
        self._returns = None

    def save(self, path) -> None:
        """Versioned exact model inputs; does not rebuild patches from a changed map."""
        if self._advantages is None or not self.size:
            raise ValueError("Only nonempty, finalized rollouts can be archived")
        names = ("dynamics", "act_histories", "valid_masks", "actions",
                 "raw_actions", "log_probs", "values", "rewards", "dones", "noise_prev",
                 "noise_scale")
        arrays = {name: np.asarray(getattr(self, "_" + name)) for name in names}
        frames, ids, windows = [], {}, []
        for window in self._patches:
            source = window.frames if isinstance(window, PatchWindow) else tuple(window)
            ctx = window.context_len if isinstance(window, PatchWindow) else len(window)
            index = [-1] * (ctx - len(source))
            for frame in source:
                if id(frame) not in ids:
                    ids[id(frame)] = len(frames)
                    frames.append(frame)
                index.append(ids[id(frame)])
            windows.append(index)
        arrays["patch_frames"] = np.stack(frames)
        arrays["patch_indices"] = np.array(windows, dtype=np.int32)
        if self._contours and all(item is not None for item in self._contours):
            arrays["contours"] = np.stack(self._contours)
            arrays["cmasks"] = np.stack(self._cmasks)
            arrays["relations"] = np.stack(self._relations)
            arrays["rmasks"] = np.stack(self._rmasks)
        np.savez_compressed(path, **arrays, advantages=self._advantages, returns=self._returns,
                            metadata=json.dumps({"version": 2, "policy_context": self.policy_context}))

    def patch_storage_bytes(self) -> int:
        """Physical frame bytes, counting shared history references only once."""
        arrays = {}
        for window in self._patches:
            for frame in window.frames if isinstance(window, PatchWindow) else (window,):
                arrays[id(frame)] = frame
        return sum(a.nbytes for a in arrays.values())

    @classmethod
    def load(cls, path, device="cpu") -> PPORolloutBuffer:
        buffer = cls(device)
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata["version"] == 3:
                from .decision_buffer import DecisionRolloutBuffer
                return DecisionRolloutBuffer.load(path, device)
            if metadata["version"] not in (1, 2):
                raise ValueError("Unsupported PPO input archive version")
            buffer.policy_context = metadata["policy_context"]
            for name in ("dynamics", "act_histories", "valid_masks", "actions",
                         "raw_actions", "log_probs", "values", "rewards", "dones", "noise_prev"):
                setattr(buffer, "_" + name, list(data[name]))
            if "noise_scale" in data.files:
                buffer._noise_scale = [float(value) for value in data["noise_scale"]]
            else:
                buffer._noise_scale = [1.0] * buffer.size
            if metadata["version"] == 1:
                buffer._patches = list(data["patches"])
            else:
                frames = list(data["patch_frames"])
                buffer._patches = [PatchWindow(tuple(frames[i] for i in ids if i >= 0), len(ids))
                                   for ids in data["patch_indices"]]
            buffer._advantages = data["advantages"]
            buffer._returns = data["returns"]
            n = buffer.size
            if "contours" in data.files:
                buffer._contours = list(data["contours"])
                buffer._cmasks = list(data["cmasks"])
                buffer._relations = list(data["relations"])
                buffer._rmasks = list(data["rmasks"])
            else:
                buffer._contours = [None] * n
                buffer._cmasks = [None] * n
                buffer._relations = [None] * n
                buffer._rmasks = [None] * n
        return buffer
