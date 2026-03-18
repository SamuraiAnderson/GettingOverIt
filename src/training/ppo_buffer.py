"""
PPO Rollout Buffer — 存储采集数据 + GAE 优势计算 + minibatch 迭代。

每步存储模型输入窗口（dynamics, patches, action_history）和采集输出
（action, log_prob, value, reward, done）。每个 agent 使用独立 buffer，
采集完成后 per-agent 计算 GAE，再通过 merge() 合并用于训练。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generator

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class PPOBatch:
    """PPO 训练用的一个 minibatch。"""
    dynamics: torch.Tensor       # (B, T, state_dim)
    patches: torch.Tensor        # (B, T, 4, 32, 32)
    act_history: torch.Tensor    # (B, T, 2)
    actions: torch.Tensor        # (B, 2)
    old_log_probs: torch.Tensor  # (B,)
    advantages: torch.Tensor     # (B,)
    returns: torch.Tensor        # (B,)
    old_values: torch.Tensor     # (B,)


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
        self._actions: list[np.ndarray] = []
        self._log_probs: list[float] = []
        self._values: list[float] = []
        self._rewards: list[float] = []
        self._dones: list[bool] = []

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
        action: np.ndarray,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        """添加一个 timestep 的数据（单 agent）。包含 NaN/Inf 的数据会被丢弃。"""
        if not (
            np.isfinite(dynamics_window).all()
            and np.isfinite(patch_window).all()
            and np.isfinite(action).all()
            and np.isfinite(log_prob)
            and np.isfinite(value)
            and np.isfinite(reward)
        ):
            logger.warning(
                "Dropping timestep with NaN/Inf: "
                "dyn=%s patch=%s action=%s lp=%.4g v=%.4g r=%.4g",
                np.isfinite(dynamics_window).all(),
                np.isfinite(patch_window).all(),
                np.isfinite(action).all(),
                log_prob, value, reward,
            )
            return

        self._dynamics.append(dynamics_window)
        self._patches.append(patch_window)
        self._act_histories.append(act_history_window)
        self._actions.append(action)
        self._log_probs.append(log_prob)
        self._values.append(value)
        self._rewards.append(reward)
        self._dones.append(done)

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

        dynamics_arr = np.array(self._dynamics)
        patches_arr = np.array(self._patches)
        act_hist_arr = np.array(self._act_histories)
        actions_arr = np.array(self._actions)
        log_probs_arr = np.array(self._log_probs, dtype=np.float32)
        values_arr = np.array(self._values, dtype=np.float32)

        adv = self._advantages.copy()
        if normalize_advantages and len(adv) > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            yield PPOBatch(
                dynamics=torch.from_numpy(dynamics_arr[idx]).to(self.device),
                patches=torch.from_numpy(patches_arr[idx]).to(self.device),
                act_history=torch.from_numpy(act_hist_arr[idx]).to(self.device),
                actions=torch.from_numpy(actions_arr[idx]).to(self.device),
                old_log_probs=torch.from_numpy(log_probs_arr[idx]).to(self.device),
                advantages=torch.from_numpy(adv[idx]).to(self.device),
                returns=torch.from_numpy(self._returns[idx]).to(self.device),
                old_values=torch.from_numpy(values_arr[idx]).to(self.device),
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
            merged._dynamics.extend(b._dynamics)
            merged._patches.extend(b._patches)
            merged._act_histories.extend(b._act_histories)
            merged._actions.extend(b._actions)
            merged._log_probs.extend(b._log_probs)
            merged._values.extend(b._values)
            merged._rewards.extend(b._rewards)
            merged._dones.extend(b._dones)
        merged._advantages = np.concatenate([b._advantages for b in buffers])
        merged._returns = np.concatenate([b._returns for b in buffers])
        return merged

    def reset(self) -> None:
        """清空所有数据，准备下一轮采集。"""
        self._dynamics.clear()
        self._patches.clear()
        self._act_histories.clear()
        self._actions.clear()
        self._log_probs.clear()
        self._values.clear()
        self._rewards.clear()
        self._dones.clear()
        self._advantages = None
        self._returns = None
