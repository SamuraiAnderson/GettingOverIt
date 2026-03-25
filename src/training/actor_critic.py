"""
ActorCritic — PPO 用 Actor-Critic 网络。

共享 backbone（SpatialEncoder + Transformer），双输出头：
- Actor: Gaussian 策略 π(a|s)，输出 (mean, std)
- Critic: 状态价值 V(s)，输出 scalar

backbone 结构与 ActionPredictor 完全一致，便于对比和迁移。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from .model import SpatialEncoder

if TYPE_CHECKING:
    from .config import TrainConfig

logger = logging.getLogger(__name__)


class ActionValue(NamedTuple):
    action: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor


class ActorCritic(nn.Module):
    """
    共享 backbone 的 Actor-Critic 网络。

    输入格式与 ActionPredictor 相同：
    - dynamics: (B, T, state_dim)
    - patches:  (B, T, 4, 32, 32)
    - actions:  (B, T, 2)

    Actor 输出经 tanh 压缩到 [-action_scale, action_scale]。
    log_prob 会做 tanh squashing 修正。
    """

    def __init__(self, config: TrainConfig):
        super().__init__()
        d = config.d_model
        self.config = config

        # ── 共享 backbone（与 ActionPredictor 结构一致）──
        self.spatial_encoder = SpatialEncoder(config.patch_channels, d)
        self.dynamics_proj = nn.Linear(config.state_dim, d)
        self.fusion = nn.Linear(d * 2, d)
        self.action_proj = nn.Linear(config.action_dim, d)
        self.pos_embed = nn.Embedding(2 * config.context_len, d)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.nhead,
            dim_feedforward=d * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )

        # ── Actor head ──
        self.actor_mean = nn.Linear(d, config.action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(config.action_dim))

        # ── Critic head ──
        self.critic_head = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )

        self._action_scale = config.action_scale
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        for m in self.critic_head:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def load_from_bc(self, bc_state_dict: dict[str, torch.Tensor]) -> None:
        """
        从 BC（ActionPredictor）的 state_dict 迁移权重。

        backbone 层 key 完全一致，直接复制。
        action_head → actor_mean（同形状 Linear(d, 2)）。
        actor_log_std 和 critic_head 保持当前初始化。
        """
        my_state = self.state_dict()
        loaded, skipped = 0, 0

        for key, param in bc_state_dict.items():
            target_key = key
            if key.startswith("action_head."):
                target_key = key.replace("action_head.", "actor_mean.", 1)

            if target_key in my_state:
                if my_state[target_key].shape == param.shape:
                    my_state[target_key] = param
                    loaded += 1
                else:
                    logger.warning(
                        "BC→PPO 形状不匹配, 跳过: %s %s vs %s",
                        target_key, param.shape, my_state[target_key].shape,
                    )
                    skipped += 1
            else:
                logger.warning("BC→PPO key 不存在, 跳过: %s", key)
                skipped += 1

        self.load_state_dict(my_state)

        _LOG_STD_FLOOR = -3.0  # std >= 0.05, 防止探索能力塌缩
        with torch.no_grad():
            w = self.actor_mean.weight  # (action_dim, d_model)
            output_scale = w.norm(dim=1).mean()
            calibrated = max(torch.log(output_scale).item(), _LOG_STD_FLOOR)
        self.actor_log_std.data.fill_(calibrated)

        logger.info(
            "BC→PPO 权重迁移完成: %d 层已加载, %d 层跳过 "
            "(output_scale=%.4f, actor_log_std=%.2f → std=%.4f, "
            "critic_head 保持初始化)",
            loaded, skipped, output_scale.item(),
            calibrated, float(self.actor_log_std.exp().mean()),
        )

    # ── backbone forward ──

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float("-inf"),
            diagonal=1,
        )

    def _backbone(
        self,
        dynamics: torch.Tensor,
        patches: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """共享特征提取，返回最后一个 token 的 embedding (B, d)。"""
        B, T, _ = dynamics.shape
        d = self.config.d_model

        dyn_feat = self.dynamics_proj(dynamics)

        patches_flat = patches.reshape(B * T, *patches.shape[2:])
        spat_feat = self.spatial_encoder(patches_flat)
        spat_feat = spat_feat.reshape(B, T, d)

        fused = torch.cat([dyn_feat, spat_feat], dim=-1)
        obs_embed = self.fusion(fused)

        act_embed = self.action_proj(actions)

        seq = torch.zeros(B, 2 * T, d, device=dynamics.device)
        seq[:, 0::2, :] = obs_embed
        seq[:, 1::2, :] = act_embed

        positions = torch.arange(2 * T, device=dynamics.device)
        seq = seq + self.pos_embed(positions).unsqueeze(0)

        mask = self._causal_mask(2 * T, dynamics.device)
        out = self.transformer(seq, mask=mask)

        return out[:, -1, :]

    # ── tanh squashing helpers ──

    def _squash_action(self, raw: torch.Tensor) -> torch.Tensor:
        """raw → tanh(raw) * scale"""
        return torch.tanh(raw) * self._action_scale

    @staticmethod
    def _log_prob_squash(
        log_prob_raw: torch.Tensor, raw_action: torch.Tensor
    ) -> torch.Tensor:
        """修正 tanh 变换的 log_prob: log π(a) = log π_raw(u) - Σ log(1 - tanh²(u))"""
        correction = torch.log(1.0 - torch.tanh(raw_action).pow(2) + 1e-6)
        return log_prob_raw - correction.sum(dim=-1)

    # ── public API ──

    def forward(
        self,
        dynamics: torch.Tensor,
        patches: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[Normal, torch.Tensor]:
        """
        返回 (distribution_on_raw_space, value)。

        注意：返回的 Normal 分布是 squash 之前的。
        调用方需通过 get_action_and_value / evaluate_actions 获取修正后的量。
        """
        feat = self._backbone(dynamics, patches, actions)
        mean = self.actor_mean(feat)
        _LOG_STD_MIN, _LOG_STD_MAX = -2.0, 0.5
        std = self.actor_log_std.clamp(_LOG_STD_MIN, _LOG_STD_MAX).exp().expand_as(mean)
        dist = Normal(mean, std)
        value = self.critic_head(feat).squeeze(-1)
        return dist, value

    def get_action_and_value(
        self,
        dynamics: torch.Tensor,
        patches: torch.Tensor,
        actions: torch.Tensor,
    ) -> ActionValue:
        """
        采集用：从策略中采样动作，返回 (squashed_action, log_prob, value)。
        """
        dist, value = self.forward(dynamics, patches, actions)
        raw_action = dist.rsample()
        log_prob_raw = dist.log_prob(raw_action).sum(dim=-1)
        log_prob = self._log_prob_squash(log_prob_raw, raw_action)
        action = self._squash_action(raw_action)
        return ActionValue(action=action, log_prob=log_prob, value=value)

    def evaluate_actions(
        self,
        dynamics: torch.Tensor,
        patches: torch.Tensor,
        act_history: torch.Tensor,
        taken_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        训练用：对已执行的动作重新计算 log_prob、entropy、value。

        taken_actions: (B, 2) — squashed 空间中的动作
        returns: (log_prob, entropy, value)
        """
        dist, value = self.forward(dynamics, patches, act_history)

        # 反 squash: squashed → raw
        clamped = taken_actions.clamp(
            -self._action_scale * 0.999, self._action_scale * 0.999
        )
        raw_action = torch.atanh(clamped / self._action_scale)

        log_prob_raw = dist.log_prob(raw_action).sum(dim=-1)
        log_prob = self._log_prob_squash(log_prob_raw, raw_action)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy, value

    @torch.no_grad()
    def predict_deterministic(
        self,
        dynamics: np.ndarray,
        patches: np.ndarray,
        action_history: np.ndarray,
    ) -> np.ndarray:
        """
        确定性推理（评估用）：直接用 mean，不采样。

        dynamics: (T, state_dim) 或 (1, T, state_dim)
        patches: (T, 4, 32, 32) 或 (1, T, 4, 32, 32)
        action_history: (T, 2) 或 (1, T, 2)
        returns: (2,)
        """
        self.eval()
        device = next(self.parameters()).device

        if dynamics.ndim == 2:
            dynamics = dynamics[np.newaxis]
        if patches.ndim == 4:
            patches = patches[np.newaxis]
        if action_history.ndim == 2:
            action_history = action_history[np.newaxis]

        dyn_t = torch.from_numpy(dynamics.astype(np.float32)).to(device)
        pat_t = torch.from_numpy(patches.astype(np.float32)).to(device)
        act_t = torch.from_numpy(action_history.astype(np.float32)).to(device)

        feat = self._backbone(dyn_t, pat_t, act_t)
        mean = self.actor_mean(feat)
        action = self._squash_action(mean)
        return action.cpu().numpy().squeeze(0)
