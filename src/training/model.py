"""
ActionPredictor — 多模态 Transformer 动作预测网络。

双模态输入:
- 动力学分支: Linear(state_dim, d_model) 编码平移等变动力学特征
  （速度/角速度 + 角度 sin/cos + 部件相对坐标），并做观测归一化
- 空间感知分支: CNN(4ch, 32x32) → 128D 局部 patch
融合后送入 Transformer Encoder，交错拼接 [obs, act] 序列，预测 a_{t+1}。
左填充位通过 key_padding_mask 在注意力中被忽略。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .config import TrainConfig


class SpatialEncoder(nn.Module):
    """CNN 编码 4 通道 32x32 局部 patch → d_model 维特征。"""

    def __init__(self, in_channels: int = 4, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 32 -> 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 16 -> 8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 8 -> 4
            nn.Flatten(),      # 128 * 4 * 4 = 2048
            nn.Linear(128 * 4 * 4, d_model),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 4, 32, 32) → (N, d_model)"""
        return self.net(x)


def build_key_padding_mask(valid_mask: torch.Tensor | None) -> torch.Tensor | None:
    """(B, T) 有效标记(1=有效, 0=填充) → (B, 2T) 加性 key_padding_mask。

    交错序列 [obs_t, act_t] 中同一时间步的两个 token 共享有效性，
    因此在时间维上做 repeat_interleave(2) 展开到 2T。
    返回浮点加性掩码（填充位 -inf，与 causal attn_mask 类型一致，避免类型混用告警）；
    最后一个时间步恒为有效，保证不会出现整行 -inf 导致的 NaN。
    """
    if valid_mask is None:
        return None
    pad = (valid_mask < 0.5).repeat_interleave(2, dim=1)  # (B, 2T) True=填充
    return torch.zeros_like(pad, dtype=torch.float32).masked_fill(pad, float("-inf"))


class _DynamicsNormMixin:
    """动力学观测归一化：以 buffer 形式随 state_dict 持久化，warmup 标定后冻结。

    归一化在 forward 内部完成，故各调用方（dataset/rollout/buffer）只需构建
    原始（未归一化）的 build_dynamics 特征，训练与推理天然一致。
    """

    def _register_dynamics_norm(self, dim: int) -> None:
        self.register_buffer("dyn_mean", torch.zeros(dim))
        self.register_buffer("dyn_std", torch.ones(dim))
        self.register_buffer("dyn_stats_ready", torch.tensor(False))

    def set_dynamics_stats(self, mean, std) -> None:
        """写入标定得到的 per-dim mean/std 并标记就绪。"""
        mean_t = torch.as_tensor(mean, dtype=torch.float32).to(self.dyn_mean.device)
        std_t = (
            torch.as_tensor(std, dtype=torch.float32)
            .clamp_min(1e-6)
            .to(self.dyn_std.device)
        )
        self.dyn_mean.copy_(mean_t)
        self.dyn_std.copy_(std_t)
        self.dyn_stats_ready.fill_(True)

    @property
    def dynamics_stats_ready(self) -> bool:
        return bool(self.dyn_stats_ready.item())

    def _normalize_dynamics(self, dynamics: torch.Tensor) -> torch.Tensor:
        return (dynamics - self.dyn_mean) / self.dyn_std


class ActionPredictor(_DynamicsNormMixin, nn.Module):
    """
    Transformer 动作预测网络。

    输入: (dynamics, patches, actions) → 预测 a_{t+1}
    - dynamics: (B, T, state_dim=33)
    - patches: (B, T, 4, 32, 32)
    - actions: (B, T, 2)

    序列构造: 交错拼接 [obs_0, act_0, obs_1, act_1, ..., obs_{T-1}, act_{T-1}]
    取最后一个 token 输出，预测下一步动作。
    """

    def __init__(self, config: TrainConfig):
        super().__init__()
        d = config.d_model
        self.config = config

        # 观测归一化 buffer（随 state_dict 保存/加载）
        self._register_dynamics_norm(config.state_dim)

        # 双模态编码
        self.spatial_encoder = SpatialEncoder(config.patch_channels, d)
        self.dynamics_proj = nn.Linear(config.state_dim, d)

        # 融合: concat(128 + 128) → 128
        self.fusion = nn.Linear(d * 2, d)

        # 动作编码
        self.action_proj = nn.Linear(config.action_dim, d)

        # 位置编码: 交错序列长度 = 2 * context_len
        self.pos_embed = nn.Embedding(2 * config.context_len, d)

        # Transformer Encoder
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

        # Action head
        self.action_head = nn.Linear(d, config.action_dim)

        self._action_scale = config.action_scale

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """生成上三角 causal mask，防止看到未来信息。"""
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float("-inf"),
            diagonal=1,
        )

    def forward(
        self,
        dynamics: torch.Tensor,
        patches: torch.Tensor,
        actions: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        训练前向。

        dynamics: (B, T, state_dim)
        patches: (B, T, 4, 32, 32)
        actions: (B, T, 2)
        valid_mask: (B, T) 可选，1=有效 0=左填充；填充位在注意力中被忽略
        returns: (B, 2)
        """
        B, T, _ = dynamics.shape
        d = self.config.d_model

        # 观测归一化（标定前 mean=0/std=1，等价恒等）
        dynamics = self._normalize_dynamics(dynamics)

        # 动力学分支
        dyn_feat = self.dynamics_proj(dynamics)  # (B, T, d)

        # 空间分支: reshape → CNN → reshape
        patches_flat = patches.reshape(B * T, *patches.shape[2:])  # (B*T, 4, 32, 32)
        spat_feat = self.spatial_encoder(patches_flat)  # (B*T, d)
        spat_feat = spat_feat.reshape(B, T, d)  # (B, T, d)

        # 融合
        fused = torch.cat([dyn_feat, spat_feat], dim=-1)  # (B, T, 2d)
        obs_embed = self.fusion(fused)  # (B, T, d)

        # 动作编码
        act_embed = self.action_proj(actions)  # (B, T, d)

        # 交错拼接: [obs_0, act_0, obs_1, act_1, ...]
        seq = torch.zeros(B, 2 * T, d, device=dynamics.device)
        seq[:, 0::2, :] = obs_embed
        seq[:, 1::2, :] = act_embed

        # 位置编码
        positions = torch.arange(2 * T, device=dynamics.device)
        seq = seq + self.pos_embed(positions).unsqueeze(0)

        # Transformer + causal mask + padding mask
        mask = self._causal_mask(2 * T, dynamics.device)
        key_padding_mask = build_key_padding_mask(valid_mask)
        out = self.transformer(
            seq, mask=mask, src_key_padding_mask=key_padding_mask
        )  # (B, 2T, d)

        # 取最后一个 token → action head
        last_token = out[:, -1, :]  # (B, d)
        raw_action = self.action_head(last_token)  # (B, 2)

        return torch.tanh(raw_action) * self._action_scale

    @torch.no_grad()
    def predict(
        self,
        dynamics: np.ndarray,
        patches: np.ndarray,
        action_history: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        单步推理。

        dynamics: (T, state_dim) 或 (1, T, state_dim)
        patches: (T, 4, 32, 32) 或 (1, T, 4, 32, 32)
        action_history: (T, 2) 或 (1, T, 2)
        valid_mask: (T,) 或 (1, T) 可选，1=有效 0=左填充
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

        vm_t = None
        if valid_mask is not None:
            vm = valid_mask[np.newaxis] if valid_mask.ndim == 1 else valid_mask
            vm_t = torch.from_numpy(vm.astype(np.float32)).to(device)

        pred = self.forward(dyn_t, pat_t, act_t, valid_mask=vm_t)  # (1, 2)
        return pred.cpu().numpy().squeeze(0)
