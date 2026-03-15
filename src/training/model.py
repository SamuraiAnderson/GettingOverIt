"""
ActionPredictor — 多模态 Transformer 动作预测网络。

双模态输入:
- 动力学分支: Linear(17, d_model) 编码 17D 纯动力学状态
- 空间感知分支: CNN(4ch, 32x32) → 128D 局部 patch
融合后送入 Transformer Encoder，交错拼接 [obs, act] 序列，预测 a_{t+1}。
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


class ActionPredictor(nn.Module):
    """
    Transformer 动作预测网络。

    输入: (dynamics, patches, actions) → 预测 a_{t+1}
    - dynamics: (B, T, 17)
    - patches: (B, T, 4, 32, 32)
    - actions: (B, T, 2)

    序列构造: 交错拼接 [obs_0, act_0, obs_1, act_1, ..., obs_{T-1}, act_{T-1}]
    取最后一个 token 输出，预测下一步动作。
    """

    def __init__(self, config: TrainConfig):
        super().__init__()
        d = config.d_model
        self.config = config

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
    ) -> torch.Tensor:
        """
        训练前向。

        dynamics: (B, T, 17)
        patches: (B, T, 4, 32, 32)
        actions: (B, T, 2)
        returns: (B, 2)
        """
        B, T, _ = dynamics.shape
        d = self.config.d_model

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

        # Transformer + causal mask
        mask = self._causal_mask(2 * T, dynamics.device)
        out = self.transformer(seq, mask=mask)  # (B, 2T, d)

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
    ) -> np.ndarray:
        """
        单步推理。

        dynamics: (T, 17) 或 (1, T, 17)
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

        pred = self.forward(dyn_t, pat_t, act_t)  # (1, 2)
        return pred.cpu().numpy().squeeze(0)
