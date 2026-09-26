"""
ActorCritic — PPO 用 Actor-Critic 网络。

共享 backbone（SpatialEncoder + Transformer），双输出头：
- Actor: Gaussian 策略 π(a|s)，输出 (mean, std)
- Critic: 状态价值 V(s)，输出 scalar

backbone 结构与 ActionPredictor 完全一致，便于对比和迁移。
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .contact_features import CONTACT_DIM
from .dataset import BASE_DYNAMICS_DIM, DYNAMICS_DIM
from .model import (DualScaleSpatialEncoder, SparseGeometryMixin, _DynamicsNormMixin,
                    build_attn_mask, expand_state_dict_for_dynamics_dim)

if TYPE_CHECKING:
    from .config import TrainConfig

logger = logging.getLogger(__name__)


class ActionValue(NamedTuple):
    action: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    # 采样使用的 OU 归一化噪声 z_t（raw = mean + std·z_t）。iid 路径返回该步的 η_t（marginal
    # 相同 N(0,1)），供 buffer 存 noise_prev 用于下一步条件式 log_prob（PPO ratio 自洽）。
    noise: torch.Tensor | None = None
    raw_action: torch.Tensor | None = None


class ActorCritic(SparseGeometryMixin, _DynamicsNormMixin, nn.Module):
    """
    共享 backbone 的 Actor-Critic 网络。

    输入格式与 ActionPredictor 相同：
    - dynamics: (B, T, state_dim)
    - patches:  (B, T, C, H, H)  # H = config.patch_size
    - actions:  (B, T, 2)
    - valid_mask: (B, T) 可选，1=有效 0=左填充

    Actor 输出经 tanh 压缩到 [-action_scale, action_scale]。
    log_prob 会做 tanh squashing 修正。
    """

    def __init__(self, config: TrainConfig):
        super().__init__()
        d = config.d_model
        self.config = config

        # 观测归一化 buffer（随 state_dict 保存/加载；从 BC 迁移时一并复制）
        # 维度用 DYNAMICS_DIM（build_dynamics 输出=39 = 34 base + 5 contact），
        # 而非 config.state_dim（原始 33 维）
        self._register_dynamics_norm(DYNAMICS_DIM)

        # ── 共享 backbone（与 ActionPredictor 结构一致）──
        self.spatial_encoder = DualScaleSpatialEncoder(config, d)
        self.dynamics_proj = nn.Linear(DYNAMICS_DIM, d)
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
        if config.contact_heads_enabled:
            self.actor_mean_contact = nn.Linear(d, config.action_dim)
            self.actor_log_std_contact = nn.Parameter(torch.full(
                (config.action_dim,), config.contact_log_std_offset,
            ))

        # 运行时探索 boost：非可学习标量，仅在 forward 里叠加到 log_std（采集+更新一致），
        # 不入 state_dict/梯度。停滞探测器按轮设置（见 main_ppo Phase 2.6）。
        self.exploration_boost: float = 0.0

        # ── Critic head ──
        self.critic_head = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )


        self._action_scale = config.action_scale
        self._init_weights()
        if config.contact_heads_enabled:
            self.actor_mean_contact.load_state_dict(self.actor_mean.state_dict())
        # 新模块挂在参数列表末尾，旧 checkpoint / optimizer 下标仍对齐。
        self._init_sparse_encoders(config)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """34D / 单头旧权重迁移：两头复制旧均值，接触头降低初始探索方差。"""
        state = expand_state_dict_for_dynamics_dim(state_dict, self.state_dict())
        if self.config.contact_heads_enabled and "actor_mean_contact.weight" not in state:
            for suffix in ("weight", "bias"):
                state[f"actor_mean_contact.{suffix}"] = state[f"actor_mean.{suffix}"].clone()
            state["actor_log_std_contact"] = (
                state["actor_log_std"].clone() + self.config.contact_log_std_offset
            )
        return super().load_state_dict(state, strict=strict, assign=assign)

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

        bc_state_dict = expand_state_dict_for_dynamics_dim(bc_state_dict, my_state)
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

        # 迁移时初始化 actor_log_std：以 BC 输出尺度标定，但**封顶到 config.ppo_init_log_std**
        # 并下限到 ppo_log_std_min，避免中性动作处探索噪声过大冲坏 BC 开局（见 config 注释）。
        with torch.no_grad():
            w = self.actor_mean.weight  # (action_dim, d_model)
            output_scale = w.norm(dim=1).mean()
            calibrated = min(torch.log(output_scale).item(), self.config.ppo_init_log_std)
            calibrated = max(calibrated, self.config.ppo_log_std_min)
        self.actor_log_std.data.fill_(calibrated)
        if self.config.contact_heads_enabled:
            self.actor_mean_contact.load_state_dict(self.actor_mean.state_dict())
            self.actor_log_std_contact.data.fill_(calibrated + self.config.contact_log_std_offset)

        logger.info(
            "BC→PPO 权重迁移完成: %d 层已加载, %d 层跳过 "
            "(output_scale=%.4f, actor_log_std=%.2f → std=%.4f, "
            "critic_head 保持初始化)",
            loaded, skipped, output_scale.item(),
            calibrated, float(self.actor_log_std.detach().exp().mean()),
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
        valid_mask: torch.Tensor | None = None,
        contour_tokens: torch.Tensor | None = None,
        contour_mask: torch.Tensor | None = None,
        relation_tokens: torch.Tensor | None = None,
        relation_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """共享特征提取，返回最后一个 token 的 embedding (B, d)。"""
        B, T, _ = dynamics.shape
        d = self.config.d_model

        dynamics = self._normalize_dynamics(dynamics)
        dyn_feat = self.dynamics_proj(dynamics)

        patches_flat = patches.reshape(B * T, *patches.shape[2:])
        spat_feat = self.spatial_encoder(patches_flat)
        spat_feat = spat_feat.reshape(B, T, d)

        fused = torch.cat([dyn_feat, spat_feat], dim=-1)
        obs_embed = self.fusion(fused)
        obs_embed = obs_embed + self._sparse_residual(
            B * T, dynamics.device, dyn_feat.dtype,
            contour_tokens, contour_mask, relation_tokens, relation_mask,
        ).reshape(B, T, d)

        act_embed = self.action_proj(actions)

        seq = torch.zeros(B, 2 * T, d, device=dynamics.device)
        seq[:, 0::2, :] = obs_embed
        seq[:, 1::2, :] = act_embed

        positions = torch.arange(2 * T, device=dynamics.device)
        seq = seq + self.pos_embed(positions).unsqueeze(0)

        attn_mask = build_attn_mask(
            2 * T, valid_mask, self.config.nhead, dynamics.device
        )
        out = self.transformer(seq, mask=attn_mask)

        return out[:, -1, :]

    # ── tanh squashing helpers ──

    def _squash_action(self, raw: torch.Tensor) -> torch.Tensor:
        """raw → tanh(raw) * scale"""
        return torch.tanh(raw) * self._action_scale

    def _log_prob_squash(
        self, log_prob_raw: torch.Tensor, raw_action: torch.Tensor
    ) -> torch.Tensor:
        """修正 tanh 变换的 log_prob: log π(a) = log π_raw(u) - Σ log(1 - tanh²(u))"""
        # log(1-tanh(u)^2) = 2*(log(2)-u-softplus(-2u)); finite even at saturation.
        correction = 2.0 * (math.log(2.0) - raw_action - F.softplus(-2.0 * raw_action))
        correction = correction + math.log(self._action_scale)
        return log_prob_raw - correction.sum(dim=-1)

    # ── public API ──

    def contact_mask(self, dynamics: torch.Tensor) -> torch.Tensor:
        # gate 必须读原始 0/1，不能读经 dyn_mean/std 归一化后的数值。
        return dynamics[:, -1, BASE_DYNAMICS_DIM] >= self.config.contact_gate_threshold

    def airborne_mask(self, dynamics: torch.Tensor) -> torch.Tensor:
        """完全无接触（tip/body/pot 全空）为 True。读原始 0/1 接触信号（同 contact_mask）。

        接触特征列：BASE_DYNAMICS_DIM + [0]=tip_contact, [2]=body_contact, [3]=pot_contact。
        """
        thr = self.config.contact_gate_threshold
        c = dynamics[:, -1, BASE_DYNAMICS_DIM:BASE_DYNAMICS_DIM + CONTACT_DIM]
        in_contact = (c[..., 0] >= thr) | (c[..., 2] >= thr) | (c[..., 3] >= thr)
        return ~in_contact

    # ── σ 调度（M2 修复）──
    def reset_log_std(self, log_std: float) -> None:
        """--init-from 时把探索噪声重设到设计初值（PPO→PPO 暖启动会原样继承源
        checkpoint 的 log_std，实测 σ≈0.52 钉死全程）。--resume 不调此函数。"""
        with torch.no_grad():
            self.actor_log_std.data.fill_(log_std)
            contact = getattr(self, "actor_log_std_contact", None)
            if contact is not None:
                contact.data.fill_(log_std + self.config.contact_log_std_offset)

    def anneal_log_std(self, rate: float, floor: float) -> None:
        """每轮训练后 log_std 线性递减（巩固进展 → 探索退火），下限 floor。"""
        with torch.no_grad():
            self.actor_log_std.data.sub_(rate).clamp_(min=floor)
            contact = getattr(self, "actor_log_std_contact", None)
            if contact is not None:
                contact.data.sub_(rate).clamp_(min=floor + self.config.contact_log_std_offset)

    def _actor_parameters(self, feat, dynamics):
        mean = self.actor_mean(feat)
        log_std = self.actor_log_std.expand_as(mean)
        if self.config.contact_heads_enabled:
            gate = self.contact_mask(dynamics).unsqueeze(-1)
            mean = torch.where(gate, self.actor_mean_contact(feat), mean)
            log_std = torch.where(gate, self.actor_log_std_contact.expand_as(mean), log_std)
        if self.config.phase_noise_enabled:
            # 空中相放大探索噪声（弹道相的"安全探索"：噪声只改变锤子预定位，
            # 不扰动当前轨迹；接触相保持基准——其噪声是发现引擎，见 config 注释）。
            # std 是观测的确定性函数，采集/更新同口径，PPO 比率自洽。
            bonus = (self.airborne_mask(dynamics).unsqueeze(-1)
                     * self.config.phase_noise_airborne_log_bonus)
            log_std = log_std + bonus.to(log_std.dtype)
        std = (log_std + self.exploration_boost).clamp(
            self.config.ppo_log_std_min, self.config.ppo_log_std_max,
        ).exp()
        return mean, std

    def forward(
        self,
        dynamics: torch.Tensor,
        patches: torch.Tensor,
        actions: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        contour_tokens: torch.Tensor | None = None,
        contour_mask: torch.Tensor | None = None,
        relation_tokens: torch.Tensor | None = None,
        relation_mask: torch.Tensor | None = None,
    ) -> tuple[Normal, torch.Tensor]:
        """
        返回 (distribution_on_raw_space, value)。

        注意：返回的 Normal 分布是 squash 之前的。
        调用方需通过 get_action_and_value / evaluate_actions 获取修正后的量。
        """
        feat = self._backbone(
            dynamics, patches, actions, valid_mask,
            contour_tokens, contour_mask, relation_tokens, relation_mask,
        )
        mean, std = self._actor_parameters(feat, dynamics)
        dist = Normal(mean, std)
        value = self.critic_head(feat).squeeze(-1)
        return dist, value

    @staticmethod
    def _ou_conditional(
        mean: torch.Tensor, std: torch.Tensor,
        noise_prev: torch.Tensor | None, noise_phi: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """条件高斯参数 (μ_cond, σ_cond)：raw_t | z_{t-1} ~ N(mean+std·φ·z_{t-1}, std²·(1-φ²))。

        noise_phi==0.0 或 noise_prev is None 时退化为 iid 高斯（μ_cond=mean, σ_cond=std）。
        """
        if not 0.0 <= noise_phi < 1.0:
            raise ValueError("noise_phi must be in [0, 1)")
        if noise_phi == 0.0 or noise_prev is None:
            return mean, std
        # 允许 noise_prev broadcast 到 mean 的 shape：采集 (1, action_dim)、评估 (B, action_dim)
        z_prev = noise_prev.to(mean.dtype)
        mu_cond = mean + std * noise_phi * z_prev
        # sqrt(1 - φ²)：φ∈[0,1)，加 eps 防边界数值问题
        scale = float((1.0 - noise_phi * noise_phi) ** 0.5)
        sigma_cond = std * scale
        return mu_cond, sigma_cond

    def get_action_and_value(
        self,
        dynamics: torch.Tensor,
        patches: torch.Tensor,
        actions: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        noise_prev: torch.Tensor | None = None,
        noise_phi: float = 0.0,
        deterministic: bool = False,
        contour_tokens: torch.Tensor | None = None,
        contour_mask: torch.Tensor | None = None,
        relation_tokens: torch.Tensor | None = None,
        relation_mask: torch.Tensor | None = None,
    ) -> ActionValue:
        """
        采集用：从策略中采样动作，返回 (squashed_action, log_prob, value, noise=z_t)。

        当 noise_phi>0 且 noise_prev 提供时：走 OU/AR(1) 条件采样，
          z_t = φ·z_{t-1} + √(1-φ²)·η_t, η~N(0,1)
          raw = mean + std·z_t
          log_prob 按 raw|z_{t-1} 的条件高斯计算（PPO ratio 数学自洽）。
        否则退化为 iid 高斯（原行为）。
        返回 ActionValue.noise = z_t（供调用方作为下一步的 noise_prev）。
        """
        dist, value = self.forward(
            dynamics, patches, actions, valid_mask,
            contour_tokens, contour_mask, relation_tokens, relation_mask,
        )
        mean = dist.mean
        std = dist.stddev

        mu_cond, sigma_cond = self._ou_conditional(mean, std, noise_prev, noise_phi)
        cond_dist = Normal(mu_cond, sigma_cond)

        raw_action = mean if deterministic else cond_dist.rsample()
        # 归一化噪声 z_t = (raw - mean) / std（iid 路径下 = η_t，仍是 N(0,1)），
        # 交回调用方作为下一步 noise_prev；det 化以脱离计算图。
        with torch.no_grad():
            z_t = ((raw_action - mean) / std.clamp(min=1e-8)).detach()

        log_prob_raw = cond_dist.log_prob(raw_action).sum(dim=-1)
        log_prob = self._log_prob_squash(log_prob_raw, raw_action)
        action = self._squash_action(raw_action)
        return ActionValue(action=action, log_prob=log_prob, value=value, noise=z_t,
                           raw_action=raw_action.detach())

    def evaluate_actions(
        self,
        dynamics: torch.Tensor,
        patches: torch.Tensor,
        act_history: torch.Tensor,
        taken_actions: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        noise_prev: torch.Tensor | None = None,
        noise_phi: float = 0.0,
        raw_actions: torch.Tensor | None = None,
        contour_tokens: torch.Tensor | None = None,
        contour_mask: torch.Tensor | None = None,
        relation_tokens: torch.Tensor | None = None,
        relation_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        训练用：对已执行的动作重新计算 log_prob、entropy、value。

        taken_actions: (B, action_dim) — squashed 空间中的动作
        noise_prev: (B, action_dim) 采集时的 z_{t-1}；OU 模式必需，iid 模式忽略。
        noise_phi: OU AR(1) 系数，0.0 → 退化 iid。
        returns: (log_prob, entropy, value)。log_prob 与 entropy 按条件高斯口径，
        与采集时一致 → 保证 ratio = exp(new_lp - old_lp) 无系统偏差。
        """
        dist, value = self.forward(
            dynamics, patches, act_history, valid_mask,
            contour_tokens, contour_mask, relation_tokens, relation_mask,
        )
        mean = dist.mean
        std = dist.stddev

        mu_cond, sigma_cond = self._ou_conditional(mean, std, noise_prev, noise_phi)
        cond_dist = Normal(mu_cond, sigma_cond)

        # 反 squash: squashed → raw
        # tanh 饱和后无法反演原采样。在线 buffer 保留 raw，避免未更新时 ratio 已偏离 1。
        if raw_actions is None:
            eps = torch.finfo(taken_actions.dtype).eps
            raw_action = torch.atanh((taken_actions / self._action_scale).clamp(-1 + eps, 1 - eps))
        else:
            raw_action = raw_actions

        log_prob_raw = cond_dist.log_prob(raw_action).sum(dim=-1)
        log_prob = self._log_prob_squash(log_prob_raw, raw_action)
        entropy = cond_dist.entropy().sum(dim=-1)

        return log_prob, entropy, value

    @torch.no_grad()
    def predict_deterministic(
        self,
        dynamics: np.ndarray,
        patches: np.ndarray,
        action_history: np.ndarray,
        valid_mask: np.ndarray | None = None,
        contour_tokens: np.ndarray | None = None,
        contour_mask: np.ndarray | None = None,
        relation_tokens: np.ndarray | None = None,
        relation_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        确定性推理（评估用）：直接用 mean，不采样。

        dynamics: (T, state_dim) 或 (1, T, state_dim)
        patches: (T, C, H, H) 或 (1, T, C, H, H)
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

        def _opt(arr, min_ndim):
            if arr is None:
                return None
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == min_ndim:
                arr = arr[np.newaxis]
            return torch.from_numpy(arr).to(device)

        feat = self._backbone(
            dyn_t, pat_t, act_t, vm_t,
            _opt(contour_tokens, 3), _opt(contour_mask, 2),
            _opt(relation_tokens, 3), _opt(relation_mask, 2),
        )
        mean, _ = self._actor_parameters(feat, dyn_t)
        action = self._squash_action(mean)
        return action.cpu().numpy().squeeze(0)
