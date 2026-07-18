"""
PPO Trainer — Proximal Policy Optimization 训练器。

实现 clipped surrogate objective + value loss + entropy bonus，
支持多 epoch minibatch 更新和可选的 KL 散度早停。

包含运行时自适应:
- target_kl: 带硬边界 [0.01, 0.05] 的比例式自适应
- max_grad_norm: 前 N 步收集梯度范数 p95 后冻结
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

if TYPE_CHECKING:
    from .actor_critic import ActorCritic
    from .config import TrainConfig
    from .ppo_buffer import PPORolloutBuffer

logger = logging.getLogger(__name__)


class _KLAdapter:
    """target_kl 自适应: 比例式调节 + 硬边界 clamp。"""

    KL_MIN = 0.01
    KL_MAX = 0.05
    ADAPT_RATE = 0.1

    def __init__(self, initial_kl: float | None):
        if initial_kl is None:
            self.target: float | None = None
            return
        self.target = float(np.clip(initial_kl, self.KL_MIN, self.KL_MAX))

    def adapt(self, actual_kl: float) -> None:
        if self.target is None:
            return
        ratio = actual_kl / max(self.target, 1e-8)
        if ratio > 1.5:
            self.target *= (1 - self.ADAPT_RATE)
        elif ratio < 0.5:
            self.target *= (1 + self.ADAPT_RATE)
        self.target = float(np.clip(self.target, self.KL_MIN, self.KL_MAX))


class _GradNormAdapter:
    """max_grad_norm 自适应: warmup 阶段收集 p95 后冻结，带硬上限防止恢复训练时梯度爆炸。"""

    def __init__(self, default_norm: float, cap: float, warmup_updates: int = 500):
        self._default = default_norm
        self._cap = cap
        self._warmup = warmup_updates
        self._norms: list[float] = []
        self._frozen_norm: float | None = None

    def record(self, grad_norm: float) -> None:
        if self._frozen_norm is not None:
            return
        self._norms.append(grad_norm)
        if len(self._norms) >= self._warmup:
            p95 = float(np.percentile(self._norms, 95))
            adaptive = max(p95 * 1.5, 0.1)
            self._frozen_norm = min(adaptive, self._cap)
            logger.info(
                "GradNormAdapter frozen: p95=%.4f, adaptive=%.4f, cap=%.4f → max_grad_norm=%.4f",
                p95, adaptive, self._cap, self._frozen_norm,
            )

    @property
    def max_grad_norm(self) -> float:
        return self._frozen_norm if self._frozen_norm is not None else self._default


class PPOTrainer:
    """PPO 训练器，对一轮 rollout buffer 做多 epoch minibatch 更新。"""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.optimizer: AdamW | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._global_step = 0
        self._pending_optim_state: dict | None = None
        self._kl_adapter = _KLAdapter(config.target_kl)
        self._grad_norm_adapter = _GradNormAdapter(config.max_grad_norm, config.max_grad_norm_cap)

    def set_pending_optimizer_state(self, state_dict: dict) -> None:
        """暂存 optimizer state_dict，在首次 update 创建 optimizer 后应用。"""
        self._pending_optim_state = state_dict

    def update(
        self,
        model: ActorCritic,
        buffer: PPORolloutBuffer,
    ) -> dict[str, float]:
        """
        PPO 更新。

        返回本轮的平均 metrics:
          policy_loss, value_loss, entropy, approx_kl, clip_fraction, total_loss
        """
        model = model.to(self.device)
        model.train()
        buffer.device = self.device

        if self.optimizer is None:
            self.optimizer = AdamW(
                model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
            if self._pending_optim_state is not None:
                self.optimizer.load_state_dict(self._pending_optim_state)
                logger.info("已恢复 optimizer state")
                self._pending_optim_state = None

        cfg = self.config
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_frac = 0.0
        total_loss_sum = 0.0
        n_updates = 0
        early_stopped = False

        for epoch in range(cfg.ppo_epochs):
            if early_stopped:
                break

            for batch in buffer.get_batches(
                cfg.ppo_batch_size,
                normalize_advantages=cfg.normalize_advantages,
            ):
                new_log_prob, entropy, new_value = model.evaluate_actions(
                    batch.dynamics,
                    batch.patches,
                    batch.act_history,
                    batch.actions,
                    valid_mask=batch.valid_mask,
                )

                # ── 策略损失 (clipped surrogate) ──
                log_ratio = new_log_prob - batch.old_log_probs
                ratio = log_ratio.exp()

                surr1 = ratio * batch.advantages
                surr2 = (
                    ratio.clamp(1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon)
                    * batch.advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # ── 价值损失 (clipped) ──
                value_pred_clipped = batch.old_values + (
                    new_value - batch.old_values
                ).clamp(-cfg.clip_epsilon, cfg.clip_epsilon)
                vl_unclipped = (new_value - batch.returns).pow(2)
                vl_clipped = (value_pred_clipped - batch.returns).pow(2)
                value_loss = 0.5 * torch.max(vl_unclipped, vl_clipped).mean()

                # ── 熵奖励 ──
                entropy_mean = entropy.mean()
                entropy_loss = -entropy_mean

                # ── 总损失 ──
                loss = (
                    policy_loss
                    + cfg.vf_coef * value_loss
                    + cfg.ent_coef * entropy_loss
                )

                if not torch.isfinite(loss):
                    logger.warning(
                        "NaN/Inf loss detected (step=%d), skipping update",
                        self._global_step,
                    )
                    self.optimizer.zero_grad()
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                effective_grad_norm = self._grad_norm_adapter.max_grad_norm
                total_norm = clip_grad_norm_(model.parameters(), effective_grad_norm)
                self._grad_norm_adapter.record(total_norm.item())
                self.optimizer.step()

                if any(
                    not torch.isfinite(p).all()
                    for p in model.parameters()
                    if p.grad is not None
                ):
                    logger.warning(
                        "NaN/Inf weights after optimizer step (step=%d), "
                        "rolling back is not possible — training may be corrupted",
                        self._global_step,
                    )

                # ── 统计 ──
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    clip_frac = (
                        (ratio - 1.0).abs().gt(cfg.clip_epsilon).float().mean().item()
                    )

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_mean.item()
                total_approx_kl += approx_kl
                total_clip_frac += clip_frac
                total_loss_sum += loss.item()
                n_updates += 1
                self._global_step += 1

            # KL 早停 (使用自适应 target)
            effective_kl = self._kl_adapter.target
            if effective_kl is not None and n_updates > 0:
                avg_kl = total_approx_kl / n_updates
                if avg_kl > effective_kl:
                    logger.info(
                        "KL 早停: epoch %d/%d, avg_kl=%.4f > target_kl=%.4f",
                        epoch + 1, cfg.ppo_epochs, avg_kl, effective_kl,
                    )
                    early_stopped = True

        n = max(n_updates, 1)
        final_avg_kl = total_approx_kl / n

        # 迭代结束后自适应 target_kl
        self._kl_adapter.adapt(final_avg_kl)

        metrics = {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "approx_kl": final_avg_kl,
            "clip_fraction": total_clip_frac / n,
            "total_loss": total_loss_sum / n,
            "n_updates": float(n_updates),
            "target_kl": float(self._kl_adapter.target or 0),
            "max_grad_norm": self._grad_norm_adapter.max_grad_norm,
        }
        return metrics
