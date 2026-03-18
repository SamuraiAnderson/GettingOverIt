"""
PPO Trainer — Proximal Policy Optimization 训练器。

实现 clipped surrogate objective + value loss + entropy bonus，
支持多 epoch minibatch 更新和可选的 KL 散度早停。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

if TYPE_CHECKING:
    from .actor_critic import ActorCritic
    from .config import TrainConfig
    from .ppo_buffer import PPORolloutBuffer

logger = logging.getLogger(__name__)


class PPOTrainer:
    """PPO 训练器，对一轮 rollout buffer 做多 epoch minibatch 更新。"""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.optimizer: AdamW | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._global_step = 0
        self._pending_optim_state: dict | None = None

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

                # ── 价值损失 ──
                value_loss = F.mse_loss(new_value, batch.returns)

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
                clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
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

            # KL 早停
            if cfg.target_kl is not None and n_updates > 0:
                avg_kl = total_approx_kl / n_updates
                if avg_kl > cfg.target_kl:
                    logger.info(
                        "KL 早停: epoch %d/%d, avg_kl=%.4f > target_kl=%.4f",
                        epoch + 1, cfg.ppo_epochs, avg_kl, cfg.target_kl,
                    )
                    early_stopped = True

        n = max(n_updates, 1)
        metrics = {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "approx_kl": total_approx_kl / n,
            "clip_fraction": total_clip_frac / n,
            "total_loss": total_loss_sum / n,
            "n_updates": float(n_updates),
        }
        return metrics
