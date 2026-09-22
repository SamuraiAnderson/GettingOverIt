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


class StalePolicyError(ValueError):
    """A correctly recorded rollout is too far from the learner to consume."""


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


class _WeightedWindowView:
    """给 TrajectoryDataset 的窗口样本追加逐窗口权重（P3.2 优势加权 SIL 用）。

    包装而非修改 __getitem__ 的返回元数：BC 训练器（trainer.py）与旧 MSE 路径
    仍消费五元组，只有加权 SIL 路径解包六元组。weights 与 dataset 窗口索引顺序对齐。
    """

    def __init__(self, base, weights: torch.Tensor) -> None:
        self.base = base
        self.weights = weights
        # sil_update 读 dataset.trajectories 记指标；透传保持兼容
        self.trajectories = base.trajectories

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        return (*self.base[idx], self.weights[idx])


class PPOTrainer:
    """PPO 训练器，对一轮 rollout buffer 做多 epoch minibatch 更新。"""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.optimizer: AdamW | None = None
        self.critic_optimizer: AdamW | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._global_step = 0
        self._pending_optim_state: dict | None = None
        self._kl_adapter = _KLAdapter(config.target_kl)
        self._grad_norm_adapter = _GradNormAdapter(config.max_grad_norm, config.max_grad_norm_cap)

    def set_pending_optimizer_state(self, state_dict: dict) -> None:
        """暂存 optimizer state_dict，在首次 update 创建 optimizer 后应用。"""
        self._pending_optim_state = state_dict

    def update_critic_only(
        self,
        model: ActorCritic,
        buffer: PPORolloutBuffer,
    ) -> dict[str, float]:
        """
        Critic 预热：冻结 backbone + actor，只拟合 critic_head。

        目的：BC 迁移后 critic_head 是随机初始化，若直接做 PPO，优势值来自随机价值函数会把
        策略推爆。先用独立优化器（仅含 critic_head 参数，避免 AdamW 权重衰减污染冻结的 BC 权重）
        在冻结的 BC 特征上拟合价值，使优势值有意义后再开策略更新。
        """
        model = model.to(self.device)
        model.eval()  # 关 dropout：保证与采集/PPO 更新的 log_prob 口径一致（见 update 注释）
        buffer.device = self.device

        if self.critic_optimizer is None:
            self.critic_optimizer = AdamW(
                model.critic_head.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )

        from .policy_audit import audit_policy
        if buffer.policy_context.get("deterministic"):
            raise ValueError("Evaluation rollouts cannot be used for PPO training")
        audit = audit_policy(model, buffer, max_samples=128)
        cfg = self.config
        total_value_loss = 0.0
        n_updates = 0
        for _epoch in range(cfg.value_warmup_epochs):
            from .decision_buffer import evaluated_batches
            for batch, _, _, new_value in evaluated_batches(
                model, buffer, cfg.ppo_batch_size, normalize_advantages=False,
            ):
                value_loss = 0.5 * (new_value - batch.returns).pow(2).mean()
                if not torch.isfinite(value_loss):
                    model.zero_grad(set_to_none=True)
                    continue
                model.zero_grad(set_to_none=True)  # 清空全部梯度（backbone 梯度不使用）
                value_loss.backward()
                clip_grad_norm_(
                    model.critic_head.parameters(), self.config.max_grad_norm_cap
                )
                self.critic_optimizer.step()
                total_value_loss += value_loss.item()
                n_updates += 1

        n = max(n_updates, 1)
        return {
            "policy_loss": 0.0,
            "value_loss": total_value_loss / n,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "total_loss": total_value_loss / n,
            "n_updates": float(n_updates),
            "target_kl": float(self._kl_adapter.target or 0),
            "max_grad_norm": self._grad_norm_adapter.max_grad_norm,
            "value_only": 1.0,
            **audit,
        }

    def update(
        self,
        model: ActorCritic,
        buffer: PPORolloutBuffer,
        *,
        behavior_model: ActorCritic | None = None,
        max_behavior_kl: float = 0.01,
    ) -> dict[str, float]:
        """
        PPO 更新。

        返回本轮的平均 metrics:
          policy_loss, value_loss, entropy, approx_kl, clip_fraction, total_loss
        """
        model = model.to(self.device)
        # 关 dropout（eval 不影响梯度）：PPO 的 ratio 依赖 old/new log_prob 在相同权重下一致，
        # dropout 的随机 mask 会破坏这一点，使 ratio≠1、approx_kl 虚高、策略被噪声推爆。
        model.eval()
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

        from .policy_audit import audit_policy, policy_replay_metrics
        if buffer.policy_context.get("deterministic"):
            raise ValueError("Evaluation rollouts cannot be used for PPO training")
        audit = audit_policy(model if behavior_model is None else behavior_model,
                             buffer, max_samples=128)
        lag = 0.0
        if behavior_model is not None:
            if not np.isfinite(max_behavior_kl) or max_behavior_kl <= 0:
                raise ValueError("max_behavior_kl must be finite and positive")
            drift = policy_replay_metrics(model, buffer, max_samples=512)
            lag = drift["preupdate_kl"]
            if lag > max_behavior_kl:
                raise StalePolicyError(f"Behavior KL {lag:.6g} exceeds {max_behavior_kl:.6g}")
        cfg = self.config
        # OU 探索：ratio 数学自洽要求 new/old log_prob 用同一条件式（见 actor_critic.evaluate_actions），
        # 即 raw|z_{t-1} ~ N(mean+std·φ·z_{t-1}, std²·(1-φ²))。ou_enabled=False → phi=0 退化 iid。
        # evaluated_batches reads the archived OU condition for every decision.

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_frac = 0.0
        total_loss_sum = 0.0
        n_updates = 0
        early_stopped = False
        self.last_update_trace = []
        stop_kl = 0.0

        for epoch in range(cfg.ppo_epochs):
            if early_stopped:
                break

            from .decision_buffer import evaluated_batches
            for batch, new_log_prob, entropy, new_value in evaluated_batches(
                model, buffer, cfg.ppo_batch_size,
                normalize_advantages=cfg.normalize_advantages,
            ):
                # ── 策略损失 (clipped surrogate) ──
                log_ratio = new_log_prob - batch.old_log_probs
                ratio = log_ratio.exp()

                # These probabilities describe the CURRENT parameters, before
                # optimizer.step(). Stop before applying another excessive step.
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                effective_kl = self._kl_adapter.target
                if not np.isfinite(approx_kl):
                    raise ValueError("Non-finite minibatch KL")
                trace = {"epoch": epoch, "pre_step_kl": approx_kl,
                         "samples": len(batch.actions), "applied": False}
                self.last_update_trace.append(trace)
                if effective_kl is not None and approx_kl > cfg.ppo_kl_stop_factor * effective_kl:
                    early_stopped, stop_kl = True, approx_kl
                    logger.info("KL early stop BEFORE update %d: %.6f > %.6f",
                                n_updates + 1, approx_kl, cfg.ppo_kl_stop_factor * effective_kl)
                    break

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
                trace.update(applied=True, policy_loss=policy_loss.item(),
                             value_loss=value_loss.item(), grad_norm=float(total_norm))

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

        n = max(n_updates, 1)
        final_avg_kl = total_approx_kl / n

        # 迭代结束后自适应 target_kl
        self._kl_adapter.adapt(max(final_avg_kl, stop_kl))

        metrics = {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "approx_kl": final_avg_kl,
            "clip_fraction": total_clip_frac / n,
            "total_loss": total_loss_sum / n,
            "n_updates": float(n_updates),
            "early_stopped": float(early_stopped),
            "stop_kl": stop_kl,
            "behavior_kl": lag,
            "target_kl": float(self._kl_adapter.target or 0),
            "max_grad_norm": self._grad_norm_adapter.max_grad_norm,
            **audit,
        }
        return metrics

    @torch.no_grad()
    def _sil_advantage_weights(
        self, model: ActorCritic, dataset
    ) -> tuple[np.ndarray, float]:
        """P3.2 第一遍前向：用当前 critic 为全池窗口打 V，算 MC 优势并映射为窗口权重。

        A_tau = R_tau − V(s_tau)，其中 R_tau = Σ_k γ^k·credited[tau+k] 沿精英轨迹的
        折扣未来 credited 进度，并在该轨迹最后一个索引窗口处用 V 自举
        （R_full(τ) = R(τ) + γ^Δ·(V(τ_last) − R(τ_last))）：V 是无限时域口径
        （PPO 非终止截断自举训练），不自举的 R 比 V 短一个视界，A 会被系统性压负、
        frontier 窗口受害最重。剪枝后 cap 之外 credited≡0，故无需 GAE 的 λ 折中。
        A 在全池窗口上归一化（均值 0 方差 1）后 clamp 到 [sil_advantage_floor,
        sil_advantage_cap]：正优势强化（至多 cap 倍），非正保留 floor 弱权重、不删除动作。

        返回 (weights, pos_frac)：weights 为与 dataset 窗口索引顺序对齐的 float32 数组；
        pos_frac 为归一化后正优势窗口比例（诊断用）。
        """
        from torch.utils.data import DataLoader

        cfg = self.config
        n = len(dataset)
        values = np.empty(n, dtype=np.float64)
        loader = DataLoader(
            dataset, batch_size=256, shuffle=False,
            num_workers=cfg.sil_num_workers,
        )
        offset = 0
        for dyn, pat, act_hist, vm, _target in loader:
            dyn = dyn.to(self.device)
            pat = pat.to(self.device)
            act_hist = act_hist.to(self.device)
            vm = vm.to(self.device)
            _, value = model.forward(dyn, pat, act_hist, valid_mask=vm)
            b = value.shape[0]
            values[offset:offset + b] = value.detach().cpu().numpy().reshape(-1)
            offset += b

        # 逐轨迹折扣未来 credited（逆序累加，O(T)），并在每条轨迹的最后一个索引窗口
        # 处用 V 自举：V 是无限时域口径（PPO 非终止截断自举训练），R 若不补
        # γ^Δ·V(末端) 会比 V 短一个视界，A 被系统性压负（frontier 窗口受害最重）。
        traj_returns: list[np.ndarray] = []
        traj_last_value: dict[int, tuple[int, float]] = {}  # ti -> (最大 tau, 其 V)
        for idx, (ti, tau) in enumerate(dataset._index):
            prev = traj_last_value.get(ti)
            if prev is None or tau > prev[0]:
                traj_last_value[ti] = (tau, values[idx])
        for ti, traj in enumerate(dataset.trajectories):
            credited = np.asarray(traj.corridor_progress, dtype=np.float64)
            R = np.zeros(len(credited), dtype=np.float64)
            run = 0.0
            for t in range(len(credited) - 1, -1, -1):
                run = credited[t] + cfg.gamma * run
                R[t] = run
            traj_returns.append(R)

        # dataset._index 为 (traj_idx, tau) 列表，与 __getitem__ 同序（同包内访问）
        # 自举的正确形式：R_full(tau) = R(tau) + γ^Δ·(V(tau_last) − R(tau_last))，
        # 即用 V 替换 tau_last 处的不完整尾部回报，避免重复计入 R(tau_last)。
        adv = np.empty(n, dtype=np.float64)
        for idx, (ti, tau) in enumerate(dataset._index):
            last_tau, last_v = traj_last_value[ti]
            ret = (traj_returns[ti][tau]
                   + (cfg.gamma ** (last_tau - tau)) * (last_v - traj_returns[ti][last_tau]))
            adv[idx] = ret - values[idx]

        pos_frac = float(np.mean(adv > 0.0))
        std = float(adv.std())
        if std < 1e-8:
            return np.ones(n, dtype=np.float32), pos_frac
        adv = (adv - float(adv.mean())) / (std + 1e-8)
        weights = np.clip(adv, cfg.sil_advantage_floor, cfg.sil_advantage_cap)
        return weights.astype(np.float32), pos_frac

    def sil_update(
        self, model: ActorCritic, dataset, loss_coef: float | None = None
    ) -> dict[str, float]:
        """
        Self-Imitation：对精英轨迹池做 BC 监督，把 actor 均值锚向历史最优动作。

        两种损失形态：
          - sil_advantage_weighting_enabled 且全池为走廊轨迹（P3.2 真 SIL）：
            加权高斯 NLL（raw 空间，σ detach），窗口权重 = clamp(归一化 MC 优势,
            floor, cap)。梯度 w.r.t. mean = w·(μ−u)/σ²，饱和处不消失（优于 MSE），
            且 σ 不参与梯度 → 不碰 actor_log_std、不压制探索。
          - 否则（旧路径）：MSE-on-mean（squash 后均值 vs 精英动作）。

        在 update() 之后调用，复用同一 optimizer。dataset 为 TrajectoryDataset，
        __getitem__ 产出 (dynamics, patches, input_actions, valid_mask, target_action)。
        """
        from torch.utils.data import DataLoader

        cfg = self.config
        coef = cfg.sil_loss_coef if loss_coef is None else loss_coef
        if len(dataset) == 0:
            return {}

        model = model.to(self.device)
        model.eval()  # 关 dropout：与 PPO 采集/更新口径一致

        if self.optimizer is None:
            # SIL 正常在 update() 之后调用（optimizer 已建）；兜底以防单独调用。
            self.optimizer = AdamW(
                model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )

        # P3.2：优势加权权重（仅走廊模式；含非走廊轨迹时回退旧 MSE 路径）
        weights: torch.Tensor | None = None
        adv_pos_frac = float("nan")
        if getattr(cfg, "sil_advantage_weighting_enabled", False):
            if all(getattr(t, "corridor_progress", None) is not None
                   for t in dataset.trajectories):
                w_np, adv_pos_frac = self._sil_advantage_weights(model, dataset)
                weights = torch.from_numpy(w_np)
            elif not getattr(self, "_sil_adv_warned", False):
                logger.warning(
                    "sil_advantage_weighting 已启用但池内含非走廊轨迹，"
                    "本路径回退 MSE-on-mean（高度模式暂无 corridor_progress 支撑）"
                )
                self._sil_adv_warned = True

        train_view = _WeightedWindowView(dataset, weights) if weights is not None else dataset
        loader = DataLoader(
            train_view,
            batch_size=cfg.sil_batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=cfg.sil_num_workers,
        )

        max_batches = cfg.sil_max_batches if cfg.sil_max_batches > 0 else None
        total_loss = 0.0
        n_updates = 0
        stop = False
        for _epoch in range(cfg.sil_epochs):
            if stop:
                break
            for batch in loader:
                if max_batches is not None and n_updates >= max_batches:
                    stop = True
                    break
                if weights is not None:
                    dyn, pat, act_hist, vm, target, w = batch
                    w = w.to(self.device)
                else:
                    dyn, pat, act_hist, vm, target = batch
                dyn = dyn.to(self.device)
                pat = pat.to(self.device)
                act_hist = act_hist.to(self.device)
                vm = vm.to(self.device)
                target = target.to(self.device)

                dist, _ = model.forward(dyn, pat, act_hist, valid_mask=vm)
                if weights is not None:
                    # 加权高斯 NLL（raw 空间）：-log π(u|s)·w 的 mean 梯度 =
                    # w·(μ−u)/σ²；tanh 的 Jacobian 项与 μ 无关、σ detach → 只锚均值。
                    u_target = torch.atanh(
                        (target / model._action_scale).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
                    )
                    sigma = dist.stddev.detach().clamp_min(1e-6)
                    per = 0.5 * (((dist.mean - u_target) / sigma) ** 2).sum(dim=-1)
                    loss = coef * (w * per).sum() / w.sum().clamp_min(1e-8)
                else:
                    mean_sq = torch.tanh(dist.mean) * model._action_scale
                    loss = coef * F.mse_loss(mean_sq, target)

                if not torch.isfinite(loss):
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm_(
                    model.parameters(), self._grad_norm_adapter.max_grad_norm
                )
                self.optimizer.step()

                total_loss += loss.item()
                n_updates += 1

        metrics = {
            "sil_loss": total_loss / max(n_updates, 1),
            "sil_trajs": float(len(dataset.trajectories)),
            "sil_windows": float(len(dataset)),
            "sil_updates": float(n_updates),
        }
        if weights is not None:
            metrics["sil_adv_pos_frac"] = adv_pos_frac
            metrics["sil_weight_mean"] = float(weights.mean())
        return metrics
