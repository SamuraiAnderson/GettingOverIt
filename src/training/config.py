"""训练超参数配置 — 集中管理所有可调参数。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "project.json"


def _default_game_root() -> str:
    try:
        with open(_PROJECT_CONFIG, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return str(Path(cfg["game"]["executable_path"]).parent)
    except Exception:
        return ""


def _default_port() -> int:
    try:
        with open(_PROJECT_CONFIG, "r", encoding="utf-8") as f:
            return json.load(f).get("tcp_port", 9000)
    except Exception:
        return 9000


@dataclass
class TrainConfig:
    # ── 环境 ──
    game_root: str = field(default_factory=_default_game_root)
    num_agents: int = 10
    port: int = field(default_factory=_default_port)
    warmup_steps: int = 100
    settle_steps: int = 30
    steps_per_rollout: int = 500

    # ── 模型 ──
    # build_dynamics 输出维度: 13 速度/角速度 + 3 角度×2(sin/cos) + 5 部件相对坐标×2 = 29
    state_dim: int = 29
    action_dim: int = 2
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    context_len: int = 32
    patch_size: int = 32
    patch_channels: int = 4
    patch_resolution: float = 0.5
    dropout: float = 0.1

    # ── 训练 ──
    lr: float = 1e-4
    epochs: int = 10
    batch_size: int = 256
    action_scale: float = 100.0
    max_grad_norm: float = 0.5
    max_grad_norm_cap: float = 5.0   # GradNormAdapter 冻结值的硬上限 (default * cap_ratio)
    weight_decay: float = 1e-4
    val_ratio: float = 0.1
    early_stop_patience: int = 3
    num_workers: int = 4

    # ── 数据管理 ──
    max_good_trajectories: int = 300
    warmup_rollouts: int = 50
    keep_ratio: float = 0.5
    rollouts_per_iteration: int = 3

    # ── 评分 ──
    efficiency_weight: float = 50.0
    waypoint_weight: float = 0.5
    abs_height_weight: float = 0.1

    # ── 效率图 ──
    grid_resolution: float = 1.0
    diffusion_iterations: int = 50
    diffusion_alpha: float = 0.2
    height_prior_weight: float = 0.05

    # ── 投放模式 ──
    random_deploy: bool = True     # True=随机投放到表面, False=所有 agent 留在初始位置
    drop_height: float = 2.0        # 表面上方投放高度
    surface_padding: float = 0.3    # Player 包围盒高度之上的间隙
    y_max_cutoff: float = 380.0     # 屏蔽高于此值的表面
    settle_drop_threshold: float = 5.0  # settle 后跌幅 > 此值的 agent 视为不稳定并跳过

    # ── 安全 ──
    water_y_threshold: float = -10.0  # y 低于此值视为落水，轨迹废弃

    # ── 探索 ──
    explore_noise_std: float = 0.1
    noise_decay: float = 0.95

    # ── PPO ──
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    ppo_epochs: int = 4
    vf_coef: float = 0.25
    ent_coef: float = 0.001
    target_kl: float | None = 0.03
    ppo_batch_size: int = 64
    normalize_advantages: bool = True

    # ── 非对称奖励缩放 ──
    neg_reward_scale: float = 0.1     # log 压缩系数: -scale * log(1 + |dy|)
    reward_alpha: float = 0.5        # step_reward 中 height vs waypoint 的无量纲比例 (α→height, 1-α→waypoint)

    # ── 训练控制 ──
    max_iterations: int = 100
    save_interval: int = 5
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
