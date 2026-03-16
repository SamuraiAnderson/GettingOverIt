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
    env_state_dim: int = 29
    state_dim: int = 16          # 29D 去掉 12D 位置坐标 + timestamp
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
    max_grad_norm: float = 1.0
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

    # ── 投放点随机化 ──
    drop_height: float = 2.0        # 表面上方投放高度
    surface_padding: float = 0.3    # Player 包围盒高度之上的间隙
    y_max_cutoff: float = 380.0     # 屏蔽高于此值的表面

    # ── 安全 ──
    water_y_threshold: float = -10.0  # y 低于此值视为落水，轨迹废弃

    # ── 探索 ──
    explore_noise_std: float = 0.1
    noise_decay: float = 0.95

    # ── 训练控制 ──
    max_iterations: int = 100
    save_interval: int = 5
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
