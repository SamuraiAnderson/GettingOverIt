"""验证 DYNAMICS_DIM 34→39 扩展与 build_dynamics 兼容层（含 body / pot 分开的 5 维接触）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from training.contact_features import CONTACT_DIM
from training.dataset import (
    BASE_DYNAMICS_DIM,
    DYNAMICS_DIM,
    Trajectory,
    build_dynamics,
    compute_dynamics_stats,
)


def test_dim_math() -> None:
    assert BASE_DYNAMICS_DIM == 34
    assert CONTACT_DIM == 5
    assert DYNAMICS_DIM == 39


def test_build_dynamics_no_contact() -> None:
    raw = np.random.randn(5, 33).astype(np.float32)
    out = build_dynamics(raw)
    assert out.shape == (5, DYNAMICS_DIM), f"shape={out.shape}"
    # 尾部 CONTACT_DIM 列应全 0（未提供 contact）
    assert np.all(out[:, -CONTACT_DIM:] == 0.0), "尾部接触维度应=0"


def test_build_dynamics_with_contact() -> None:
    raw = np.random.randn(5, 33).astype(np.float32)
    contact = np.random.rand(5, CONTACT_DIM).astype(np.float32)
    out = build_dynamics(raw, contact)
    assert out.shape == (5, DYNAMICS_DIM)
    assert np.allclose(out[:, -CONTACT_DIM:], contact), "尾部维度应==contact"
    # 前 34 维应与无 contact 时一致（base 计算不受影响）
    out_base = build_dynamics(raw)
    assert np.allclose(out[:, :BASE_DYNAMICS_DIM], out_base[:, :BASE_DYNAMICS_DIM])


def test_build_dynamics_batched_shape() -> None:
    raw = np.random.randn(3, 5, 33).astype(np.float32)
    contact = np.random.rand(3, 5, CONTACT_DIM).astype(np.float32)
    out = build_dynamics(raw, contact)
    assert out.shape == (3, 5, DYNAMICS_DIM)


def test_stats_with_and_without_contact() -> None:
    T = 20
    t1 = Trajectory(
        raw_states=np.random.randn(T + 1, 33).astype(np.float32),
        actions=np.random.randn(T, 2).astype(np.float32),
        contact=np.random.rand(T + 1, CONTACT_DIM).astype(np.float32),
    )
    t2 = Trajectory(
        raw_states=np.random.randn(T + 1, 33).astype(np.float32),
        actions=np.random.randn(T, 2).astype(np.float32),
        contact=None,  # 缺 contact，尾部 5 维为 0
    )
    mean, std = compute_dynamics_stats([t1, t2])
    assert mean.shape == (DYNAMICS_DIM,) and std.shape == (DYNAMICS_DIM,)
    # 缺 contact 的轨迹会把接触维度往 0 拉；不严格测数值只测形状 + 非零 std（min-clip 兜底）
    assert np.all(std > 0.0), "std 不应有零（min-clip 1e-6→1.0）"


def main() -> None:
    test_dim_math()
    print("dim math OK (34+5=39)")
    test_build_dynamics_no_contact()
    print("build_dynamics(raw) 无 contact 零填充 OK")
    test_build_dynamics_with_contact()
    print("build_dynamics(raw, contact) 拼接 OK")
    test_build_dynamics_batched_shape()
    print("batched shape OK")
    test_stats_with_and_without_contact()
    print("compute_dynamics_stats 混合轨迹 OK")
    print("ALL DIM SMOKE OK")


if __name__ == "__main__":
    main()
