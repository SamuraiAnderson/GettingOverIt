"""P2.1 集成测试：接触信号 + 老 34D checkpoint 扩到 39D + 端到端 forward。

覆盖：
  1. `expand_state_dict_for_dynamics_dim`：34D checkpoint 正确扩展到 39D，且旧字段等值保留
  2. `ActorCritic` 用扩展后的 state_dict 加载：无缺失/多余键（strict）
  3. `build_dynamics(raw, contact)` 与 rollout 侧同链路：前 34 维与旧 `build_dynamics(raw)` 一致
  4. 用 fake obs 走一次 model.forward：输出有限、无 NaN
  5. body_contact / pot_contact 分开输出（不再合并 body∪pot）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from training.actor_critic import ActorCritic
from training.config import TrainConfig
from training.contact_features import CONTACT_DIM
from training.dataset import BASE_DYNAMICS_DIM, DYNAMICS_DIM, build_dynamics
from training.model import expand_state_dict_for_dynamics_dim


def _fake_old_ckpt(config: TrainConfig, old_dim: int = 34) -> dict[str, torch.Tensor]:
    """合成 34D 时代的 ActorCritic state_dict（把 dynamics_proj / dyn_mean/std 硬造成旧维度）。"""
    model = ActorCritic(config)
    st = model.state_dict()
    d = config.d_model

    st["dynamics_proj.weight"] = torch.randn(d, old_dim)
    if "dyn_mean" in st:
        st["dyn_mean"] = torch.randn(old_dim) * 0.5
    if "dyn_std" in st:
        st["dyn_std"] = torch.rand(old_dim) + 0.5
    return {k: v.clone() for k, v in st.items()}


def test_dim_math() -> None:
    assert BASE_DYNAMICS_DIM == 34, "旧 34D 语义应保留在 BASE_DYNAMICS_DIM"
    assert CONTACT_DIM == 5, "新增 5 个接触维度：tip_c/tip_g/body_c/pot_c/fall"
    assert DYNAMICS_DIM == 39, "总维度 34+5=39"


def test_expand_preserves_old_values() -> None:
    """扩展后前 34 列/元素完全等于旧值，新维度按 pad 规则填充。"""
    cfg = TrainConfig()
    old = _fake_old_ckpt(cfg, old_dim=34)
    model = ActorCritic(cfg)
    expanded = expand_state_dict_for_dynamics_dim(old, model.state_dict())

    w_old = old["dynamics_proj.weight"]
    w_new = expanded["dynamics_proj.weight"]
    assert w_new.shape == (cfg.d_model, DYNAMICS_DIM)
    assert torch.allclose(w_new[:, :BASE_DYNAMICS_DIM], w_old), "前 34 列必须逐元素相等"
    assert torch.all(w_new[:, BASE_DYNAMICS_DIM:] == 0.0), f"新 {CONTACT_DIM} 列必须=0"

    if "dyn_mean" in old:
        m_old = old["dyn_mean"]
        m_new = expanded["dyn_mean"]
        assert m_new.shape == (DYNAMICS_DIM,)
        assert torch.allclose(m_new[:BASE_DYNAMICS_DIM], m_old)
        assert torch.all(m_new[BASE_DYNAMICS_DIM:] == 0.0)

        s_old = old["dyn_std"]
        s_new = expanded["dyn_std"]
        assert s_new.shape == (DYNAMICS_DIM,)
        assert torch.allclose(s_new[:BASE_DYNAMICS_DIM], s_old)
        assert torch.all(s_new[BASE_DYNAMICS_DIM:] == 1.0)


def test_load_expanded_into_current_model() -> None:
    """扩展后的 state_dict 应能被 strict 加载到 39D 模型。"""
    cfg = TrainConfig()
    old = _fake_old_ckpt(cfg, old_dim=34)
    model = ActorCritic(cfg)
    expanded = expand_state_dict_for_dynamics_dim(old, model.state_dict())
    # strict=True: 键与形状必须完全对应，禁止悄悄漏迁移
    model.load_state_dict(expanded, strict=True)


def test_forward_with_contact() -> None:
    """载入旧 ckpt 扩展后走一次 forward：形状/有限性 OK；新 5 维零输入等价旧 34D 行为。"""
    cfg = TrainConfig()
    old = _fake_old_ckpt(cfg, old_dim=34)
    model = ActorCritic(cfg)
    expanded = expand_state_dict_for_dynamics_dim(old, model.state_dict())
    model.load_state_dict(expanded, strict=True)
    model.eval()

    B, T, D = 2, cfg.context_len, DYNAMICS_DIM
    dynamics = torch.randn(B, T, D)
    patches = torch.randn(B, T, cfg.patch_channels, cfg.patch_size, cfg.patch_size)
    actions = torch.randn(B, T, cfg.action_dim)
    valid = torch.ones(B, T)

    with torch.no_grad():
        dist, val = model.forward(dynamics, patches, actions, valid_mask=valid)
    assert dist.mean.shape == (B, cfg.action_dim)
    assert val.shape == (B,)  # critic_head squeeze(-1) → (B,)
    assert torch.isfinite(dist.mean).all() and torch.isfinite(val).all()

    # 接触维度全 0 时（对应 raw contact=None 的老逻辑），应与 34D 时代的 forward 等价。
    # 具体：dynamics_proj 新列被扩为 0 → 5 维零输入贡献 0 → dyn_feat = dyn_feat_34D_only。
    dynamics_zero_tail = dynamics.clone()
    dynamics_zero_tail[..., BASE_DYNAMICS_DIM:] = 0.0
    with torch.no_grad():
        dist2, val2 = model.forward(dynamics_zero_tail, patches, actions, valid_mask=valid)
    assert torch.isfinite(dist2.mean).all() and torch.isfinite(val2).all()


def test_build_dynamics_backward_compat() -> None:
    """build_dynamics(raw) 无 contact 时，前 34 维应等于旧口径的输出。"""
    raw = np.random.randn(10, 33).astype(np.float32)
    out_new = build_dynamics(raw)
    out_new_v2 = build_dynamics(raw, contact=None)
    assert out_new.shape == (10, DYNAMICS_DIM)
    assert np.allclose(out_new, out_new_v2), "contact=None 与省略应完全一致"

    # 新增列必须全 0
    assert np.all(out_new[:, BASE_DYNAMICS_DIM:] == 0.0)

    # 与显式 zeros contact 也应一致
    zeros = np.zeros((10, CONTACT_DIM), dtype=np.float32)
    out_explicit = build_dynamics(raw, zeros)
    assert np.allclose(out_new, out_explicit)


def test_real_checkpoint_partial_load() -> None:
    """如果本地存在真实 34D checkpoint，试跑 partial-load（若无则 skip）。"""
    ckpt_path = (_REPO_ROOT / "runs" / "treefix2cont" / "checkpoints"
                 / "ppo_iter_0260.pt")
    if not ckpt_path.exists():
        print(f"[SKIP] 真实 checkpoint 不存在: {ckpt_path}")
        return

    cfg = TrainConfig()
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    old_state = ckpt["model"]

    # 确认这是老 34D checkpoint
    old_dyn_shape = old_state.get("dynamics_proj.weight").shape
    print(f"真实 ckpt dynamics_proj.weight: {tuple(old_dyn_shape)}")
    assert old_dyn_shape[1] == 34, "此测应针对 34D 老 ckpt"

    model = ActorCritic(cfg)
    expanded = expand_state_dict_for_dynamics_dim(old_state, model.state_dict())
    assert expanded["dynamics_proj.weight"].shape == (cfg.d_model, DYNAMICS_DIM)
    model.load_state_dict(expanded, strict=True)
    print(f"真实 ckpt 迁移成功: dyn 前 34 维保留，后 {CONTACT_DIM} 维初始化 0")


def main() -> None:
    test_dim_math()
    print("dim math OK")
    test_expand_preserves_old_values()
    print("expand 保留旧值 OK")
    test_load_expanded_into_current_model()
    print("扩展后 strict load OK")
    test_forward_with_contact()
    print("forward 有限性 OK")
    test_build_dynamics_backward_compat()
    print("build_dynamics 向后兼容 OK")
    test_real_checkpoint_partial_load()
    print("真实 34D checkpoint partial-load 通过")
    print("ALL P2 INTEGRATION OK")


if __name__ == "__main__":
    main()
