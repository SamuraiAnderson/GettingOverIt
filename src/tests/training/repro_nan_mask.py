"""最小复现：定位早期填充窗口 forward NaN 是否源自注意力掩码全 -inf 行。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from training.config import TrainConfig
from training.dataset import DYNAMICS_DIM
from training.model import ActionPredictor, build_key_padding_mask


def main() -> None:
    torch.manual_seed(0)
    config = TrainConfig()
    ctx = config.context_len
    model = ActionPredictor(config)
    model.eval()

    # 构造一个 tau=0 风格样本：仅最后一步有效，其余左填充零
    B = 1
    dyn = torch.zeros(B, ctx, DYNAMICS_DIM)
    pat = torch.zeros(B, ctx, config.patch_channels, config.patch_size, config.patch_size)
    act = torch.zeros(B, ctx, config.action_dim)
    vm = torch.zeros(B, ctx)
    vm[:, -1] = 1.0  # 只有最后一步有效
    dyn[:, -1] = 0.5  # 给最后一步一些非零值

    with torch.no_grad():
        out_masked = model(dyn, pat, act, valid_mask=vm)
        out_nomask = model(dyn, pat, act, valid_mask=None)

    print(f"valid_mask 仅末步: out(masked)   = {out_masked.flatten().tolist()}")
    print(f"valid_mask=None : out(no mask)  = {out_nomask.flatten().tolist()}")

    # 检查掩码本身
    kpm = build_key_padding_mask(vm)
    print(f"key_padding_mask (2T) 填充位数 = {int(torch.isinf(kpm).sum())}/{kpm.shape[1]}")

    # 手动模拟：causal + key_padding 是否产生整行 -inf
    T2 = 2 * ctx
    causal = torch.triu(torch.ones(T2, T2) * float("-inf"), diagonal=1)
    combined = causal + kpm[0].unsqueeze(0)  # (T2, T2) 每个 query 行加上 key padding
    all_inf_rows = int((torch.isinf(combined) & (combined < 0)).all(dim=1).sum())
    print(f"causal+padding 组合中整行全 -inf 的 query 行数 = {all_inf_rows}/{T2}")


if __name__ == "__main__":
    main()
