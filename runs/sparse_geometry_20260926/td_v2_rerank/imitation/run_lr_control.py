"""Learning-rate control from the frozen epoch-69 imitation weights.

A: lr 1e-4 for 60 epochs.
B: lr 3e-5 for 30 epochs, then 1e-5 for 30 epochs.
Same windows, batch order per seed, fresh AdamW, dropout off.
Best checkpoint is chosen by a full-window eval after each epoch.
"""
import json
import shutil
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from training.config import TrainConfig
from training.experimental_policy import create_actor_critic

SRC = Path(__file__).resolve().parent / "imitation.pt"
WINDOWS = Path(__file__).resolve().parent / "windows.pt"
OUT = Path(__file__).resolve().parent / "lr_control_20260927"
FROZEN = OUT / "frozen_epoch69.pt"
EPOCHS = 60
BATCH = 64
SEEDS = (47, 48, 49)
RANK1_LEN = 815
RANK2_LEN = 491
RANK1_BEFORE = 800
RANK2_BEFORE = 476


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def perms_for(n, seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return [torch.randperm(n, generator=generator) for _ in range(EPOCHS)]


def predict(model, keys, index, dev):
    dist, _ = model.forward(
        keys[0][index].to(dev), keys[1][index].to(dev), keys[2][index].to(dev),
        keys[3][index].to(dev), keys[5][index].to(dev), keys[6][index].to(dev),
        keys[7][index].to(dev), keys[8][index].to(dev),
    )
    return torch.tanh(dist.mean) * model._action_scale


@torch.no_grad()
def forward_all(model, keys, dev):
    n = keys[0].shape[0]
    chunks = []
    for begin in range(0, n, BATCH):
        index = torch.arange(begin, min(begin + BATCH, n))
        chunks.append(predict(model, keys, index, dev).detach().cpu())
    return torch.cat(chunks, dim=0)


def eval_mse(pred, target):
    return float(torch.mean((pred - target) ** 2))


def vector_stats(pred, target):
    err = torch.linalg.norm((pred - target).detach(), dim=-1).cpu().numpy()
    return {
        "mse": eval_mse(pred, target),
        "p95": float(np.quantile(err, 0.95)) if len(err) else None,
        "max": float(err.max()) if len(err) else None,
        "n": int(len(err)),
    }


def split_metrics(pred, target):
    rank1 = slice(0, RANK1_LEN)
    rank2 = slice(RANK1_LEN, RANK1_LEN + RANK2_LEN)
    r1_before = slice(0, RANK1_BEFORE)
    r1_tail = slice(RANK1_BEFORE, RANK1_LEN)
    r2_before = slice(RANK1_LEN, RANK1_LEN + RANK2_BEFORE)
    r2_tail = slice(RANK1_LEN + RANK2_BEFORE, RANK1_LEN + RANK2_LEN)
    return {
        "overall": vector_stats(pred, target),
        "rank1": vector_stats(pred[rank1], target[rank1]),
        "rank2": vector_stats(pred[rank2], target[rank2]),
        "rank1_before_endpoint": vector_stats(pred[r1_before], target[r1_before]),
        "rank1_tail": vector_stats(pred[r1_tail], target[r1_tail]),
        "rank2_before_endpoint": vector_stats(pred[r2_before], target[r2_before]),
        "rank2_tail": vector_stats(pred[r2_tail], target[r2_tail]),
    }


def lr_for(group, epoch):
    if group == "A":
        return 1e-4
    if epoch < 30:
        return 3e-5
    return 1e-5


def train_one(group, seed, saved, keys, target, dev, batch_perms):
    run_dir = OUT / f"{group}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(**{k: v for k, v in saved["config"].items() if k in TrainConfig.__dataclass_fields__})
    model = create_actor_critic(cfg).to(dev)
    model.load_state_dict(saved["model"])
    model.eval()
    params = [p for name, p in model.named_parameters() if not name.startswith("critic_head") and p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr_for(group, 0), weight_decay=0.0)
    n = keys[0].shape[0]
    curve = []
    best_mse = None
    best_epoch = None
    for epoch in range(EPOCHS):
        opt.param_groups[0]["lr"] = lr_for(group, epoch)
        perm = batch_perms[epoch]
        for begin in range(0, n, BATCH):
            index = perm[begin:begin + BATCH]
            pred = predict(model, keys, index, dev)
            loss = torch.mean((pred - target[index].to(dev)) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        pred_all = forward_all(model, keys, dev)
        mse = eval_mse(pred_all, target)
        row = {"epoch": epoch, "lr": lr_for(group, epoch), "eval_mse": mse}
        curve.append(row)
        (run_dir / "eval_curve.json").write_text(json.dumps(curve, indent=2), encoding="utf-8")
        print(f"{group} seed {seed} epoch {epoch} lr {row['lr']:.1e} eval_mse {mse:.4f}", flush=True)
        if best_mse is None or mse < best_mse:
            best_mse = mse
            best_epoch = epoch
            payload = {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": saved["config"],
                "reference_corridor": saved.get("reference_corridor"),
                "policy_context": saved.get("policy_context", {}),
                "group": group,
                "seed": seed,
                "epoch": epoch,
                "eval_mse": mse,
                "source": str(FROZEN),
            }
            torch.save(payload, run_dir / "best.pt")
            torch.save({"optim": opt.state_dict(), "epoch": epoch, "eval_mse": mse}, run_dir / "best_optim.pt")
            metrics = split_metrics(pred_all, target)
            metrics.update(group=group, seed=seed, epoch=epoch, eval_mse=mse)
            (run_dir / "best_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"group": group, "seed": seed, "best_epoch": best_epoch, "best_eval_mse": best_mse, "curve": curve}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not FROZEN.exists():
        shutil.copy2(SRC, FROZEN)
    saved = torch.load(FROZEN, map_location="cpu", weights_only=False)
    keys = torch.load(WINDOWS, map_location="cpu", weights_only=True)
    n = int(keys[0].shape[0])
    if n != RANK1_LEN + RANK2_LEN:
        raise RuntimeError(f"expected {RANK1_LEN + RANK2_LEN} windows, got {n}")
    dev = device()
    cfg = TrainConfig(**{k: v for k, v in saved["config"].items() if k in TrainConfig.__dataclass_fields__})
    base = create_actor_critic(cfg).to(dev)
    base.load_state_dict(saved["model"])
    base.eval()
    target = keys[4]
    try:
        baseline_pred = forward_all(base, keys, dev)
    except torch.OutOfMemoryError:
        if dev.type != "cuda":
            raise
        torch.cuda.empty_cache()
        dev = torch.device("cpu")
        base = base.to(dev)
        baseline_pred = forward_all(base, keys, dev)
        print("cuda out of memory, training on cpu", flush=True)
    baseline = split_metrics(baseline_pred, target)
    baseline.update(source=str(FROZEN), device=str(dev), windows=n)
    (OUT / "baseline_epoch69.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print(f"frozen eval_mse {baseline['overall']['mse']:.4f} device {dev}", flush=True)
    del base
    config = {
        "source": str(FROZEN),
        "windows": str(WINDOWS),
        "epochs": EPOCHS,
        "batch": BATCH,
        "seeds": list(SEEDS),
        "groups": {
            "A": "lr 1e-4 for 60 epochs",
            "B": "lr 3e-5 for 30 epochs, then 1e-5 for 30 epochs",
        },
        "weight_decay": 0.0,
        "dropout": "eval mode, disabled",
        "optimizer": "fresh AdamW each run",
        "selection": "minimum full-window eval MSE after each epoch",
        "before_endpoint": {"rank1": "actions 0-799", "rank2": "actions 0-475"},
        "device": str(dev),
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    runs = []
    for seed in SEEDS:
        batch_perms = perms_for(n, seed)
        for group in ("A", "B"):
            runs.append(train_one(group, seed, saved, keys, target, dev, batch_perms))
    summary = {"baseline": baseline, "runs": [
        {k: v for k, v in run.items() if k != "curve"} | {"eval_curve": run["curve"]}
        for run in runs
    ]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("lr control finished", flush=True)


if __name__ == "__main__":
    main()
