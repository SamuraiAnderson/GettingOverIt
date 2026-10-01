"""Matched-input inference benchmark; excludes TCP, features, rollout and PPO update."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from training.config import TrainConfig
from training.experimental_policy import create_actor_critic
from training.decision_buffer import DecisionRolloutBuffer
from training.dataset import left_pad_sequence


def benchmark(checkpoint, archive, steps=64, repeats=3):
    torch.manual_seed(17)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = TrainConfig()
    for field in fields(cfg):
        if field.name in saved["config"]:
            setattr(cfg, field.name, saved["config"][field.name])
    buffer = DecisionRolloutBuffer.load(archive)
    episodes = buffer.episodes[:2]
    steps = min(steps, min(len(e["dynamics"]) - 1 for e in episodes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()
    result = {"scope": "Same archived inputs, single-step policy, no TCP/features/training; not learning performance",
              "steps": steps, "agents": len(episodes), "device": str(device), "repeats": repeats}
    for architecture in ("transformer", "gru"):
        cfg.policy_backbone = architecture
        cfg.action_segment_steps = 1
        model = create_actor_critic(cfg).to(device).eval()
        model.initialize_from(saved["model"])
        inputs = []
        for t in range(steps):
            context = cfg.context_len if architecture == "transformer" else 1
            arrays, masks = [], []
            for episode in episodes:
                windows = [left_pad_sequence(episode[name][max(0, t-context+1):t+1], context)
                           for name in ("dynamics", "patches", "act_history")]
                arrays.append([a for a, _ in windows]); masks.append(windows[0][1])
            inputs.append((*(torch.tensor(np.stack([a[k] for a in arrays]), device=device) for k in range(3)),
                           torch.tensor(np.asarray(masks), device=device)))
        times = []
        for trial in range(repeats + 1):
            hidden = None
            model.encoded_frame_count = 0
            synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                for dyn, pat, act, mask in inputs:
                    if architecture == "gru":
                        feature, hidden = model.recurrent_features(dyn, pat, act, hidden=hidden)
                        model.distribution_from_features(feature[:, 0])
                    else:
                        model(dyn, pat, act, mask)
            synchronize()
            if trial:
                times.append(time.perf_counter() - start)
        result[architecture] = {"seconds": times, "median_seconds": float(np.median(times)),
                                "cnn_frame_evaluations": model.encoded_frame_count}
        del inputs, model
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark(args.checkpoint, args.archive, args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
