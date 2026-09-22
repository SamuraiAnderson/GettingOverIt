"""Plot archived pre-action contacts against actions without rerunning physics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(directory, stem, agent, output):
    directory = Path(directory)
    trajectory = json.loads((directory / f"{stem}_agent{agent}.json").read_text(encoding="utf-8"))
    packets = json.loads((directory / f"{stem}_native.json").read_text(encoding="utf-8"))
    trace = trajectory["trace"]
    n = len(trace)
    if not all(p.get("aligned") for p in packets[:n]):
        raise ValueError("Native contacts must be aligned with archived states")
    # State/contact index t is BEFORE action t; progress in trace is AFTER it.
    x = np.arange(n)
    contacts = np.array(trajectory["contacts"])[:n]
    actions = np.array([r["action"] for r in trace])
    native = np.array([p["agents"][agent]["physicsContact"] for p in packets[:n]])
    known = np.array([p["agents"][agent]["physicsKnown"] for p in packets[:n]], dtype=bool)
    native = np.where(known, native, np.nan)
    callback = np.array([p["agents"][agent]["contact"][1] for p in packets[:n]], dtype=float)
    callback_known = [p["agents"][agent]["contactKnown"][1] for p in packets[:n]]
    callback = np.where(callback_known, callback, np.nan)
    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.4, 1, 1, 1]}, layout="constrained")
    axes[0].plot(x, actions[:, 0], label="dx", color="#247ba0", linewidth=1)
    axes[0].plot(x, actions[:, 1], label="dy", color="#f08a4b", linewidth=1)
    axes[0].set_ylabel("Executed action")
    delta = np.linalg.norm(np.diff(actions, axis=0), axis=1)
    axes[1].plot(x[1:], delta, color="#7251a0", linewidth=1, label="Action change (L2)")
    axes[1].set_ylabel("Action change")
    for axis, proxy, truth, label in (
        (axes[2], contacts[:, 0], native[:, 0], "Tip contact"),
        (axes[3], contacts[:, 3], native[:, 2], "Pot contact"),
        (axes[4], contacts[:, 1], callback, "Tip grip"),
    ):
        axis.step(x, truth, where="post", color="#247ba0", label="Native", linewidth=1.4)
        axis.step(x, proxy, where="post", color="#d45d45", label="Geometry proxy", linewidth=1, linestyle="--")
        axis.set_ylabel(label)
        axis.set_ylim(-.15, 1.15)
        axis.set_yticks([0, 1])
    # Only observed native edges, with no fabricated event at reset.
    for index in np.flatnonzero(np.diff(native[:, 0]) != 0) + 1:
        if not np.isfinite(native[index-1:index+1, 0]).all():
            continue
        for axis in axes[:2]:
            axis.axvline(index, color="#247ba0", linewidth=.6, alpha=.25)
    for axis in axes:
        axis.grid(alpha=.15)
        axis.legend(loc="upper right", framealpha=.8, fontsize=8)
    axes[0].set_title(f"{stem}, agent {agent}: pre-action contact / action alignment\n"
                      "Vertical guides = native tip enter/exit; geometry remains a proxy")
    axes[-1].set_xlabel("Action index t (contacts sampled at state t, before executing action t)")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--stem", default="exploration_000")
    parser.add_argument("--agent", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(plot(args.directory, args.stem, args.agent, args.output))
