"""Export the latest elite pool's top 10 by confirmed D for motion render."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from training.dataset import Trajectory
from training.reference_corridor import trajectory_endpoint
from training.run_logging import load_elites
from training.sil_archive import dominates

ELITES = ROOT / "runs/confirmed_endpoint_v2_formal_20260928_S_s47/elites/elites_iter_0123.npz"
META = ROOT / "runs/confirmed_endpoint_v2_formal_20260928_S_s47/run_meta.json"
OUT = ROOT / "runs/confirmed_endpoint_v2_formal_20260928_S_s47/ranks"


def main():
    rows = load_elites(ELITES)
    ranked = []
    for index, row in enumerate(rows):
        endpoint = trajectory_endpoint(Trajectory(**row))
        if not endpoint.valid or endpoint.t_cap is None:
            continue
        ranked.append({
            "source_index": index,
            "iteration": int(row["iteration"]),
            "t": int(endpoint.t_cap),
            "D": float(endpoint.distance),
            "cap": int(endpoint.cap),
            "row": row,
        })
    ranked.sort(key=lambda item: (-item["D"], item["t"], item["source_index"]))
    front = []
    for item in sorted(ranked, key=lambda item: (item["t"], -item["D"])):
        point = (item["t"], item["D"])
        if any(dominates((other["t"], other["D"]), point) for other in front):
            continue
        front = [other for other in front if not dominates(point, (other["t"], other["D"]))]
        front.append(item)
    front_ids = {id(item) for item in front}
    top = ranked[:10]
    episodes = OUT / "episodes"
    episodes.mkdir(parents=True, exist_ok=True)
    config = json.loads(META.read_text(encoding="utf-8"))["config"]
    (episodes / "manifest.json").write_text(
        json.dumps({"config": config}, ensure_ascii=False), encoding="utf-8")
    listing = []
    for rank, item in enumerate(top, start=1):
        states = np.asarray(item["row"]["raw_states"])
        path = episodes / f"rank{rank:02d}.json"
        payload = {
            "states": states.tolist(),
            "trace": [{"step": step} for step in range(1, len(states))],
            "metrics": {
                "credited_progress_m": item["D"],
                "arrival_steps": None,
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        listing.append({
            "rank": rank,
            "path": str(path),
            "source_index": item["source_index"],
            "iteration": item["iteration"],
            "t": item["t"],
            "D": item["D"],
            "cap": item["cap"],
            "on_front": id(item) in front_ids,
        })
    (OUT / "ranking.json").write_text(json.dumps({
        "source": str(ELITES),
        "criterion": "confirmed D descending, then smaller t",
        "episodes": listing,
    }, indent=2), encoding="utf-8")
    print(json.dumps([{k: v for k, v in row.items() if k != "path"} for row in listing], indent=2))


if __name__ == "__main__":
    main()
