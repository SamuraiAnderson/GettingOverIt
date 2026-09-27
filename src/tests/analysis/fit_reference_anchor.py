"""CLI for fitting an unverified body-local TAS anchor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from training.anchor_fit import (
    build_unverified_profile,
    evaluate_fit,
    fit_local_anchor,
    split_fit_holdout,
)


def load_samples(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["samples"] if isinstance(payload, dict) else payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit an unverified body-local TAS anchor")
    parser.add_argument("samples", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    samples = load_samples(args.samples)
    fit_set, hold_set = split_fit_holdout(samples, seed=args.seed)
    fitted = fit_local_anchor(fit_set)
    holdout = evaluate_fit(fitted["local_xy"], hold_set)
    profile = build_unverified_profile(fitted, holdout, args.samples)
    if holdout["meets_candidate_p95_m"]:
        profile["note"] = (
            "Holdout met the candidate P95 target, but status stays unverified until "
            "a reviewer writes status=verified after visual overlay checks."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"fit": fitted, "holdout": holdout, "status": profile["status"]}, indent=2))


if __name__ == "__main__":
    main()
