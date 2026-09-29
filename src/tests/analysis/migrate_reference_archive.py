"""Migrate old corridor archives into a new reference fingerprint.

Never silently fills missing 35D anchors. 33D-only files stay legacy unless a
verified pose proxy is supplied. Pruned files are marked partial.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from training.config import TrainConfig
from training.reference_corridor import ReferenceCorridor, replay_corridor_result
from training.reference_fingerprint import reference_fingerprint, write_fingerprint_sidecar
from training.reference_anchor import ANCHOR_PLAYER_ORIGIN, ANCHOR_POT_CALIBRATED


def _copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def migrate_npz(
    src: Path,
    dest: Path,
    route: ReferenceCorridor | None,
    config: TrainConfig,
    *,
    allow_proxy: bool = False,
) -> dict:
    data = np.load(src)
    states = np.asarray(data["raw_states"], dtype=np.float32)
    width = int(states.shape[-1])
    kind = "legacy_kept"
    note = "33D archive kept on the player-origin definition"
    if width == 35 and config.reference_anchor_id == ANCHOR_POT_CALIBRATED:
        kind = "native_or_reconstructed"
        note = "35D columns present; corridor fields recomputed from [33:35] if verified"
    elif width == 33 and config.reference_anchor_id == ANCHOR_POT_CALIBRATED:
        if not allow_proxy:
            kind = "legacy_kept"
            note = "33D cannot invent a calibrated pot anchor; file kept as player-origin legacy"
        else:
            kind = "proxy_migration"
            note = "player→hub proxy requested; caller must have validated the error bound"
    if "actions" in data.files and len(data["actions"]) + 1 != len(states):
        raise ValueError(f"{src}: states length must be actions+1")
    dest.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: data[key] for key in data.files}
    if (
        route is not None
        and kind != "legacy_kept"
        and "corridor_progress" in arrays
        and config.reference_anchor_id != ANCHOR_PLAYER_ORIGIN
    ):
        arrays.pop("corridor_progress", None)
        arrays.pop("guidance_index", None)
    if route is not None and kind != "legacy_kept":
        played = replay_corridor_result(route, states)
        arrays["corridor_progress"] = np.asarray(played["legacy_credit"], dtype=np.float32)
        arrays["guidance_index"] = played["guidance_index"]
        arrays["corridor_credit_events"] = played["credit_events"]
        arrays["corridor_confirmed_progress"] = played["confirmed_progress"]
        arrays["step_rewards"] = played["rewards"]
        arrays["progress_mode"] = np.asarray(played["mode"])
        note += f"; replayed {played['mode']} D={played['distance']:.4f} cap={played['cap']}"
        if "terminated" not in data.files:
            meta_termination = False
        else:
            meta_termination = True
    else:
        meta_termination = "terminated" in data.files
    if dest.suffix == ".npz":
        np.savez_compressed(dest, **arrays)
    else:
        _copy(src, dest)
    meta = {
        "source": str(src),
        "destination": str(dest),
        "migration_kind": kind,
        "partial": bool("corridor_progress" in data.files and kind == "legacy_kept"),
        "termination_known": meta_termination,
        "note": note,
        "raw_width": width,
    }
    dest.with_suffix(dest.suffix + ".migration.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    write_fingerprint_sidecar(dest, reference_fingerprint(
        config, route_meta=route.metadata if route is not None else None,
        schema_id="anchor35" if width == 35 else "legacy33",
        raw_dim=width,
    ))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-trajectory", type=Path, default=None)
    parser.add_argument("--allow-proxy", action="store_true")
    args = parser.parse_args()
    config = TrainConfig()
    route = None
    if args.reference_trajectory is not None:
        route = ReferenceCorridor.from_file(args.reference_trajectory, config)
    _copy(args.source, args.output.with_name(args.source.name + ".orig"))
    print(json.dumps(migrate_npz(args.source, args.output, route, config, allow_proxy=args.allow_proxy), indent=2))


if __name__ == "__main__":
    main()
