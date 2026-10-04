"""Copy reviewed artifacts without changing sources; verify hashes and NPZ structure."""
from pathlib import Path
import hashlib
import json
import shutil
from datetime import datetime, timezone

import numpy as np

DEST = Path(__file__).resolve().parent
ROOT = DEST.parent.parent
RUNS = (
    "confirmed_endpoint_v2_formal_20260928_S_s47",
    "delayed_exploration_8h_20260930",
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    entries = []

    def copy(source, target, purpose):
        source_hash = digest(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if digest(target) != source_hash:
                raise RuntimeError(f"Refusing to overwrite different data: {target}")
        else:
            shutil.copy2(source, target)
        if digest(target) != source_hash or digest(source) != source_hash:
            raise RuntimeError(f"Copy verification failed: {source}")
        entries.append({
            "source": source.relative_to(ROOT).as_posix(),
            "destination": target.relative_to(DEST).as_posix(),
            "bytes": target.stat().st_size,
            "sha256": source_hash,
            "purpose": purpose,
            "approved_for_training_data": False,
        })

    pools = 0
    checkpoints = 0
    for name in RUNS:
        source = ROOT / "runs" / name
        for checkpoint in sorted((source / "checkpoints").glob("*.pt")):
            copy(checkpoint, DEST / "warm_start_candidates" / name / checkpoint.name,
                 "Model initialization candidate only; not validated under corrected resets. Use init-from, not full resume.")
            checkpoints += 1
        for archive in sorted((source / "elites").glob("*.npz")):
            target = DEST / "analysis_only" / name / "elites" / archive.name
            copy(archive, target, "Historical trajectory analysis only; no BC/SIL restore or clean-start ranking.")
            with np.load(target, allow_pickle=False) as data:
                count = int(data["n_trajectories"])
                if len(data["scores"]) != count or len(data["iterations"]) != count:
                    raise ValueError(f"Invalid metadata length: {target}")
                for i in range(count):
                    states = data[f"t{i}_states"]
                    actions = data[f"t{i}_actions"]
                    if states.ndim != 2 or states.shape[1] not in (33, 35):
                        raise ValueError(f"Invalid states: {target}, trajectory {i}")
                    if actions.shape != (len(states) - 1, 2):
                        raise ValueError(f"Invalid actions: {target}, trajectory {i}")
                    if not np.isfinite(states).all() or not np.isfinite(actions).all():
                        raise ValueError(f"Nonfinite data: {target}, trajectory {i}")
                entries[-1]["validation"] = {"trajectory_count": count, "finite_states_actions": True, "shapes_valid": True}
            sidecar = archive.with_name(archive.name + ".fingerprint.json")
            copy(sidecar, target.with_name(sidecar.name), "Original trajectory fingerprint; does not certify reset quality.")
            pools += 1
        for filename in ("run_meta.json", "metrics.csv", "console.log"):
            copy(source / filename, DEST / "provenance" / name / filename,
                 "Historical collection context; scores are not a corrected-reset benchmark.")

    for relative in (
        "runs/solver_residue_20261002/result.json",
        "src/tests/control_interaction/test_solver_residue.py",
        "src/training/rollout.py",
        "src/training/main_ppo.py",
        "src/GameRuntime_v2/Core/StepController.cs",
    ):
        copy(ROOT / relative, DEST / "evidence" / relative,
             "Review evidence; source code is a migration-time snapshot, not proof of the historical deployed build.")

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "verified_copy_sources_preserved",
        "scope": list(RUNS),
        "approved_training_trajectories": 0,
        "warm_start_candidate_checkpoints": checkpoints,
        "analysis_only_pool_archives": pools,
        "file_count": len(entries),
        "copied_bytes": sum(e["bytes"] for e in entries),
        "limitations": [
            "No gameplay replay or corrected-reset evaluation performed.",
            "Checkpoint files are byte-verified, not loaded or certified for transfer quality.",
            "NPZ validation checks structure and finite values, not physical reproducibility.",
            "Historical pool snapshots overlap; do not concatenate them as independent trajectories.",
            "No arbitrary prefix removal, reward modification, or state velocity zeroing performed.",
            "Original code/configuration dependencies remain in the repository; this is not a standalone runtime bundle.",
        ],
        "files": entries,
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
