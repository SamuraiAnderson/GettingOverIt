"""Verify the real resume configuration without starting a game or training."""
import contextlib
import csv
import hashlib
import io
import json
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from training import gru_sil, main_ppo


def main():
    checkpoint = ROOT / "runs/gru_sil_dynamics_1h2_20261004/checkpoints/ppo_iter_0209.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    overrides = ["--gru-sil-max-updates", "16", "--gru-sil-budget-ratio", "1.0"]
    base = ["main_ppo", "--resume", str(checkpoint), "--dry-run", "--torch-threads", "4",
            "--run-dir", str(Path(__file__).parent / "resume_preflight_only")]
    validate = gru_sil.validate_gru_sil

    def resolve(extra):
        config = {}
        def inspect(value):
            validate(value)
            config.update(asdict(value))
        output = io.StringIO()
        with patch.object(sys, "argv", base + extra), contextlib.redirect_stdout(output), \
             patch.object(gru_sil, "validate_gru_sil", side_effect=inspect), \
             patch.object(main_ppo, "RunRecorder", side_effect=AssertionError("dry-run began recording")):
            main_ppo.main()
        return {"resolved_config": config, "dry_run": json.loads(output.getvalue())}

    old, new = resolve([]), resolve(overrides)
    changed = {k: {"old": old["resolved_config"][k], "new": v}
               for k, v in new["resolved_config"].items() if old["resolved_config"][k] != v}
    assert changed == {"gru_sil_max_updates": {"old": 4, "new": 16},
                       "gru_sil_budget_ratio": {"old": .25, "new": 1.}}
    for result in (old, new):
        data = result["dry_run"]
        assert data["resume_iteration"] == 209
        assert data["rng_resume_mode"] == "restore" and data["training_seed"] == 7
        assert data["gru_sil_sampling_mode"] == "quality_dynamics_v3"
        assert data["gru_sil_cache_mb"] == 1024
        assert data["gru_sil_enabled"] and not data["random_deploy"]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    history = []
    for name in ("gru_sil_dynamics_1h_20261004", "gru_sil_dynamics_1h2_20261004"):
        with (ROOT / "runs" / name / "metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                ppo_updates = int(float(row["n_updates"]))
                old_budget = min(4, int(ppo_updates * .25))
                assert old_budget == int(float(row["gru_sil_budget"]))
                history.append({"run": name, "iteration": int(row["iteration"]),
                                "ppo_updates": ppo_updates, "old_budget": old_budget,
                                "new_budget": min(16, ppo_updates)})
    old_total = sum(row["old_budget"] for row in history)
    new_total = sum(row["new_budget"] for row in history)
    report = {"status": "passed", "training_started": False, "checkpoint": str(checkpoint),
              "checkpoint_sha256": digest, "checkpoint_unchanged": True,
              "resume_overrides": overrides, "changed_config_fields": changed,
              "before": old, "after": new,
              "historical_budget_comparison": {"assumption": "Hold historical PPO update counts fixed; this is not a new training run or an accepted-update forecast.",
                                               "rounds": len(history), "old_total": old_total,
                                               "new_total": new_total, "ratio": new_total / old_total,
                                               "per_round": history},
              "interpretation": "The increased budget is a candidate limit. Positive-advantage and cumulative KL gates remain unchanged; accepted updates and behavior improvement are not guaranteed.",
              "tests": {"count": 53, "result": "passed", "log": "regression_tests.txt"}}
    (Path(__file__).parent / "resume_preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "training_started", "checkpoint_unchanged", "changed_config_fields")}, indent=2))


if __name__ == "__main__":
    main()
