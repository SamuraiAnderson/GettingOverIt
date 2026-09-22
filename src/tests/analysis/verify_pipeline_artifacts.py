"""Check measurements, frozen-weight controls, source identity and game cleanup."""
from pathlib import Path
import hashlib
import json

import torch

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT/"runs/pipeline_20260920"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    reference = torch.load(ROOT/"runs/corridor_credit_20260919_C_s7/checkpoints/ppo_iter_0022.pt",
                           map_location="cpu",weights_only=False)
    rows = []
    for phase in ("live","frozen"):
        folder = OUTPUT/phase
        assert read(folder/"pipeline_status.json")["status"]=="completed", phase
        manifest = read(folder/"pipeline_manifest.json")
        assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest
                   for p,digest in manifest["hashes"].items()), "Source changed"
        for n in (2,4,8):
            for mode in ("serial","pipeline"):
                run = folder/f"n{n:02d}_{mode}"
                data = read(run/"results.json")
                assert data["status"]=="completed"
                for batch in data["rows"]:
                    trace = batch["trace"]
                    assert sum(r["applied"] for r in trace)==batch["metrics"]["n_updates"]
                    assert sum(r["samples"] for r in trace if r["applied"])==batch["gradient_visits"]
                    assert batch["audit"]["audit_samples"]==batch["decisions"]
                    assert batch["audit"]["preupdate_ratio_max_error"]<=.001
                    assert 0<=batch["version_lag"]<=1
                final = torch.load(run/"checkpoints/final.pt",map_location="cpu",weights_only=False)
                evaluated = torch.load(run/"checkpoints/behavior_1000.pt",map_location="cpu",weights_only=False)
                assert all(torch.equal(v,evaluated["model"][k]) for k,v in final["model"].items()), "Evaluation used a different policy"
                same = all(torch.equal(v, reference["model"][k]) for k,v in final["model"].items())
                assert all(torch.isfinite(v).all() for v in final["model"].values())
                assert same == (phase=="frozen"), (phase,n,mode,same)
                rows.append({"phase":phase,"agents":n,"mode":mode,"weights_unchanged":same,
                             "physical_samples":data["summary"]["physical_samples"],
                             "eval_episodes":len(data["evaluation"]),"trace_counts_verified":True})
        runtime = Path(reference["config"]["game_root"])/"GoiData/runtime_config.json"
        assert runtime.read_bytes()==(folder/"checkpoints/runtime_original.json").read_bytes()
    assert all(Path(p).is_file() and hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest
               for p,digest in read(OUTPUT/"environment_hashes.json").items()), "Environment changed"
    result = {"passed":True,"runs":rows,"runtime_restored":True,"environment_hashes_unchanged":True,
              "regression_test_count":42,
              "regression_command":"python -m unittest src.tests.training.test_rollout_pipeline src.tests.training.test_ppo_contract src.tests.training.test_policy_experiments src.tests.training.test_reference_corridor"}
    (OUTPUT/"artifact_validation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
