"""Assemble the 2026-10-02 Hub depth fix acceptance from preserved raw reports."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NAMES = ["native", "long", "reference", "reset_2agents", "reset_8agents"]
paths = {n: ROOT / f"runs/hub_depth_fix_{n}_20261002/result.json" for n in NAMES}
reports = {n: json.loads(p.read_text(encoding="utf-8-sig")) for n, p in paths.items()}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plugin = ROOT / "bin/HubDepthFix/GameRuntime_v2.dll"
summary = {"status": "PASS", "plugin_sha256": sha(plugin), "reports": {n: str(p.relative_to(ROOT)) for n, p in paths.items()},
           "cold_start_repeatability": {}, "reset": {}, "limitations": [
               "Single machine and measured fixtures; not universal determinism or complete game equivalence.",
               "Hub visual depth no longer follows hand animation after Awake.",
               "Reset checks use the existing experimental controlled scene reset and native input options.",
               "Native start acceptance is tolerance-based, not bitwise equivalence to the reset runtime."]}
for name in ("native", "long", "reference"):
    r = reports[name]
    assert r["status"] == "COMPLETE" and r["production_unchanged"] and not r["errors"]
    assert r["fingerprints"]["plugin"] == summary["plugin_sha256"]
    for case in r["cases"]:
        log = paths[name].parent / (case["id"] + ".game.log")
        active = "Hub visual-depth write suppressed; native Awake initialization preserved" in log.read_text(encoding="utf-8-sig")
        assert active == case["hub_depth_fix"]
    if name != "reference":
        assert r["analysis"]["repeatability"]["passed"]
        assert all(c["hub_depth_fix"] for c in r["cases"])
        summary["cold_start_repeatability"][name] = {"launches": len(r["cases"]), "steps_per_launch": r["parameters"]["steps"],
                                                  "all_recorded_fields_bitwise_equal": True}
ref = reports["reference"]
for key in ("native_parameters", "native_input", "native_start"):
    assert ref["analysis"][key]["passed"]
assert all(c["hub_depth_fix"] == (c["mode"] == "candidate") for c in ref["cases"])
summary["native_reference"] = {
    "original_reference_patch_disabled": True,
    "parameters_passed": True, "input_rows_bitwise_equal": sum(c["rows"] for c in ref["analysis"]["native_input"]["cases"]),
    "start_within_limits": True,
    "max_start_position_component_error_m": max(c["bodies"]["errors"]["position_m"] for c in ref["analysis"]["native_start"]["cases"]),
    "max_start_angle_error_deg": max(c["bodies"]["errors"]["angle_deg"] for c in ref["analysis"]["native_start"]["cases"])}
for name in ("reset_2agents", "reset_8agents"):
    r = reports[name]
    assert r["status"] == "PASS" and not r["errors"] and len(r["cases"]) == r["expected_cases"]
    assert r["fingerprints"]["deployed_plugin"] == summary["plugin_sha256"]
    assert "suppressHubDepthWrite" not in r["runtime_config_requested"]
    assert all(s["runtime_config_effective"]["suppressHubDepthWrite"] for s in r["sessions"])
    frames = [f for c in r["cases"] for f in c["assessment"]["frames"]]
    assert len(frames) == len(r["cases"]) * (r["parameters"]["probe_steps"] + 1)
    assert all(f["vs_cold_reference"]["bitwise_equal_without_timestamp"] for f in frames)
    summary["reset"][name] = {"cases": len(r["cases"]), "starts": len(r["sessions"]), "bitwise_frames_excluding_timestamp": len(frames),
                             "default_enabled_for_old_config": True}
summary["production_before"] = ref["production_before"]
summary["production_after"] = {p: sha(Path(p)) for p in ref["production_before"]}
summary["production_unchanged"] = summary["production_before"] == summary["production_after"]
assert summary["production_unchanged"]
summary["source_sha256"] = {p: sha(ROOT / p) for p in [
    "src/GameRuntime_v2/PlayerControl/HubDepthWritePatch.cs", "src/GameRuntime_v2/Core/GameRuntimeManager.cs",
    "src/GameRuntime_v2/Core/Configuration/RuntimeConfig.cs", "src/tests/control_interaction/test_native_dynamics.py",
    "src/tests/control_interaction/HubDepthPatchChecks.cs"]}
target = paths["native"].parent / "acceptance_summary.json"
target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k not in ("source_sha256", "production_before", "production_after")}, indent=2))
