"""一次有截止时间的执行外壳：不接管已开的游戏，记下源码哈希，结束时恢复运行配置。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import psutil

from .io import ROOT, now, write_json
from .runner import BudgetStop


def refuse_running_game():
    if any(p.info["name"] == "GettingOverIt.exe" for p in psutil.process_iter(["name"])):
        raise RuntimeError("An existing game is running; refusing to take it over")


def game_root():
    project = json.loads((ROOT / "src/config/project.json").read_text(encoding="utf-8"))
    return Path(project["game"]["executable_path"]).parent


def python_tree(*extra):
    paths = [path for folder in ("src/training", "src/env", "src/start")
             for path in (ROOT / folder).rglob("*.py")]
    paths.extend(extra)
    return paths


def write_source_hashes(campaign, paths):
    write_json(campaign / "source_hashes.json", [
        {"path": os.path.relpath(path, ROOT), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in paths if Path(path).is_file()
    ])


def backup_runtime(campaign):
    runtime = game_root() / "GoiData/runtime_config.json"
    original = runtime.read_bytes() if runtime.exists() else None
    if original is not None:
        (campaign / "runtime_original.json").write_bytes(original)
    return runtime, original


def restore_runtime(runtime, original):
    if original is not None:
        runtime.write_bytes(original)
    elif runtime.exists():
        runtime.unlink()
    if original is not None:
        return runtime.read_bytes() == original
    return not runtime.exists()


def close_campaign(runner, manifest, campaign, runtime, original, status, error, report):
    runner.kill_own_child()
    restored = restore_runtime(runtime, original)
    manifest.update(status=status, finished_at=now(), error=error, runtime_restored=restored)
    write_json(campaign / "manifest.json", manifest)
    runner.status(status, error=error, runtime_restored=restored)
    report(runner, status, error)
    runner.event("finished", status=status, error=error, runtime_restored=restored)
    return restored
