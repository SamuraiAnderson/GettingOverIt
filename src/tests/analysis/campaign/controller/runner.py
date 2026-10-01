"""一个子进程的截止、启动、停止和 checkpoint 核对。"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime

from .io import ROOT, now, write_json
from .stats import stats_for_run

DEFAULT_SOURCE = "runs/p2_contact/checkpoints/ppo_iter_0335.pt"


class BudgetStop(Exception):
    pass


class Runner:
    def __init__(self, deadline, *, campaign, flags, source=DEFAULT_SOURCE):
        self.deadline = deadline
        self.campaign = campaign
        self.flags = flags
        self.source = source
        self.job = None
        self.child = None
        self.invocation = datetime.now().strftime("%H%M%S")
        self.frozen_sources = json.loads((self.campaign / "source_hashes.json").read_text(encoding="utf-8-sig"))

    def run_name(self, arm, seed):
        if arm not in self.flags:
            raise ValueError(f"Unconfigured arm: {arm}")
        return f"{self.campaign.name}_{arm}_s{seed}"

    def stats(self, arm, seed):
        return stats_for_run(ROOT / "runs" / self.run_name(arm, seed))

    def event(self, kind, **data):
        row = {"time": now(), "event": kind, **data}
        with (self.campaign / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def guard(self, estimated_seconds=0):
        if (self.campaign / "STOP").exists():
            raise BudgetStop("STOP marker requested a stop at the next job boundary")
        if datetime.now().astimezone().timestamp() + estimated_seconds + 60 >= self.deadline.timestamp():
            raise BudgetStop("Insufficient time for another bounded job before phase deadline")
        for source in self.frozen_sources:
            if hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest().lower() != source["sha256"].lower():
                raise ValueError(f"Frozen training source changed: {source['path']}")

    def kill_own_child(self):
        if self.child is not None and self.child.poll() is None:
            subprocess.run(["taskkill", "/PID", str(self.child.pid), "/T", "/F"], capture_output=True)
            self.child.wait(timeout=30)

    def call(self, module, arguments, label, estimate):
        self.guard(estimate)
        self.job = {"label": label, "started_at": now(), "command": [sys.executable, "-m", module, *map(str, arguments)]}
        self.event("job_started", **self.job)
        log = self.campaign / "logs" / f"{self.invocation}_{label}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as stream:
            self.child = subprocess.Popen(self.job["command"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                          creationflags=subprocess.CREATE_NO_WINDOW)
            last_status = 0.
            try:
                while self.child.poll() is None:
                    if datetime.now().astimezone().timestamp() >= self.deadline.timestamp():
                        self.kill_own_child()
                        raise BudgetStop("Phase deadline reached; stopped only the launched process tree")
                    if time.monotonic() - last_status > 30:
                        live = {"status": "running", "deadline": self.deadline.isoformat(), "job": self.job,
                                "updated_at": now(), "child_pid": self.child.pid}
                        write_json(self.campaign / "status.json", live)
                        last_status = time.monotonic()
                    time.sleep(1)
            except BaseException:
                self.kill_own_child()
                raise
        code = self.child.returncode
        self.event("job_finished", label=label, exit_code=code, log=str(log.relative_to(ROOT)))
        if code:
            raise RuntimeError(f"{label} failed (exit {code}); inspect {log}")

    def checkpoint(self, arm, seed):
        current = self.stats(arm, seed)
        path = ROOT / "runs" / self.run_name(arm, seed) / "checkpoints" / f"ppo_iter_{current['last_iteration']:04d}.pt"
        if not path.exists():
            raise RuntimeError(f"Metrics endpoint has no matching checkpoint: {path}; no silent rollback")
        return path
