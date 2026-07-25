"""训练运行产物落盘 — 一次实验一个自包含目录。

历史教训：早期实验把 checkpoint 丢进 `checkpoints_<tag>/`、效率图丢进 `logs_<tag>/`，
指标只存在控制台里，精英池只存在内存里。结果是 13 组实验跑完后，"这次用了什么参数、
曲线长什么样、精英轨迹是什么形状"三样全部无法复原，只能靠翻终端 scrollback 抢救。

本模块把一次运行的全部产物收敛到 `runs/<tag>/`：

    runs/<tag>/
      checkpoints/    ppo_iter_*.pt + cache/     （重，gitignore）
      logs/           ppo_effmap_iter_*.png      （重，gitignore）
      elites/         elites_iter_*.npz          （中，gitignore）
      console.log     全量控制台输出              （重，gitignore）
      run_meta.json   argv + config + git commit （轻，入库）
      metrics.csv     每 iter 指标                （轻，入库）
      RESULT.md       结论                        （轻，入库）

轻量的三样入库，重资产忽略。这样即使权重被清掉，"做了什么实验、结果如何"仍然永久可查。
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

RUNS_ROOT = "runs"

META_NAME = "run_meta.json"
METRICS_NAME = "metrics.csv"
CONSOLE_NAME = "console.log"
RESULT_NAME = "RESULT.md"

_LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
_LOG_DATEFMT = "%H:%M:%S"


@dataclass(frozen=True)
class RunPaths:
    """一次运行的目录布局。"""

    tag: str
    root: Path
    checkpoints: Path
    logs: Path
    elites: Path

    @property
    def meta(self) -> Path:
        return self.root / META_NAME

    @property
    def metrics(self) -> Path:
        return self.root / METRICS_NAME

    @property
    def console(self) -> Path:
        return self.root / CONSOLE_NAME

    @property
    def result(self) -> Path:
        return self.root / RESULT_NAME

    def mkdirs(self) -> None:
        for d in (self.root, self.checkpoints, self.logs, self.elites):
            d.mkdir(parents=True, exist_ok=True)


def resolve_run_paths(run_dir: str | Path, repo_root: Path | None = None) -> RunPaths:
    """把 `--run-dir` 的取值解析成目录布局。

    裸名字（`ou_me`）落到 `runs/ou_me`；带分隔符的当作路径原样使用，便于把产物写到
    仓库外的大盘。
    """
    raw = str(run_dir)
    candidate = Path(raw)
    if candidate.is_absolute() or "/" in raw or "\\" in raw:
        root = candidate
    else:
        base = Path(repo_root) if repo_root is not None else Path.cwd()
        root = base / RUNS_ROOT / raw
    root = root.resolve()
    return RunPaths(
        tag=root.name,
        root=root,
        checkpoints=root / "checkpoints",
        logs=root / "logs",
        elites=root / "elites",
    )


def git_commit(repo_root: Path | None = None) -> str | None:
    """当前 HEAD 的短 hash，带 `-dirty` 后缀表示工作区有未提交改动。"""
    cwd = str(repo_root) if repo_root is not None else None
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=10,
        )
        if head.returncode != 0:
            return None
        commit = head.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd, capture_output=True, text=True, timeout=15,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            commit += "-dirty"
        return commit
    except (OSError, subprocess.SubprocessError):
        return None


def _num_or_str(value: str) -> Any:
    try:
        return float(value)
    except ValueError:
        return value


def _jsonable(value: Any) -> Any:
    """把 config 里的非 JSON 原生类型降级成可序列化形式。"""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return repr(value)


def save_elites(path: Path, trajectories: Sequence[Any]) -> int:
    """把 SIL 精英池落盘成单个 npz。

    每条轨迹按 `t{i}_states / t{i}_actions / t{i}_contact` 存；分数与来源迭代存成
    并列数组。压缩后一池（cap=20 × 500 步）约 1~3 MB，相对 13.5MB 的 checkpoint
    可以忽略，却是"策略当时到底走出了什么行为"的唯一记录。
    """
    if not trajectories:
        return 0
    arrays: dict[str, np.ndarray] = {}
    scores, iters = [], []
    for i, traj in enumerate(trajectories):
        arrays[f"t{i}_states"] = np.asarray(traj.raw_states, dtype=np.float32)
        arrays[f"t{i}_actions"] = np.asarray(traj.actions, dtype=np.float32)
        contact = getattr(traj, "contact", None)
        if contact is not None:
            arrays[f"t{i}_contact"] = np.asarray(contact, dtype=np.float32)
        scores.append(float(getattr(traj, "score", 0.0)))
        iters.append(int(getattr(traj, "iteration", 0)))
    arrays["scores"] = np.asarray(scores, dtype=np.float32)
    arrays["iterations"] = np.asarray(iters, dtype=np.int32)
    arrays["n_trajectories"] = np.asarray(len(trajectories), dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return len(trajectories)


def load_elites(path: str | Path) -> list[dict[str, Any]]:
    """读回 save_elites 写的精英池快照。

    返回纯 dict 列表而非 Trajectory，避免分析脚本被迫依赖训练侧的数据类定义。
    """
    with np.load(path) as data:
        n = int(data["n_trajectories"])
        scores = data["scores"]
        iters = data["iterations"]
        out = []
        for i in range(n):
            out.append({
                "raw_states": data[f"t{i}_states"],
                "actions": data[f"t{i}_actions"],
                "contact": data[f"t{i}_contact"] if f"t{i}_contact" in data else None,
                "score": float(scores[i]),
                "iteration": int(iters[i]),
            })
    return out


class RunRecorder:
    """一次运行的产物记录器：元数据 / 指标 / 控制台 / 精英池。

    `metrics.csv` 的列在运行中会变（预热轮没有 sil_*，非预热轮才有），所以行全部
    留在内存里，每次追加都按列的并集整文件重写。行数最多几百，代价可以忽略，换来
    的是任何时刻中断都能得到一个列对齐的完整 CSV。
    """

    def __init__(self, paths: RunPaths, *, repo_root: Path | None = None) -> None:
        self.paths = paths
        self._repo_root = repo_root
        self._rows: list[dict[str, Any]] = []
        self._columns: list[str] = ["iteration"]
        self._pending: dict[str, Any] = {}
        self._handler: logging.Handler | None = None
        self._meta: dict[str, Any] = {}
        paths.mkdirs()
        self._load_existing_metrics()

    def _load_existing_metrics(self) -> None:
        """续训时接着已有的 metrics.csv 写，而不是从头覆盖。

        `--resume` 到同一个 run-dir 是常规操作，如果每次都重建 CSV，一条跑了几百轮
        的实验只要中途续过一次，前半段曲线就没了。
        """
        if not self.paths.metrics.is_file():
            return
        try:
            with self.paths.metrics.open("r", encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    parsed = {k: _num_or_str(v) for k, v in row.items() if v != ""}
                    if "iteration" not in parsed:
                        continue
                    parsed["iteration"] = int(parsed["iteration"])
                    self._rows.append(parsed)
                    for key in parsed:
                        if key not in self._columns:
                            self._columns.append(key)
        except (OSError, csv.Error, ValueError):
            logger.warning("已有 metrics.csv 无法解析，将从空表重建: %s",
                           self.paths.metrics)
            self._rows, self._columns = [], ["iteration"]

    # ── 元数据 ──

    def write_meta(self, config: Any, argv: Sequence[str] | None = None,
                   extra: dict[str, Any] | None = None) -> None:
        cfg = asdict(config) if is_dataclass(config) else dict(vars(config))
        self._meta = {
            "tag": self.paths.tag,
            "status": "running",
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "finished_at": None,
            "last_iteration": None,
            "argv": list(argv if argv is not None else sys.argv),
            "git_commit": git_commit(self._repo_root),
            "python": platform.python_version(),
            "host": platform.node(),
            "config": {k: _jsonable(v) for k, v in cfg.items()},
        }
        if extra:
            self._meta.update({k: _jsonable(v) for k, v in extra.items()})
        self._flush_meta()

    def update_meta(self, **fields: Any) -> None:
        if not self._meta:
            return
        self._meta.update({k: _jsonable(v) for k, v in fields.items()})
        self._flush_meta()

    def finalize(self, status: str, last_iteration: int | None = None,
                 error: str | None = None) -> None:
        self.update_meta(
            status=status,
            finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            last_iteration=last_iteration,
            error=error,
        )
        self.detach_console()

    def _flush_meta(self) -> None:
        self.paths.meta.write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    # ── 控制台 ──

    def attach_console(self, level: int = logging.INFO) -> None:
        """把 root logger 同时写进 `console.log`，取代手工 `Tee-Object`。"""
        if self._handler is not None:
            return
        handler = logging.FileHandler(self.paths.console, mode="a", encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
        logging.getLogger().addHandler(handler)
        self._handler = handler

    def detach_console(self) -> None:
        if self._handler is None:
            return
        logging.getLogger().removeHandler(self._handler)
        self._handler.close()
        self._handler = None

    # ── 指标 ──

    def stage_metrics(self, **fields: Any) -> None:
        """暂存指标，等本轮 `append_metrics` 一并落盘。

        供训练循环里早于 `log_metrics` 产生的指标使用（MAP-Elites 的 cells、探索
        boost 等），让它们和同一 iteration 的 PPO 指标落在同一行。
        """
        self._pending.update(fields)

    def append_metrics(self, iteration: int, *metric_dicts: dict[str, Any] | None) -> None:
        row: dict[str, Any] = {"iteration": int(iteration)}
        row.update(self._pending)
        self._pending = {}
        for d in metric_dicts:
            if d:
                row.update(d)
        # 续训可能重跑已有轮次，同 iteration 以新值为准而不是并存两行
        for existing in self._rows:
            if existing["iteration"] == row["iteration"]:
                existing.update(row)
                break
        else:
            self._rows.append(row)
            self._rows.sort(key=lambda r: r["iteration"])
        for key in row:
            if key not in self._columns:
                self._columns.append(key)
        self._flush_metrics()

    def _flush_metrics(self) -> None:
        with self.paths.metrics.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._rows)

    # ── 精英池 ──

    def save_elites(self, trajectories: Iterable[Any], iteration: int) -> None:
        trajs = list(trajectories)
        if not trajs:
            return
        path = self.paths.elites / f"elites_iter_{iteration:04d}.npz"
        n = save_elites(path, trajs)
        logger.info("精英池快照已保存: %s (%d 条轨迹)", path, n)
