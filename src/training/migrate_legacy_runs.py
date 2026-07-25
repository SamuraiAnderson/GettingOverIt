"""把散落的 `checkpoints_<tag>/` + `logs_<tag>/` 归档成 `runs/<tag>/`。

一次性迁移工具，但保留在仓库里：它同时是这批历史实验的**索引**——
`LEGACY_RUNS` 表记录了每组实验的意图与状态，是从终端 scrollback 里考据出来的，
删掉就再也拼不回来了。

迁移动作：
  checkpoints_<tag>/*.pt   → runs/<tag>/checkpoints/   （按 --keep 策略瘦身）
  checkpoints_<tag>/cache/ → runs/<tag>/checkpoints/cache/
  logs_<tag>/*.png         → runs/<tag>/logs/
  控制台输出                → runs/<tag>/console.log     （Tee 文件优先，否则拼终端记录）
  解析控制台                → runs/<tag>/metrics.csv     （抢救丢失的训练曲线）
  考据出的 CLI + config    → runs/<tag>/run_meta.json
  意图 + 客观结果摘要       → runs/<tag>/RESULT.md

标记为 aborted 的实验只保留轻量记录（meta/metrics/console/RESULT），权重与效率图删除。

用法:
  python -m src.training.migrate_legacy_runs                # 预演，不改动任何文件
  python -m src.training.migrate_legacy_runs --apply
  python -m src.training.migrate_legacy_runs --apply --keep all
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from training.run_log_parser import parse_console_log, write_metrics_csv
from training.run_logging import CONSOLE_NAME, META_NAME, METRICS_NAME, RESULT_NAME, RUNS_ROOT

_TERMINALS_DIR = Path(
    r"C:\Users\Symbol\.cursor\projects\c-Users-Symbol-aCodes-GettingOverIt\terminals"
)


@dataclass(frozen=True)
class LegacyRun:
    tag: str
    intent: str
    status: str  # "completed" | "aborted"
    note: str = ""
    # 早期跑在默认输出目录（没给 --checkpoint-dir）的实验，产物和共享资产混在
    # checkpoints/ 与 logs/ 里，需要显式指定源目录和文件模式才能摘出来。
    src_checkpoints: str | None = None
    src_logs: str | None = None
    ckpt_glob: str = "*.pt"
    log_glob: str = "*.png"
    prune_source_dirs: bool = True
    # 没给 --checkpoint-dir 的实验无法按目录名认领终端记录，改用命令行子串匹配
    terminal_match: str | None = None


# 意图与状态从终端 scrollback 的启动命令 + 路线图引用考据而来。
LEGACY_RUNS: tuple[LegacyRun, ...] = (
    LegacyRun(
        "bcseed", "BC 种子 + 随机投放的首轮 PPO 试跑", "completed",
        "最早一批 random-deploy 实验，后被固定起点最小任务（P0.1）取代。",
    ),
    LegacyRun(
        "bcseed2", "bcseed 重跑，加 --y-max-cutoff 125 限定低处投放区", "aborted",
        "跑到 iter5 即中止，被 bcseed3 以同配方接手。",
    ),
    LegacyRun(
        "bcseed3", "bcseed2 配方重跑", "aborted",
        "跑到 iter5 即中止，被 bcseed4 接手。",
    ),
    LegacyRun(
        "bcseed4", "bcseed3 配方 + 显式 --max-iterations 10", "aborted",
        "跑到 iter5 即中止。此后放弃 random-deploy 路线，转向固定起点最小任务。",
    ),
    LegacyRun(
        "start1", "P0.1 固定起点最小任务首试", "aborted",
        "跑到 iter5 即中止；发现 dropout 未关，由 start2 修复后重跑。",
    ),
    LegacyRun(
        "start2", "start1 + dropout-off 修复", "aborted",
        "跑到 iter10 即中止，配方由 start_long 接手做长跑。",
    ),
    LegacyRun(
        "start_long", "固定起点长跑 baseline（60 轮）", "completed",
        "SIL 对照组的 baseline，eval 记录见 runs/_eval/。",
    ),
    LegacyRun(
        "start_sil", "固定起点 + SIL 精英池（190 轮）", "completed",
        "路线图瓶颈判断的证据之一：iter190 效率图价值仍只在起点盆地。",
    ),
    LegacyRun(
        "treefix", "地形 mask 3x3 子采样修复后重跑（110 轮）", "completed",
        "验证 mask 混叠修复；由 treefix2 叠加 PBRS gamma=1 继续。",
    ),
    LegacyRun(
        "treefix2", "mask 3x3 + PBRS gamma=1（190 轮）", "completed",
        "路线图瓶颈判断的主要证据：精英轨迹 top-10 全部在起点附近横向游走，无一翻越 Deadtree。",
    ),
    LegacyRun(
        "treefix2cont", "treefix2 从 iter190 续训到 260", "completed",
        "路线图证据：跑到 260 轮仍无纵向价值传播，确认平台期非训练时长不足。",
    ),
    LegacyRun(
        "treefix3", "treefix2 配方重跑", "aborted",
        "跑到 iter5 即中止。",
    ),
    LegacyRun(
        "ou_me", "P1.1 OU 连贯噪声 + P1.2 MAP-Elites 精英池", "completed",
        "首个 sprint 的验证跑；结论是 4.28m 天花板，触发转向 P2.1 接触信号。",
    ),
    LegacyRun(
        "default_randomdeploy",
        "跑在默认输出目录的随机投放 PPO（16 agents，y-max-cutoff 125）", "completed",
        "早于 --checkpoint-dir 约定，产物混在 checkpoints/ 与 logs/ 里。"
        "只摘走 ppo_iter_*.pt 与 ppo_effmap_*.png；BC 权重、warmup_state 与"
        "环境资产（environment.json 等）留在原地，它们是共享输入而非本次产物。",
        src_checkpoints="checkpoints",
        src_logs="logs",
        ckpt_glob="ppo_iter_*.pt",
        log_glob="ppo_effmap_iter_*.png",
        prune_source_dirs=False,
        terminal_match="main_ppo --num-agents 16 --random-deploy",
    ),
)


_CMD_RE = re.compile(r"^command:\s*(.*)$")
_START_RE = re.compile(r"^started_at:\s*\"?([^\"\s]+)")
_CKPT_DIR_RE = re.compile(r"--checkpoint-dir\s+checkpoints_([A-Za-z0-9_]+)")


def _terminal_header(path: Path) -> tuple[str | None, str | None]:
    """读终端记录开头的 command / started_at 元数据。"""
    try:
        head = path.read_text(encoding="utf-8", errors="replace").split("\n", 12)[:12]
    except OSError:
        return None, None
    command = started = None
    for line in head:
        if command is None and (m := _CMD_RE.match(line)):
            command = m.group(1)
        if started is None and (m := _START_RE.match(line)):
            started = m.group(1)
    return command, started


def discover_terminal_logs(terminals_dir: Path) -> dict[str, list[tuple[str, Path]]]:
    """扫描终端记录，按 `--checkpoint-dir checkpoints_<tag>` 归类到实验 tag。

    返回 tag → [(started_at, path)]，按启动时间升序（同一实验可能有多次重启）。
    """
    found: dict[str, list[tuple[str, Path]]] = {}
    if not terminals_dir.is_dir():
        return found

    for path in terminals_dir.glob("*.txt"):
        command, started = _terminal_header(path)
        if not command or "main_ppo" not in command:
            continue
        if m := _CKPT_DIR_RE.search(command):
            found.setdefault(m.group(1), []).append((started or "", path))

    for entries in found.values():
        entries.sort(key=lambda e: e[0])
    return found


def find_terminals_by_command(terminals_dir: Path, needle: str) -> list[tuple[str, Path]]:
    """按命令行子串认领终端记录，供没给 --checkpoint-dir 的早期实验使用。"""
    if not terminals_dir.is_dir():
        return []
    hits = []
    for path in terminals_dir.glob("*.txt"):
        command, started = _terminal_header(path)
        if command and needle in command:
            hits.append((started or "", path))
    hits.sort(key=lambda e: e[0])
    return hits


def _extract_meta_from_console(text: str) -> dict[str, object]:
    """从控制台文本里捞出 CLI 与 config 快照。

    config 原样存字符串而不解析：`TrainConfig(...)` 的 repr 里有嵌套元组和字符串，
    半吊子的解析只会引入静默错误，而这里的用途是人读与 diff。
    """
    meta: dict[str, object] = {}
    if m := re.search(r"^command:\s*(.*?)\s*$", text, re.MULTILINE):
        raw = m.group(1)
        # 终端记录里的 command 是 JSON 字符串，直接取会带一身 \" 转义
        try:
            meta["command"] = json.loads(raw)
        except json.JSONDecodeError:
            meta["command"] = raw.strip('"')
    if m := re.search(r"PPO Config: (TrainConfig\(.*)$", text, re.MULTILINE):
        meta["config_repr"] = m.group(1).strip()
    if m := re.search(r"从 PPO checkpoint 恢复 \(iteration=(\d+)\): (\S+)", text):
        meta["resumed_from"] = m.group(2)
        meta["resumed_at_iteration"] = int(m.group(1))
    if m := re.search(r"从 BC checkpoint 迁移 backbone 权重: (\S+)", text):
        meta["resumed_bc"] = m.group(1)
    return meta


def _checkpoint_iter(path: Path) -> int:
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def _select_checkpoints(ckpts: list[Path], keep: str) -> tuple[list[Path], list[Path]]:
    """按保留策略切分成 (保留, 删除)。"""
    if not ckpts:
        return [], []
    ordered = sorted(ckpts, key=_checkpoint_iter)
    if keep == "all":
        return ordered, []
    if keep == "last":
        return ordered[-1:], ordered[:-1]
    if keep == "every25":
        kept = [c for c in ordered if _checkpoint_iter(c) % 25 == 0]
        if ordered[-1] not in kept:
            kept.append(ordered[-1])
        return kept, [c for c in ordered if c not in kept]
    raise ValueError(f"未知保留策略: {keep}")


def _summarize_metrics(rows: list[dict]) -> dict[str, object]:
    """从指标行里提炼客观结果摘要，供 RESULT.md 使用。"""
    if not rows:
        return {}
    iters = [r["iteration"] for r in rows]
    out: dict[str, object] = {
        "iterations_recovered": len(rows),
        "iteration_range": [min(iters), max(iters)],
    }
    for key in ("mean_reward", "explore_pool_best_sdy", "me_cells", "value_loss"):
        vals = [r[key] for r in rows if key in r and r[key] == r[key]]
        if vals:
            out[f"{key}_final"] = vals[-1]
            out[f"{key}_max"] = max(vals)
    return out


_RESULT_TEMPLATE = """# {tag}

**意图**：{intent}
**状态**：{status}
**时间**：{when}

{note}

## 命令

```
{command}
```

## 客观结果

{summary}

## 结论

{conclusion}
"""


def _existing_conclusion(path: Path) -> str | None:
    """取回 RESULT.md 里手写的结论段，避免重新渲染时被模板覆盖。"""
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    _, sep, tail = text.partition("\n## 结论\n")
    if not sep:
        return None
    body = tail.strip()
    return body or None


def _render_result(run: LegacyRun, meta: dict, summary: dict,
                   conclusion: str | None = None) -> str:
    if summary:
        lines = []
        rng = summary.get("iteration_range")
        if rng:
            lines.append(f"- 恢复到 {summary['iterations_recovered']} 轮指标"
                         f"（iter {rng[0]}–{rng[1]}）")
        for key, label in (
            ("mean_reward", "mean_reward"),
            ("explore_pool_best_sdy", "精英池最佳 secured_dy"),
            ("me_cells", "MAP-Elites 占据 cell 数"),
        ):
            if f"{key}_final" in summary:
                lines.append(
                    f"- {label}：末轮 {summary[f'{key}_final']:.4g}，"
                    f"峰值 {summary[f'{key}_max']:.4g}"
                )
        summary_text = "\n".join(lines)
    else:
        summary_text = "_控制台输出未留存，指标无法恢复。_"

    if conclusion is None:
        conclusion = (
            "_未记录（实验中止，无有效结论）。_" if run.status == "aborted"
            else "_待补。_"
        )
    return _RESULT_TEMPLATE.format(
        tag=run.tag,
        intent=run.intent,
        status="已完成" if run.status == "completed" else "中止",
        when=meta.get("started_at") or "未知",
        note=run.note,
        command=meta.get("command", "未考据到启动命令"),
        summary=summary_text,
        conclusion=conclusion,
    )


def migrate_one(run: LegacyRun, repo: Path, terminal_logs: list[tuple[str, Path]],
                keep: str, apply: bool) -> dict[str, object]:
    src_ckpt = repo / (run.src_checkpoints or f"checkpoints_{run.tag}")
    src_logs = repo / (run.src_logs or f"logs_{run.tag}")
    dst = repo / RUNS_ROOT / run.tag

    report: dict[str, object] = {"tag": run.tag, "status": run.status}

    # 已归档过就跳过：重跑 --apply 不能把手写进 RESULT.md 的结论冲掉。
    # 要按新的 LEGACY_RUNS 描述重新渲染，走 --refresh。
    if (dst / META_NAME).is_file() and not any(
        (repo / d).is_dir() and any((repo / d).glob(g))
        for d, g in ((src_ckpt.name, run.ckpt_glob), (src_logs.name, run.log_glob))
    ):
        report.update(skipped=True, ckpt_kept=[], ckpt_dropped=0,
                      png_kept=0, bytes_freed=0, iterations_recovered=0)
        return report

    # 1. 控制台：根目录的 Tee 文件最完整（终端 scrollback 可能被截断），优先采用
    console_text = ""
    console_sources: list[str] = []
    tee = repo / f"logs_{run.tag}_run.txt"
    if tee.is_file():
        console_text = tee.read_text(encoding="utf-8", errors="replace")
        console_sources.append(tee.name)
    else:
        chunks = []
        for started, path in terminal_logs:
            chunks.append(
                f"# ==== 归档自终端记录 {path.name} (started_at={started}) ====\n"
                + path.read_text(encoding="utf-8", errors="replace")
            )
            console_sources.append(f"{path.name}@{started}")
        console_text = "\n".join(chunks)

    meta_from_log = _extract_meta_from_console(console_text) if console_text else {}
    if terminal_logs and "started_at" not in meta_from_log:
        meta_from_log["started_at"] = terminal_logs[0][0]

    rows = []
    if console_text:
        if apply:
            dst.mkdir(parents=True, exist_ok=True)
            console_path = dst / CONSOLE_NAME
            console_path.write_text(console_text, encoding="utf-8")
            rows = parse_console_log(console_path)
        else:
            rows = _parse_text(console_text)
    report["console_sources"] = console_sources
    report["iterations_recovered"] = len(rows)

    # 2. checkpoints
    ckpts = sorted(src_ckpt.glob(run.ckpt_glob)) if src_ckpt.is_dir() else []
    if run.status == "aborted":
        kept, dropped = [], ckpts
    else:
        kept, dropped = _select_checkpoints(ckpts, keep)
    report["ckpt_kept"] = [p.name for p in kept]
    report["ckpt_dropped"] = len(dropped)
    report["bytes_freed"] = sum(p.stat().st_size for p in dropped)

    # 3. 效率图
    pngs = sorted(src_logs.glob(run.log_glob)) if src_logs.is_dir() else []
    png_kept = [] if run.status == "aborted" else pngs
    report["png_kept"] = len(png_kept)

    if not apply:
        return report

    (dst / "checkpoints").mkdir(parents=True, exist_ok=True)
    (dst / "logs").mkdir(parents=True, exist_ok=True)
    for p in kept:
        shutil.move(str(p), str(dst / "checkpoints" / p.name))
    for p in png_kept:
        shutil.move(str(p), str(dst / "logs" / p.name))
    cache = src_ckpt / "cache"
    if cache.is_dir() and run.status != "aborted" and run.prune_source_dirs:
        shutil.move(str(cache), str(dst / "checkpoints" / "cache"))
    if tee.is_file():
        tee.unlink()

    # 4. 轻量记录
    summary = _summarize_metrics(rows)
    if rows:
        write_metrics_csv(rows, dst / METRICS_NAME)
    meta = {
        "tag": run.tag,
        "intent": run.intent,
        "status": run.status,
        "note": run.note,
        "archived_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "archived_from": [f"checkpoints_{run.tag}", f"logs_{run.tag}"],
        "console_sources": console_sources,
        "checkpoints_kept": [p.name for p in kept],
        "checkpoints_dropped": len(dropped),
        "keep_policy": "none (aborted)" if run.status == "aborted" else keep,
        "metrics_summary": summary,
        **meta_from_log,
    }
    (dst / META_NAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (dst / RESULT_NAME).write_text(_render_result(run, meta, summary), encoding="utf-8")

    # 5. 清空源目录
    for p in dropped:
        p.unlink()
    for p in pngs:
        if p.exists() and p not in png_kept:
            p.unlink()
    if run.prune_source_dirs:
        for src in (src_ckpt, src_logs):
            if src.is_dir():
                shutil.rmtree(src, ignore_errors=True)
    return report


def refresh_one(run: LegacyRun, repo: Path) -> dict[str, object]:
    """从已归档的 console.log 重新生成 metrics/meta/RESULT。

    迁移是一次性的，但索引不是：改了 `LEGACY_RUNS` 里的意图描述、或补完某组实验的
    结论后，用这个模式重新渲染即可，不必也不能再跑一遍迁移。手写的结论段会保留。
    """
    dst = repo / RUNS_ROOT / run.tag
    report: dict[str, object] = {
        "tag": run.tag, "status": run.status,
        "ckpt_kept": [], "ckpt_dropped": 0, "png_kept": 0,
        "bytes_freed": 0, "iterations_recovered": 0,
    }
    if not dst.is_dir():
        return report

    console = dst / CONSOLE_NAME
    rows = parse_console_log(console) if console.is_file() else []
    summary = _summarize_metrics(rows)
    if rows:
        write_metrics_csv(rows, dst / METRICS_NAME)

    kept = sorted((dst / "checkpoints").glob("*.pt"))
    report["ckpt_kept"] = [p.name for p in kept]
    report["png_kept"] = len(list((dst / "logs").glob("*.png")))
    report["iterations_recovered"] = len(rows)

    old_meta = {}
    if (dst / META_NAME).is_file():
        old_meta = json.loads((dst / META_NAME).read_text(encoding="utf-8"))
    meta = {
        **old_meta,
        "tag": run.tag,
        "intent": run.intent,
        "status": run.status,
        "note": run.note,
        "checkpoints_kept": [p.name for p in kept],
        "metrics_summary": summary,
        **(_extract_meta_from_console(console.read_text(encoding="utf-8", errors="replace"))
           if console.is_file() else {}),
    }
    (dst / META_NAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (dst / RESULT_NAME).write_text(
        _render_result(run, meta, summary, _existing_conclusion(dst / RESULT_NAME)),
        encoding="utf-8",
    )
    return report


def _parse_text(text: str) -> list[dict]:
    """预演路径：不落盘地解析控制台文本。"""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".log", encoding="utf-8",
                                     delete=False) as fh:
        fh.write(text)
        tmp = fh.name
    try:
        return parse_console_log(tmp)
    finally:
        Path(tmp).unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="归档历史训练产物到 runs/<tag>/")
    ap.add_argument("--apply", action="store_true", help="真正移动/删除文件（默认只预演）")
    ap.add_argument("--keep", choices=["last", "every25", "all"], default="last",
                    help="已完成实验的 checkpoint 保留策略（默认只留最后一个）")
    ap.add_argument("--refresh", action="store_true",
                    help="不迁移，只从已归档的 console.log 重新生成 metrics/meta/RESULT")
    ap.add_argument("--terminals-dir", default=str(_TERMINALS_DIR),
                    help="终端记录目录，用于抢救控制台输出")
    args = ap.parse_args()

    repo = _REPO_ROOT
    if args.refresh:
        for run in LEGACY_RUNS:
            report = refresh_one(run, repo)
            print(f"{run.tag:<14} 指标{report['iterations_recovered']:>4}轮 "
                  f"ckpt{len(report['ckpt_kept'])} png{report['png_kept']}")
        return 0

    terminal_logs = discover_terminal_logs(Path(args.terminals_dir))
    print(f"终端记录命中 {len(terminal_logs)} 组实验: {', '.join(sorted(terminal_logs))}\n")

    total_freed = 0
    for run in LEGACY_RUNS:
        if run.terminal_match:
            logs_for_run = find_terminals_by_command(
                Path(args.terminals_dir), run.terminal_match,
            )
        else:
            logs_for_run = terminal_logs.get(run.tag, [])
        report = migrate_one(run, repo, logs_for_run, args.keep, args.apply)
        total_freed += int(report["bytes_freed"])
        if report.get("skipped"):
            print(f"{run.tag:<21} 已归档，跳过")
            continue
        print(
            f"{run.tag:<21} {report['status']:<10} "
            f"ckpt留{len(report['ckpt_kept']):>2}/删{report['ckpt_dropped']:<3} "
            f"png留{report['png_kept']:<3} "
            f"指标{report['iterations_recovered']:>4}轮 "
            f"释放{report['bytes_freed'] / 1e6:>7.1f}MB"
        )

    print(f"\n{'已释放' if args.apply else '预计释放'}: {total_freed / 1e6:.1f} MB")
    if not args.apply:
        print("这是预演，未改动任何文件。加 --apply 执行。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
