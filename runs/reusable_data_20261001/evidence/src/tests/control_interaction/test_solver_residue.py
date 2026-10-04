"""
求解器残留对照：刚载入 vs 搅动后再 reset vs 新进程。

每个剂量单独启动一次游戏。可见快照保持初始化时拍下的那一份，
搅动之后不调用 new_snapshot()。指标是剔除 timestamp 后的早期残差，
不是长时域末端的最大差。

  live     连上后直接跑探针，不 reset
  B=0      连上后立刻 reset，再跑探针（只有 Reset 自带的 settle）
  B>0      先跑 B 步相同的剧烈动作，再 reset，再跑同一探针

每种剂量跑两轮探针，中间再 reset 一次，用来看第二次初始化是否回到冷启动。

用法：
  conda run -n getting-over-it-analysis python -u src/tests/control_interaction/test_solver_residue.py
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from env.goi_env import GoiEnv
from start.game_launcher import GameLauncher
from start.game_mode_controller import GameModeController
from training.config import TrainConfig
from training.rollout import _write_num_duplicates

logger = logging.getLogger("solver_residue")

TIMESTAMP_IDX = 28
# player xy、tip xy：检查 reset 有没有把可见位置拉回同一快照
POSE_IDX = (0, 1, 23, 24)
# player 速度与角速度：settle 那一帧的冲量会先出现在这里
VEL_IDX = (2, 3, 4)
HOLD = np.array([40.0, 30.0], dtype=np.float32)
THRESHOLDS = (1e-4, 1e-3, 1e-2, 0.1)
SAMPLE_STEPS = (1, 5, 10, 20, 40)


def _cols(dim: int) -> np.ndarray:
    return np.array([i for i in range(dim) if i != TIMESTAMP_IDX], dtype=int)


def _max_abs(a: np.ndarray, b: np.ndarray, cols: np.ndarray) -> float:
    return float(np.abs(a[..., cols] - b[..., cols]).max())


def _subset_max(a: np.ndarray, b: np.ndarray, idx: tuple[int, ...]) -> float:
    return float(np.abs(a[..., list(idx)] - b[..., list(idx)]).max())


def _cross_agent(obs: np.ndarray, cols: np.ndarray) -> float:
    ref = np.repeat(obs[0:1], obs.shape[0], axis=0)
    return _max_abs(obs, ref, cols)


def _worst_field(a: np.ndarray, b: np.ndarray, cols: np.ndarray) -> tuple[int, float]:
    diff = np.abs(a[..., cols] - b[..., cols])
    flat = diff.reshape(-1, diff.shape[-1])
    col = int(flat.max(axis=0).argmax())
    return int(cols[col]), float(flat[:, col].max())


def _crossings(curve: list[float]) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for threshold in THRESHOLDS:
        step = None
        for i, value in enumerate(curve):
            if value > threshold:
                step = i + 1
                break
        out[f"{threshold:.0e}"] = step
    return out


def _samples(curve: list[float]) -> dict[str, float | None]:
    return {f"s{t}": (curve[t - 1] if t <= len(curve) else None) for t in SAMPLE_STEPS}


def _pose(obs0: np.ndarray) -> dict[str, float]:
    return {
        "x": float(obs0[0]),
        "y": float(obs0[1]),
        "vx": float(obs0[2]),
        "vy": float(obs0[3]),
        "w": float(obs0[4]),
        "tip_x": float(obs0[23]),
        "tip_y": float(obs0[24]),
        "hammer": float(obs0[27]),
    }


def _kill_game() -> None:
    subprocess.run(
        ["taskkill", "/F", "/IM", "GettingOverIt.exe", "/T"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _stop_process(proc) -> None:
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
    except Exception:
        logger.warning("结束游戏进程失败", exc_info=True)
    try:
        proc.wait(timeout=15)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    _kill_game()
    time.sleep(1.5)


def _actions(kind: str, n: int) -> np.ndarray:
    if kind == "zero":
        return np.zeros((n, 2), dtype=np.float32)
    if kind == "hold":
        return np.tile(HOLD, (n, 1))
    raise ValueError(kind)


def _probe(env: GoiEnv, n: int, kind: str, horizon: int, cols: np.ndarray) -> dict:
    act = _actions(kind, n)
    curve: list[float] = []
    traj0 = []
    last = None
    for _ in range(horizon):
        obs, _ = env.step(act)
        last = obs
        curve.append(_cross_agent(obs, cols))
        traj0.append(obs[0].copy())
    ref = np.repeat(last[0:1], n, axis=0)
    field, field_value = _worst_field(last, ref, cols)
    return {
        "cross": _samples(curve),
        "cross_curve": curve,
        "cross_at": _crossings(curve),
        "worst_field": field,
        "worst_field_value": field_value,
        "traj0": np.stack(traj0),
    }


def _scramble(env: GoiEnv, n: int, steps: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for i in range(steps):
        one = rng.uniform(-100.0, 100.0, size=(2,)).astype(np.float32)
        env.step(np.tile(one, (n, 1)))
        if steps >= 200 and (i + 1) % 200 == 0:
            logger.info("  搅动 %d/%d", i + 1, steps)


def _serial(first: dict, second: dict, cols: np.ndarray) -> dict:
    a = first["traj0"]
    b = second["traj0"]
    curve = [_max_abs(a[t], b[t], cols) for t in range(len(a))]
    field, field_value = _worst_field(a, b, cols)
    return {
        "serial": _samples(curve),
        "serial_curve": curve,
        "serial_at": _crossings(curve),
        "worst_field": field,
        "worst_field_value": field_value,
    }


def _arm_record(probe: dict, reset_obs: np.ndarray | None, cols: np.ndarray) -> dict:
    record = {
        "cross": probe["cross"],
        "cross_at": probe["cross_at"],
        "worst_field": probe["worst_field"],
        "worst_field_value": probe["worst_field_value"],
    }
    if reset_obs is None:
        record["reset_cross"] = None
        record["pose"] = None
        record["pose_spread"] = None
        record["vel_spread"] = None
    else:
        record["reset_cross"] = _cross_agent(reset_obs, cols)
        record["pose"] = _pose(reset_obs[0])
        record["pose_spread"] = _subset_max(
            reset_obs, np.repeat(reset_obs[0:1], reset_obs.shape[0], axis=0), POSE_IDX
        )
        record["vel_spread"] = _subset_max(
            reset_obs, np.repeat(reset_obs[0:1], reset_obs.shape[0], axis=0), VEL_IDX
        )
    return record


def _run_connected(env: GoiEnv, n: int, horizon: int, burn: int, seed: int, live: bool) -> dict:
    """每种探针都从一次复位重新开始。live 的第一轮 hold 不复位，直接从载入状态走。"""
    env._sock.settimeout(60.0)
    # 读当前刚体，不推进物理。live 用来记下载入姿态；剂量会话不改快照。
    origin = env.new_snapshot() if live else None
    if origin is not None and origin.shape[0] != n:
        raise RuntimeError(f"回包 agent 数 {origin.shape[0]}，期望 {n}")
    if burn > 0:
        logger.info("搅动 %d 步", burn)
        _scramble(env, n, burn, seed)

    # hold 放在前面：它是判定用的探针，必须从本剂量的第一次复位出发，
    # 不能先被零动作探针走脏。
    kinds = ("hold",) if live else ("hold", "zero")
    cols = _cols(origin.shape[-1]) if origin is not None else None
    first_rounds: dict = {}
    second_rounds: dict = {}
    serial: dict = {}
    pose_restore = None
    vel_restore = None
    for kind in kinds:
        pair = []
        reset_pair = []
        for index in range(2):
            use_loaded = live and index == 0
            reset_obs = origin if use_loaded else env.reset()
            if reset_obs.shape[0] != n:
                raise RuntimeError(f"回包 agent 数 {reset_obs.shape[0]}，期望 {n}")
            if cols is None:
                cols = _cols(reset_obs.shape[-1])
            ref = np.repeat(reset_obs[0:1], n, axis=0)
            logger.info(
                "  %s 轮 %d 起点跨 agent=%.3e 位置散布=%.3e 速度散布=%.3e",
                kind, index + 1,
                _cross_agent(reset_obs, cols),
                _subset_max(reset_obs, ref, POSE_IDX),
                _subset_max(reset_obs, ref, VEL_IDX),
            )
            probe = _probe(env, n, kind, horizon, cols)
            logger.info(
                "  %s 轮 %d 跨 agent s1=%s s5=%s s10=%s",
                kind, index + 1,
                _fmt_one(probe["cross"].get("s1")),
                _fmt_one(probe["cross"].get("s5")),
                _fmt_one(probe["cross"].get("s10")),
            )
            arm = _arm_record(probe, reset_obs, cols)
            if index == 0:
                first_rounds[kind] = arm
            else:
                second_rounds[kind] = arm
            pair.append(probe)
            reset_pair.append(reset_obs)
        compared = _serial(pair[0], pair[1], cols)
        del compared["serial_curve"]
        serial[kind] = compared
        logger.info(
            "  %s 跨 reset s1=%s s5=%s s10=%s",
            kind,
            _fmt_one(compared["serial"].get("s1")),
            _fmt_one(compared["serial"].get("s5")),
            _fmt_one(compared["serial"].get("s10")),
        )
        if pose_restore is None:
            pose_restore = _subset_max(reset_pair[0][0], reset_pair[1][0], POSE_IDX)
            vel_restore = _subset_max(reset_pair[0][0], reset_pair[1][0], VEL_IDX)
    return {
        "rounds": [first_rounds, second_rounds],
        "serial": serial,
        "pose_restore": pose_restore,
        "vel_restore": vel_restore,
        "mean_step_s": float(np.mean(env.step_times)) if env.step_times else None,
    }


def _fmt_one(value) -> str:
    if value is None:
        return "—"
    return f"{value:.3e}"


def _session(args, dose: str, burn: int, repeat: int, live: bool) -> dict:
    logger.info("==== %s  重复 %d/%d ====", dose, repeat + 1, args.repeats)
    game_root = Path(TrainConfig().game_root)
    _write_num_duplicates(game_root, args.agents)
    GameModeController().set_game_runtime_mode()
    proc = GameLauncher().launch(wait=False)
    logger.info("游戏 PID=%d，等待连接", proc.pid)
    record = {
        "dose": dose,
        "burn": burn,
        "repeat": repeat,
        "live": live,
        "pid": proc.pid,
    }
    try:
        with GoiEnv(
            port=args.port,
            num_agents=args.agents,
            timeout=args.timeout,
            startup_settle_s=args.startup_settle,
        ) as env:
            body = _run_connected(env, args.agents, args.horizon, burn, args.seed, live)
        record.update(body)
        record["ok"] = True
    except Exception as exc:
        record["ok"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
        logger.error("会话失败: %s", record["error"])
        logger.error(traceback.format_exc())
    finally:
        _stop_process(proc)
    return record


def _fmt(values: list[float]) -> str:
    if not values:
        return "—"
    ordered = sorted(values)
    mid = ordered[len(ordered) // 2]
    return f"{mid:.3e} [{ordered[0]:.3e}, {ordered[-1]:.3e}] n={len(values)}"


def _summarize(rows: list[dict], doses: list[str]) -> list[str]:
    lines = []
    lines.append("早期残差中位数 [最小, 最大]。跨 agent 看复制体是否分开，跨 reset 看同一条探针重放是否分开。")
    for kind in ("zero", "hold"):
        lines.append(f"探针 {kind}")
        for dose in doses:
            cross = _pick(rows, dose, ("rounds", 0, kind, "cross", "s5"))
            serial = _pick(rows, dose, ("serial", kind, "serial", "s5"))
            reset_cross = _pick(rows, dose, ("rounds", 0, kind, "reset_cross"))
            lines.append(
                f"  {dose:>6}  reset散布 {_fmt(reset_cross)}  "
                f"跨agent s5 {_fmt(cross)}  跨reset s5 {_fmt(serial)}"
            )
    lines.append("可见位置是否拉回：两次 reset 的 player/tip 位置差")
    for dose in doses:
        lines.append(f"  {dose:>6}  {_fmt(_pick(rows, dose, ('pose_restore',)))}")
    hold0 = _pick(rows, "B0", ("rounds", 0, "hold", "cross", "s5"))
    hold_far = None
    far_name = None
    for dose in reversed(doses):
        if dose.startswith("B") and dose != "B0":
            hold_far = _pick(rows, dose, ("rounds", 0, "hold", "cross", "s5"))
            far_name = dose
            break
    lines.append(_verdict(hold0, hold_far, far_name, rows, doses))
    return lines


def _pick(rows: list[dict], dose: str, path: tuple) -> list[float]:
    found = []
    for row in rows:
        if not row.get("ok") or row["dose"] != dose:
            continue
        node = row
        ok = True
        for key in path:
            if isinstance(key, int):
                if not isinstance(node, list) or key >= len(node):
                    ok = False
                    break
                node = node[key]
            else:
                if not isinstance(node, dict) or key not in node or node[key] is None:
                    ok = False
                    break
                node = node[key]
        if ok and isinstance(node, (int, float)):
            found.append(float(node))
    return found


def _separated(low: list[float], high: list[float]) -> bool:
    if not low or not high:
        return False
    return min(high) > max(low)


def _verdict(hold0, hold_far, far_name, rows, doses) -> str:
    if not hold0 or not hold_far:
        return "结论：有效会话不足，不能判断。"
    if _separated(hold0, hold_far):
        text = (
            f"结论：hold 探针第 5 步的跨 agent 残差，{far_name} 全部高于 B0。"
            "同一次可见复位之后，长搅动留下了更大的早期种子。"
        )
    elif _separated(hold_far, hold0):
        text = f"结论：hold 探针第 5 步的跨 agent 残差，B0 全部高于 {far_name}。与“搅动后更不稳”相反。"
    else:
        text = (
            f"结论：hold 探针第 5 步的跨 agent 残差，B0 与 {far_name} 的区间重叠。"
            "这批重复里看不出搅动把早期种子抬高。"
        )
    second = _pick(rows, far_name, ("serial", "hold", "serial", "s5")) if far_name else []
    cold_serial = _pick(rows, "B0", ("serial", "hold", "serial", "s5"))
    if second and cold_serial and _separated(cold_serial, second):
        text += " 第二次 reset 的重放差也全部高于 B0，重新初始化没有回到冷启动。"
    elif second and cold_serial and not _separated(cold_serial, second):
        text += " 第二次 reset 的重放差与 B0 重叠，跨 reset 这一项没有分开。"
    live = _pick(rows, "live", ("rounds", 0, "hold", "cross", "s5"))
    if live and hold0 and _separated(live, hold0):
        text += " 不 reset 的载入探针低于立刻 reset 的 B0。"
    return text


def _strip_curves(record: dict) -> dict:
    """结果文件只留采样点和首次越阈，曲线已在统计时用过。"""
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="冷启动 vs 搅动后 reset 的求解器残留")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--startup-settle", type=float, default=8.0)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--doses", default="0,200,1000")
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--skip-live", action="store_true")
    parser.add_argument(
        "--out",
        default=str(_REPO_ROOT / "runs" / "solver_residue_20261002" / "result.json"),
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    doses = [int(part) for part in args.doses.split(",") if part.strip() != ""]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    game_root = Path(TrainConfig().game_root)
    config_path = game_root / "GoiData" / "runtime_config.json"
    original = config_path.read_text(encoding="utf-8") if config_path.exists() else None
    _kill_game()
    time.sleep(1.0)
    rows: list[dict] = []
    jobs: list[tuple[str, int, bool]] = []
    if not args.skip_live:
        jobs.append(("live", 0, True))
    for burn in doses:
        jobs.append((f"B{burn}", burn, False))
    try:
        for repeat in range(args.repeats):
            for dose, burn, live in jobs:
                # live 读完姿态就探针，不再次 reset 之前不能先搅动
                row = _session(args, dose, 0 if live else burn, repeat, live)
                rows.append(_strip_curves(row))
                _save(out, args, rows)
    finally:
        if original is None:
            if config_path.exists():
                config_path.unlink()
        else:
            config_path.write_text(original, encoding="utf-8")
        _kill_game()
    dose_names = [name for name, _, _ in jobs]
    for line in _summarize(rows, dose_names):
        logger.info("%s", line)
    _save(out, args, rows)
    logger.info("结果已写入 %s", out)
    return 0 if any(row.get("ok") for row in rows) else 1


def _save(path: Path, args, rows: list[dict]) -> None:
    payload = {
        "agents": args.agents,
        "horizon": args.horizon,
        "repeats": args.repeats,
        "doses": args.doses,
        "seed": args.seed,
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
