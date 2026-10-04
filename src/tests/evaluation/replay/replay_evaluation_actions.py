"""Select an archived evaluation episode and replay its exact executed actions."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from training.config import TrainConfig
from training.reference_corridor import ReferenceCorridor
from training.rollout import RolloutWorker


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def select_episode(campaign):
    """Rank actual archived episodes, without success/dwell or height criteria."""
    rows = []
    manifest = read(campaign / "manifest.json")
    for arm, seed in manifest["order"]:
        run = campaign.parent / f"{campaign.name}_{arm}_s{seed}"
        for path in sorted(run.glob("checkpoints/eval_*/*_agent*.json")):
            data = read(path)
            if not data["trace"]:
                continue
            metrics, last = data["metrics"], data["trace"][-1]
            rows.append({"path": str(path.resolve()), "arm": arm, "seed": seed,
                         "mode": metrics["mode"], "trial": metrics["trial"], "agent": metrics["agent"],
                         "credited_progress_m": metrics["credited_progress_m"],
                         "terminal_projection_m": last["progress_m"],
                         "outside_fraction": metrics["outside_fraction"], "steps": metrics["steps"]})
    if not rows:
        raise ValueError("No complete archived evaluation episodes found")
    rows.sort(key=lambda r: (-r["credited_progress_m"], -r["terminal_projection_m"], r["steps"], r["path"]))
    return rows


def load_batch(directory, mode, trial, agents):
    """Preserve original agent IDs and zero actions after each archived termination."""
    episodes = [read(directory / f"{mode}_{trial:03d}_agent{i}.json") for i in range(agents)]
    length = max(len(e["trace"]) for e in episodes)
    actions = np.zeros((length, agents, 2), dtype=np.float32)
    for i, e in enumerate(episodes):
        trace = e["trace"]
        sequence = np.asarray([t["action"] for t in trace], dtype=np.float32)
        states = np.asarray(e["states"], dtype=np.float32)
        if (e["metrics"]["agent"] != i or e["metrics"]["trial"] != trial or e["metrics"]["mode"] != mode
                or sequence.shape != (len(trace), 2) or states.shape != (len(trace)+1, 33)
                or not np.isfinite(sequence).all() or not np.isfinite(states).all()
                or np.any(np.abs(sequence) > 100.001)
                or [t["step"] for t in trace] != list(range(1, len(trace)+1))):
            raise ValueError(f"Invalid archived episode: {mode}/{trial}/{i}")
        actions[:len(sequence), i] = sequence
    return actions, episodes


def batches_through(mode, trial, rounds):
    order = [(m, t) for m in ("deterministic", "exploration") for t in range(rounds)]
    return order[:order.index((mode, trial))+1]


def route_metrics(states, cfg):
    route = ReferenceCorridor.from_file(cfg.reference_trajectory, cfg)
    from training.reference_anchor import scoring_context
    schema, meta = scoring_context(states[0], cfg, None, route)
    tracker = route.tracker_from_state(states[0], schema, meta)
    start, segment = tracker.progress, tracker.segment_index
    progress, credit, inside = [0.], [0.], [tracker.previous_inside]
    for state in states[1:]:
        result = route.step_state(tracker, state, schema, meta)
        # Local comparison is defined only within this reference segment.
        if tracker.segment_index != segment:
            raise ValueError("Cross-segment replay requires an explicit ordered-segment metric")
        progress.append(tracker.progress-start)
        credit.append(tracker.total_progress)
        inside.append(result.inside and not result.gap)
    return {"credited_progress_m": float(credit[-1]), "terminal_projection_m": float(progress[-1]),
            "maximum_projection_m": float(max(progress)),
            "outside_fraction": float(1-np.mean(inside[1:])),
            "progress": progress, "credited": credit}


def render_comparison(output, cfg, original, replay, dt):
    """Saved-state animation, explicitly separate from native game rendering."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from tests.control_interaction.test_l7_surface_airdrop import load_colliders, extract_polygons
    environment, _ = load_colliders(Path(cfg.game_root))
    polygons = extract_polygons(environment)
    route = ReferenceCorridor.from_file(cfg.reference_trajectory, cfg)
    n = min(len(original), len(replay))
    points = np.concatenate([s[:n, [0, 1]] for s in (original, replay)] +
                            [s[:n, [23, 24]] for s in (original, replay)])
    low, high = points.min(axis=0)-2, points.max(axis=0)+2
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), layout="constrained")
    colors = ("#236b9d", "#be5624")
    artists = []
    for ax, label, data, color in zip(axes, ("Archived original", "Native action replay"), (original, replay), colors):
        for poly in polygons:
            if np.all(poly.max(axis=0) >= low) and np.all(poly.min(axis=0) <= high):
                ax.fill(poly[:, 0], poly[:, 1], color="0.87", edgecolor="0.65", linewidth=.4)
        for segment in route.segments:
            ax.plot(segment.points[:, 0], segment.points[:, 1], color="#479267", linewidth=1, alpha=.7)
        ax.plot(data[:, 0], data[:, 1], color=color, alpha=.2, linewidth=1)
        trail, = ax.plot([], [], color=color, linewidth=1.4)
        body, = ax.plot([], [], "o", color=color, markersize=10)
        hammer, = ax.plot([], [], "o-", color=color, linewidth=2, markersize=4)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel="World x (m)", ylabel="World y (m)", title=label)
        ax.set_aspect("equal")
        artists.append((trail, body, hammer))
    title = fig.suptitle("Recorded body / hub / hammer-tip anchors")
    def draw(index):
        for data, (trail, body, hammer) in zip((original, replay), artists):
            row = data[index]
            trail.set_data(data[:index+1, 0], data[:index+1, 1])
            body.set_data([row[0]], [row[1]])
            hammer.set_data([row[5], row[23]], [row[6], row[24]])
        title.set_text(f"Recorded anchors | physical time {index*dt:.2f} s | step {index}")
        return [title, *(a for group in artists for a in group)]
    frames = sorted(set([*range(0, n, 6), n-1]))
    movie = FuncAnimation(fig, draw, frames=frames, interval=50, blit=False)
    movie.save(output / "comparison.gif", writer=PillowWriter(fps=20), dpi=90)
    draw(n-1)
    fig.savefig(output / "comparison.png", dpi=140)
    plt.close(fig)


def replay_elite(args, output):
    """Broadcast the farthest complete elite's executed actions to native players."""
    from env import make_goi_env
    from start.multi_player import launch_players
    from training.dataset import Trajectory
    from training.elite_archive import EliteTrajectoryArchive
    from training.reference_corridor import trajectory_endpoint
    from training.rollout import _warmup_and_snapshot
    from training.run_logging import load_elites

    manifest = read(args.elite_manifest)
    pool = Path(manifest["source_pool"])
    with np.load(pool) as archive:
        metadata = json.loads(str(archive["archive_metadata_json"]))
    if (metadata.get("schema") != "full_episode_elites_v1"
            or metadata.get("origin") != "episode_reset" or metadata.get("truncation") != "none"):
        raise ValueError("Expected a complete reset-to-end elite archive")
    trajectories = [Trajectory(**row) for row in load_elites(pool)]
    if not trajectories or any(not EliteTrajectoryArchive._complete(t) for t in trajectories):
        raise ValueError("Incomplete elite trajectories")
    candidates = [(i, trajectory_endpoint(t)) for i, t in enumerate(trajectories)]
    candidates = [(i, p) for i, p in candidates if p.valid and p.t_cap is not None]
    index, endpoint = min(candidates, key=lambda item: (-item[1].distance, item[1].t_cap, item[0]))
    trajectory = trajectories[index]
    commands = np.asarray(trajectory.actions, dtype=np.float32)
    original = np.asarray(trajectory.raw_states, dtype=np.float32)
    if not np.isfinite(commands).all() or np.any(np.abs(commands) > 100.001):
        raise ValueError("Invalid archived actions")
    cfg = TrainConfig(**manifest["config"])
    project = read(ROOT / "src/config/project.json")
    cfg.game_root = str(Path(project["game"]["executable_path"]).parent)
    cfg.port = project["tcp_port"]
    cfg.num_agents = args.num_agents
    if cfg.random_deploy:
        raise ValueError("Elite replay requires the canonical spawn")
    route = ReferenceCorridor.from_file(cfg.reference_trajectory, cfg)
    if route.metadata["sha256"] != manifest["reference"]["sha256"]:
        raise ValueError("Archived reference file changed")
    selected = {"pool": str(pool), "pool_sha256": hashlib.sha256(pool.read_bytes()).hexdigest(),
                "pool_index": index, "source_iteration": int(trajectory.iteration),
                "distance_m": endpoint.distance, "cap_step": endpoint.t_cap,
                "steps": len(commands), "players": cfg.num_agents,
                "criterion": "maximum confirmed route distance; earliest cap breaks ties",
                "broadcast": "identical archived actions to every player"}
    write(output / "selected.json", selected)
    if args.select_only:
        print(json.dumps(selected), flush=True)
        return
    if any(p.info["name"] == "GettingOverIt.exe" for p in psutil.process_iter(["name"])):
        raise RuntimeError("An existing game is running; refusing to take it over")
    runtime = Path(cfg.game_root) / "GoiData/runtime_config.json"
    before = runtime.read_bytes() if runtime.exists() else None
    if before is not None:
        (output / "runtime_original.json").write_bytes(before)
    worker = RolloutWorker(cfg)
    replay = []
    status = {"status": "starting", "started_at": datetime.now().astimezone().isoformat(),
              "selected": selected, "config": asdict(cfg), "fps": args.fps,
              "desktop": "current" if args.current_desktop else "GoiRuntime"}
    write(output / "result.json", status)
    try:
        options = {"desktop": None} if args.current_desktop else {}
        worker._game_process = launch_players(cfg.num_agents, game_root=cfg.game_root,
                                               port=cfg.port, **options)
        worker.env = make_goi_env(cfg)
        worker.env.connect()
        worker.env.reset()
        _warmup_and_snapshot(worker.env, cfg.num_agents, cfg.warmup_steps)
        worker.env._sock.settimeout(15.)
        # Preserve training's wireframe renderer and free camera for live viewing.
        worker.env.set_camera_free(True)
        worker.env.toggle_collider_visual(True)
        # Diagnostics are opt-in: enable capture before the canonical reset.
        worker.env.observation_diagnostics()
        obs = worker._reset_and_deploy()
        packet = worker.env.observation_diagnostics()
        dt = packet["fixedDeltaTime"] * packet["stepFrames"]
        if not packet["aligned"] or not np.isclose(dt, cfg.feature_step_seconds, atol=1e-6):
            raise ValueError(f"Game physical step does not match archive: {packet}")
        physical = [i for i in range(original.shape[1]) if i != 28]
        if obs.shape != (cfg.num_agents, original.shape[1]):
            raise ValueError("Replay state schema differs from archive")
        initial_error = float(np.max(np.abs(obs[:, physical] - original[0, physical])))
        if initial_error > 1e-5:
            raise ValueError(f"Canonical spawn mismatch: {initial_error}")
        status.update(status="ready", game_pid=worker._game_process.pid,
                      initial_physical_max_error=initial_error, physical_step_seconds=dt)
        write(output / "result.json", status)
        if args.start_signal:
            deadline = time.monotonic() + 300
            while not args.start_signal.exists():
                if worker._game_process.poll() is not None:
                    raise RuntimeError("Game exited before replay")
                if time.monotonic() > deadline:
                    raise TimeoutError("Replay start signal not received within 300 seconds")
                time.sleep(.1)
        logging.info("Native elite replay starts in 3 seconds: %s", selected)
        time.sleep(3)
        status.update(status="replaying")
        write(output / "result.json", status)
        replay.append(obs.copy())
        for command in commands:
            started = time.perf_counter()
            obs, _ = worker.env.step(np.tile(command, (cfg.num_agents, 1)))
            worker.env.observation_diagnostics()
            replay.append(obs.copy())
            time.sleep(max(0., 1 / args.fps - (time.perf_counter() - started)))
        replay_array = np.asarray(replay, dtype=np.float32)
        np.save(output / "replay_states.npy", replay_array)
        np.save(output / "replayed_actions.npy", commands)
        errors = np.abs(replay_array[:, :, physical] - original[:, None, physical])
        body_errors = np.linalg.norm(replay_array[:, :, :2] - original[:, None, :2], axis=-1)
        status.update(status="completed", actions_replayed=len(commands),
                      maximum_physical_state_error=float(errors.max()),
                      maximum_body_error_m=float(body_errors.max()),
                      maximum_player_spread=float(np.max(np.abs(
                          replay_array[:, :, physical] - replay_array[:, :1, physical]))),
                      game_left_open=bool(args.keep_game_open))
    except BaseException as exc:
        status.update(status="failed", error=repr(exc))
        if replay:
            np.save(output / "partial_replay_states.npy", np.asarray(replay))
        raise
    finally:
        if status["status"] == "completed" and args.keep_game_open:
            worker.env.close()
            worker.env = None
            worker._game_process = None
        else:
            worker.teardown()
        if before is not None:
            runtime.write_bytes(before)
        elif runtime.exists():
            runtime.unlink()
        status["finished_at"] = datetime.now().astimezone().isoformat()
        write(output / "result.json", status)
    print(json.dumps({k: v for k, v in status.items() if k != "config"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=ROOT/"runs/corridor_formal_20260922")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--fps", type=float, default=30., help="Maximum wall-clock STEP rate for selected episode")
    parser.add_argument("--elite-manifest", type=Path, help="Saved elite export manifest; replay the farthest full-pool elite")
    parser.add_argument("--num-agents", type=int, default=8, help="Player count for elite action broadcast")
    parser.add_argument("--current-desktop", action="store_true", help="Launch elite replay on the visible desktop")
    parser.add_argument("--keep-game-open", action="store_true", help="Leave successful elite replay paused at its last state")
    parser.add_argument("--start-signal", type=Path, help="Wait up to 300 seconds for this file before elite replay")
    args = parser.parse_args()
    if not 1 <= args.fps <= 120:
        parser.error("fps must be in [1, 120]")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.elite_manifest:
        replay_elite(args, output)
        return
    ranking = select_episode(args.campaign.resolve())
    selected = ranking[0]
    write(output/"ranking.json", {"criterion": "credited corridor progress descending; terminal projection breaks ties; no dwell/height test", "episodes": ranking})
    write(output/"selected.json", selected)
    if args.select_only:
        print(json.dumps(selected, ensure_ascii=False)); return
    if any(p.info["name"] == "GettingOverIt.exe" for p in psutil.process_iter(["name"])):
        raise RuntimeError("An existing game is running; refusing to take it over")
    directory = Path(selected["path"]).parent
    manifest = read(directory/"manifest.json")
    cfg = TrainConfig(**manifest["config"])
    project = read(ROOT/"src/config/project.json")
    cfg.game_root = str(Path(project["game"]["executable_path"]).parent)
    cfg.port = project["tcp_port"]
    if cfg.random_deploy:
        raise ValueError("Replay currently requires archived fixed deployment")
    route = ReferenceCorridor.from_file(cfg.reference_trajectory, cfg)
    if route.metadata["sha256"] != manifest["reference"]["sha256"]:
        raise ValueError("Archived reference file changed")
    offsets = np.asarray(manifest["offsets"], np.float32)
    if offsets.shape[1:] != (cfg.num_agents, 2):
        raise ValueError("Offset shape does not match original player count")
    order = batches_through(selected["mode"], selected["trial"], len(offsets))
    batches = [(mode, trial, *load_batch(directory, mode, trial, cfg.num_agents)) for mode, trial in order]
    original = np.asarray(read(selected["path"])["states"], np.float32)
    write(output/"original_episode.json", read(selected["path"]))
    runtime = Path(cfg.game_root)/"GoiData/runtime_config.json"
    before = runtime.read_bytes() if runtime.exists() else None
    if before is not None:
        (output/"runtime_original.json").write_bytes(before)
    worker = RolloutWorker(cfg)
    replay = []
    status = {"status": "running", "started_at": datetime.now().astimezone().isoformat(),
              "selected": selected, "prefix_batches": order[:-1], "config": asdict(cfg),
              "selection_scope": f"{len(ranking)} archived evaluation episodes; training raw states were not archived",
              "source_sha256": hashlib.sha256(Path(selected["path"]).read_bytes()).hexdigest(),
              "note": "Exact saved actions; reset/warmup/jitter and preceding batches restored, not a full historical physics-engine snapshot."}
    write(output/"result.json", status)
    try:
        worker.launch_game()
        worker.env._sock.settimeout(15.)
        # The selected best episode is agent 0, so the native camera can follow it.
        if selected["agent"] == 0:
            worker.env.set_camera_free(False)
        worker.env.toggle_collider_visual(False)
        for mode, trial, actions, episodes in batches:
            target = (mode, trial) == (selected["mode"], selected["trial"])
            logging.info("%s %s trial %d: %d physical steps", "REPLAY" if target else "PREFIX", mode, trial, len(actions))
            worker.env.observation_diagnostics()
            obs = worker._reset_and_deploy()
            positions = obs[:, :2].copy()+offsets[trial]
            for i, position in enumerate(positions):
                if np.any(offsets[trial, i]):
                    obs = worker.env.teleport(float(position[0]), float(position[1]), agent_index=i)
            packet = worker.env.observation_diagnostics()
            dt = packet["fixedDeltaTime"]*packet["stepFrames"]
            if not packet["aligned"] or not np.isclose(dt, cfg.feature_step_seconds, atol=1e-6):
                raise ValueError("Game physical step does not match archive")
            if target:
                replay = [obs[selected["agent"]].copy()]
                np.save(output/"replayed_batch_actions.npy", actions)
                logging.info("Selected action replay begins in 3 seconds")
                time.sleep(3)
            for index, command in enumerate(actions):
                started = time.perf_counter()
                obs, _ = worker.env.step(command)
                worker.env.observation_diagnostics()
                if target:
                    replay.append(obs[selected["agent"]].copy())
                    time.sleep(max(0., 1/args.fps-(time.perf_counter()-started)))
            if target:
                time.sleep(2)
        replay = np.asarray(replay, np.float32)[:len(original)]
        np.save(output/"replay_states.npy", replay)
        errors = np.linalg.norm(original[:, :2]-replay[:, :2], axis=1)
        tip_errors = np.linalg.norm(original[:, 23:25]-replay[:, 23:25], axis=1)
        over = np.flatnonzero(errors>.1)
        status.update(status="completed", original=route_metrics(original, cfg), replay=route_metrics(replay, cfg),
                      player_position_rmse_m=float(np.sqrt(np.mean(errors**2))),
                      initial_player_error_m=float(errors[0]), final_player_error_m=float(errors[-1]),
                      maximum_player_error_m=float(errors.max()), maximum_tip_error_m=float(tip_errors.max()),
                      first_step_player_error_over_10cm=int(over[0]) if len(over) else None,
                      actions_replayed=len(replay)-1, physical_step_seconds=dt)
        print(json.dumps({k:v for k,v in status.items() if k not in ("config", "original", "replay")},ensure_ascii=False),flush=True)
    except BaseException as exc:
        status.update(status="failed", error=repr(exc))
        if len(replay):
            np.save(output/"partial_replay_states.npy", np.asarray(replay))
        raise
    finally:
        worker.teardown()
        if before is not None:
            runtime.write_bytes(before)
        elif runtime.exists():
            runtime.unlink()
        status.update(finished_at=datetime.now().astimezone().isoformat(),
                      runtime_restored=runtime.read_bytes()==before if before is not None else not runtime.exists())
        write(output/"result.json", status)
    render_comparison(output, cfg, original, replay, dt)


if __name__ == "__main__":
    main()
