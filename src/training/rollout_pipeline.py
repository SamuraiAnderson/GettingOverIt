"""Experimental one-request actor process for fixed-corridor PPO.

The actor owns the game/TCP connection and a frozen model. Only immutable files
cross the process boundary; the parent owns the learner and its optimizer.
This is bounded-lag PPO, not a claim of strict on-policy equivalence.
"""
from __future__ import annotations

from dataclasses import asdict
import multiprocessing as mp
from pathlib import Path
import time
import traceback


def check_version(behavior_version, learner_version, max_lag=1):
    lag = learner_version-behavior_version
    if not 0 <= lag <= max_lag:
        raise ValueError(f"Policy version lag {lag} outside [0, {max_lag}]")
    return lag


def _actor(connection, config, output):
    import numpy as np
    import torch
    from .config import TrainConfig
    from .experimental_policy import create_actor_critic
    from .policy_audit import audit_policy
    from .reward import ClimbingEfficiencyMap
    from .rollout import RolloutWorker
    from .local_evaluation import score_corridor_episode
    # Same collider reader as the existing training entrypoint.
    from tests.control_interaction.test_l7_surface_airdrop import load_colliders, extract_polygons
    torch.set_num_threads(4)
    cfg = TrainConfig(**config)
    worker = RolloutWorker(cfg)
    try:
        model = create_actor_critic(cfg).cuda().eval()
        env, player = load_colliders(Path(cfg.game_root))
        polygons = extract_polygons(env)
        terrain = ClimbingEfficiencyMap(polygons, grid_resolution=cfg.grid_resolution,
            diffusion_iterations=cfg.diffusion_iterations, diffusion_alpha=cfg.diffusion_alpha,
            cache_path=str(Path(output)/"checkpoints/terrain.npz"), height_prior_weight=cfg.height_prior_weight)
        worker.set_terrain_info(terrain.traversable, terrain._min_gx, terrain._min_gy)
        worker.set_deploy_obstacle_geometry(polygons, player)
        worker.launch_game()
        connection.send({"kind":"ready", "game_pid":worker._game_process.pid})
        while True:
            job = connection.recv()
            if job is None:
                break
            state = torch.load(job["snapshot"], map_location="cpu", weights_only=False)
            if state["version"] != job["version"]:
                raise ValueError("Snapshot version differs from requested collection version")
            model.load_state_dict(state["model"])
            model.exploration_boost = state["exploration_boost"]
            np.random.seed(job["seed"])
            torch.manual_seed(job["seed"])
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            buffer, trajectories = worker.collect_ppo(model, job["steps"], terrain,
                                                       deterministic=job.get("deterministic", False))
            torch.cuda.synchronize()
            collection_end = time.perf_counter()
            audit = audit_policy(model, buffer)  # Always replay ALL frozen inputs.
            audit_end = time.perf_counter()
            path = Path(output)/"checkpoints"/f"rollout_{job['id']:04d}.npz"
            buffer.save(path)
            scores = [score_corridor_episode(t, worker.reference_corridor, 5., 10)[0]
                      for t in trajectories]
            connection.send({"kind":"rollout", "id":job["id"], "version":job["version"],
                "snapshot":job["snapshot"], "buffer":str(path), "audit":audit,
                "physical_samples":buffer.environment_steps, "decisions":buffer.size,
                "started":started, "collection_end":collection_end, "finished":time.perf_counter(),
                "collection_seconds":collection_end-started, "audit_seconds":audit_end-collection_end,
                "profile":worker.last_profile, "scores":scores,
                "gpu_allocated_gb":torch.cuda.max_memory_allocated()/2**30,
                "gpu_reserved_gb":torch.cuda.max_memory_reserved()/2**30})
    except BaseException:
        try:
            connection.send({"kind":"error", "traceback":traceback.format_exc()})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        worker.close_game()
        connection.close()


class RolloutPipeline:
    """Single in-flight request; caller explicitly chooses serial or overlapping use."""
    def __init__(self, config, output, timeout=300):
        if (config.random_deploy or config.sil_enabled or config.explore_boost_enabled
                or not config.reference_trajectory or config.policy_backbone != "transformer"
                or config.action_segment_steps != 1 or not config.normalize_action_history):
            raise ValueError("Pipeline pilot requires fixed corridor, B1 transformer/H1, fixed exploration and no SIL")
        import psutil
        if any(p.info["name"] == "GettingOverIt.exe" for p in psutil.process_iter(["name"])):
            raise RuntimeError("An existing game is running; do not take over its TCP session")
        output = Path(output)
        (output/"checkpoints").mkdir(parents=True, exist_ok=True)
        runtime = Path(config.game_root)/"GoiData/runtime_config.json"
        self.runtime_path = runtime
        self.runtime_original = runtime.read_bytes() if runtime.exists() else None
        self.timeout, self.pending = timeout, None
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_actor, args=(child, asdict(config), str(output)))
        self.process.start()
        child.close()
        try:
            self.ready = self._receive()
            if self.ready["kind"] != "ready":
                raise RuntimeError("Actor did not announce readiness")
        except BaseException:
            self.close()
            raise

    def _receive(self):
        deadline = time.monotonic()+self.timeout
        while not self.connection.poll(.2):
            if not self.process.is_alive():
                raise RuntimeError(f"Collector exited with code {self.process.exitcode}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Collector deadline exceeded")
        result = self.connection.recv()
        if result["kind"] == "error":
            raise RuntimeError(result["traceback"])
        return result

    def submit(self, job):
        if self.pending is not None:
            raise RuntimeError("Only one collection may be in flight")
        self.connection.send(job)
        self.pending = dict(job)

    def receive(self):
        if self.pending is None:
            raise RuntimeError("No collection requested")
        result = self._receive()
        if (result.get("id"), result.get("version")) != (self.pending["id"], self.pending["version"]):
            raise RuntimeError("Collector returned the wrong request/version")
        self.pending = None
        return result

    def close(self):
        try:
            self.connection.send(None)
        except (BrokenPipeError, EOFError, OSError):
            pass
        self.process.join(timeout=5)
        if self.process.is_alive():
            # Only descendants of OUR collector, including its game, are terminated.
            import psutil
            try:
                descendants = psutil.Process(self.process.pid).children(recursive=True)
                for child in reversed(descendants):
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs(descendants, timeout=3)
                for child in alive:
                    child.kill()
            except psutil.NoSuchProcess:
                pass
            self.process.terminate()
            self.process.join(timeout=5)
        self.connection.close()
        if (self.runtime_original is not None
                and self.runtime_path.read_bytes() != self.runtime_original):
            self.runtime_path.write_bytes(self.runtime_original)
