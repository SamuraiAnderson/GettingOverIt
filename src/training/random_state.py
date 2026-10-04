"""Checkpoint the training random streams at complete rollout boundaries."""
import copy
import random

import numpy as np
import torch


def seed_training(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def capture_random_state(deployment_rng=None):
    return {
        "version": 1,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "deployment": (copy.deepcopy(deployment_rng.bit_generator.state)
                       if deployment_rng is not None else None),
    }


def restore_random_state(state, deployment_rng=None):
    if state.get("version") != 1:
        raise ValueError("Unsupported training RNG checkpoint")
    cuda = state["torch_cuda"]
    if len(cuda) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError("CUDA device count changed; use --reset-rng for a new random stream")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if cuda:
        torch.cuda.set_rng_state_all([s.cpu() for s in cuda])
    if deployment_rng is not None and state.get("deployment") is not None:
        deployment_rng.bit_generator.state = copy.deepcopy(state["deployment"])
