"""PPO decisions over physical frame sequences, including variable-duration options."""
import json
from types import SimpleNamespace
import numpy as np
import torch

from .dataset import left_pad_sequence

_SPARSE_NAMES = ("contour_tokens", "contour_mask", "relation_tokens", "relation_mask")


def duration_gae(rewards, values, dones, durations, bootstrap, gamma, lam):
    """Rewards already contain sum_j gamma**j * r_j; lambda decays on the environment clock."""
    advantage = np.zeros(len(values), dtype=np.float32)
    following_value, following_advantage = float(bootstrap), 0.
    for i in reversed(range(len(values))):
        alive, k = 1 - float(dones[i]), int(durations[i])
        delta = rewards[i] + gamma**k * following_value * alive - values[i]
        following_advantage = delta + (gamma * lam)**k * alive * following_advantage
        advantage[i] = following_advantage
        following_value = values[i]
    return advantage, advantage + np.asarray(values, np.float32)


def evaluated_batches(model, buffer, batch_size, normalize_advantages=True):
    if isinstance(buffer, DecisionRolloutBuffer):
        yield from buffer.evaluated_batches(model, batch_size, normalize_advantages)
    else:
        for batch in buffer.get_batches(batch_size, normalize_advantages):
            yield (batch, *model.evaluate_actions(batch.dynamics, batch.patches, batch.act_history,
                batch.actions, valid_mask=batch.valid_mask, noise_prev=batch.noise_prev,
                noise_phi=buffer.policy_context["noise_phi"], raw_actions=batch.raw_actions,
                **batch.sparse_kwargs()))


class DecisionRolloutBuffer:
    """One frame copy per agent, decision indices per option, true episode reset boundaries."""
    def __init__(self, policy_context=None, device="cpu"):
        self.policy_context = policy_context or {}
        self.device = torch.device(device)
        self.episodes = []

    @property
    def size(self):
        return sum(len(e["decision_steps"]) for e in self.episodes)

    @property
    def environment_steps(self):
        return sum(int(e["durations"].sum()) for e in self.episodes)

    @property
    def _rewards(self):
        return np.concatenate([e["step_rewards"] for e in self.episodes]) if self.episodes else np.zeros(0)

    def add_episode(self, frames, decisions, bootstrap, gamma, lam):
        if not decisions:
            return
        episode = {name: np.asarray(value, dtype=np.float32) for name, value in frames.items()}
        for name in ("decision_steps", "durations", "actions", "raw_actions", "noise_prev",
                     "log_probs", "values", "rewards", "dones"):
            episode[name] = np.asarray([d[name] for d in decisions], dtype=(
                np.int64 if name in ("decision_steps", "durations") else np.float32))
        if not all(np.isfinite(x).all() for x in episode.values()):
            raise ValueError("Non-finite decision episode")
        starts, durations = episode["decision_steps"], episode["durations"]
        if (starts[0] != 0 or np.any(durations <= 0)
                or not np.array_equal(starts[1:], (starts + durations)[:-1])
                or starts[-1] + durations[-1] != len(episode["dynamics"]) - 1):
            raise ValueError("Decisions must partition consecutive physical steps from reset")
        episode["advantages"], episode["returns"] = duration_gae(
            episode["rewards"], episode["values"], episode["dones"], durations, bootstrap, gamma, lam)
        self.episodes.append(episode)

    def patch_storage_bytes(self):
        return sum(e["patches"].nbytes for e in self.episodes)

    def _tensor(self, array):
        return torch.as_tensor(np.asarray(array), dtype=torch.float32, device=self.device)

    def _batch(self, rows, advantages):
        def data(name):
            return self._tensor(np.stack([self.episodes[e][name][i] for e, i in rows]))
        return SimpleNamespace(row_ids=rows, actions=data("actions"), raw_actions=data("raw_actions"),
            noise_prev=data("noise_prev"), old_log_probs=data("log_probs"), old_values=data("values"),
            returns=data("returns"), advantages=self._tensor([advantages[e][i] for e, i in rows]))

    def _inputs(self, episode, start, end):
        return tuple(self._tensor(episode[name][start:end])[None]
                     for name in ("dynamics", "patches", "act_history"))

    def _sparse_kwargs(self, episode, start=None, end=None, *, windows=None):
        if windows:
            return dict(zip(_SPARSE_NAMES, (
                self._tensor(np.stack([w[k] for w in windows])) for k in range(4)
            )))
        if episode is None or "contour_tokens" not in episode:
            return {}
        return {name: self._tensor(episode[name][start:end])[None] for name in _SPARSE_NAMES}

    def evaluated_batches(self, model, batch_size, normalize_advantages=True):
        self.device = next(model.parameters()).device
        phi = self.policy_context["noise_phi"]
        advantages = [e["advantages"].copy() for e in self.episodes]
        if normalize_advantages and self.size > 1:
            all_adv = np.concatenate(advantages)
            advantages = [(a-all_adv.mean()) / (all_adv.std()+1e-8) for a in advantages]
        if model.config.policy_backbone == "gru":
            size = min(batch_size, model.config.recurrent_seq_len)
            chunks = [(e, start, min(start + size, len(episode["decision_steps"])))
                      for e, episode in enumerate(self.episodes)
                      for start in range(0, len(episode["decision_steps"]), size)]
            for chunk in np.random.permutation(len(chunks)):
                e, first, last = chunks[chunk]
                episode = self.episodes[e]
                steps = episode["decision_steps"][first:last]
                start, end = int(steps[0]), int(steps[-1]) + 1
                hidden = None
                # Recompute the entire prefix with CURRENT weights. Do not reuse stale
                # behavior hidden states or silently zero-initialize arbitrary chunks.
                if start:
                    with torch.no_grad():
                        _, hidden = model.recurrent_features(
                            *self._inputs(episode, 0, start),
                            **self._sparse_kwargs(episode, 0, start),
                        )
                features, _ = model.recurrent_features(
                    *self._inputs(episode, start, end), hidden=hidden,
                    **self._sparse_kwargs(episode, start, end),
                )
                features = features[0, torch.as_tensor(steps-start, device=self.device)]
                batch = self._batch([(e, i) for i in range(first, last)], advantages)
                yield (batch, *model.evaluate_features(features, batch.raw_actions, batch.noise_prev, phi))
        else:
            rows = [(e, i) for e, episode in enumerate(self.episodes) for i in range(len(episode["decision_steps"]))]
            order = np.random.permutation(len(rows))
            ctx = model.config.context_len
            for begin in range(0, len(rows), batch_size):
                selected = [rows[i] for i in order[begin:begin+batch_size]]
                windows, masks, sparse_windows = [], [], []
                for e, i in selected:
                    episode = self.episodes[e]
                    step = int(episode["decision_steps"][i])
                    sl = slice(max(0, step-ctx+1), step+1)
                    frames = [left_pad_sequence(episode[name][sl], ctx)
                              for name in ("dynamics", "patches", "act_history")]
                    windows.append([value for value, mask in frames])
                    masks.append(frames[0][1])
                    if "contour_tokens" in episode:
                        sparse_windows.append([
                            left_pad_sequence(episode[name][sl], ctx)[0] for name in _SPARSE_NAMES
                        ])
                dynamics, patches, actions = (self._tensor(np.stack([w[k] for w in windows])) for k in range(3))
                batch = self._batch(selected, advantages)
                yield (batch, *model.evaluate_actions(dynamics, patches, actions, batch.actions,
                    valid_mask=self._tensor(masks), noise_prev=batch.noise_prev, noise_phi=phi,
                    raw_actions=batch.raw_actions,
                    **self._sparse_kwargs(episode, windows=sparse_windows or None)))

    def save(self, path):
        if not self.size:
            raise ValueError("Cannot archive an empty decision buffer")
        arrays = {f"episode_{i}_{name}": array for i, ep in enumerate(self.episodes) for name, array in ep.items()}
        metadata = {"version": 3, "kind": "physical_sequences", "policy_context": self.policy_context,
                    "episodes": len(self.episodes), "fields": list(self.episodes[0])}
        np.savez_compressed(path, **arrays, metadata=json.dumps(metadata))

    @classmethod
    def load(cls, path, device="cpu"):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if meta["version"] != 3 or meta.get("kind") != "physical_sequences":
                raise ValueError("Unsupported decision archive")
            result = cls(meta["policy_context"], device)
            result.episodes = [{name: data[f"episode_{i}_{name}"] for name in meta["fields"]}
                               for i in range(meta["episodes"])]
        return result
