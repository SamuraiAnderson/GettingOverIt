"""Independent PPO feature / GRU / segment candidates with strict checkpoint semantics."""
import torch
from torch import nn
from torch.distributions import Normal

from .actor_critic import ActorCritic
from .dataset import DYNAMICS_DIM
from .model import build_attn_mask
from .policy_features import experimental_enabled, feature_layout, validate_experiments


def create_actor_critic(config):
    return ExperimentalActorCritic(config) if experimental_enabled(config) else ActorCritic(config)


class ExperimentalActorCritic(ActorCritic):
    def __init__(self, config):
        validate_experiments(config)
        if config.contact_heads_enabled:
            raise ValueError("Representation experiments use a single actor head; disable contact_heads_enabled")
        super().__init__(config)
        d = config.d_model
        self.layout, self.extra_dim = feature_layout(config)
        self.encoded_frame_count = 0  # Runtime profiling only; never part of policy state.
        self.feature_projections = nn.ModuleDict({
            name: nn.Linear(group.stop - group.start, d)
            for name, group in self.layout.items() if name != "events" or not config.event_branch
        })
        if config.event_branch:
            group = self.layout["events"]
            self.event_encoder = nn.Sequential(nn.Linear(group.stop - group.start, 32), nn.Tanh())
            self.event_to_backbone = nn.Linear(32, d)
            self.event_to_heads = nn.Linear(32, d)
        if config.policy_backbone == "gru":
            del self.transformer
            del self.pos_embed
            self.gru = nn.GRU(2*d, d, batch_first=True)
        self.parameter_dim = 4 if config.action_segment_steps > 1 else config.action_dim
        if self.parameter_dim != config.action_dim:
            self.actor_mean = nn.Linear(d, self.parameter_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(self.parameter_dim))
            self._init_weights()

    def initialize_from(self, state):
        """Explicit warm start, never restore old optimizer or pretend architectures are equivalent."""
        from .model import expand_state_dict_for_dynamics_dim
        state = expand_state_dict_for_dynamics_dim(state, self.state_dict())
        target = self.state_dict()
        loaded = []
        for name, value in state.items():
            if name not in target:
                continue
            if value.shape == target[name].shape:
                target[name] = value
                loaded.append(name)
            elif name in ("actor_mean.weight", "actor_mean.bias", "actor_log_std") and value.shape[0] == 2 and self.parameter_dim == 4:
                target[name] = torch.cat([value, value], dim=0)
                loaded.append(name)
        # Added inputs initially have zero contribution when explicitly warm-starting.
        # Existing trained feature projections are retained when present.
        for name in target:
            if name not in state and name.startswith(("feature_projections.", "event_to_backbone.", "event_to_heads.")):
                target[name] = torch.zeros_like(target[name])
        self.load_state_dict(target)
        return loaded

    def _encode_frames(self, dynamics, patches, actions,
                       contour_tokens=None, contour_mask=None,
                       relation_tokens=None, relation_mask=None):
        b, t, width = dynamics.shape
        self.encoded_frame_count += b * t
        if width != DYNAMICS_DIM + self.extra_dim:
            raise ValueError(f"Expected {DYNAMICS_DIM + self.extra_dim} features, received {width}")
        base, extras = dynamics[..., :DYNAMICS_DIM], dynamics[..., DYNAMICS_DIM:]
        dynamics_embed = self.dynamics_proj(self._normalize_dynamics(base))
        spatial = self.spatial_encoder(patches.reshape(b*t, *patches.shape[2:])).reshape(b, t, -1)
        obs = self.fusion(torch.cat([dynamics_embed, spatial], dim=-1))
        obs = obs + self._sparse_residual(
            b * t, dynamics.device, dynamics_embed.dtype,
            contour_tokens, contour_mask, relation_tokens, relation_mask,
        ).reshape(b, t, -1)
        for name, projection in self.feature_projections.items():
            obs = obs + projection(extras[..., self.layout[name]])
        event = None
        if self.config.event_branch:
            event = self.event_encoder(extras[..., self.layout["events"]])
            obs = obs + self.event_to_backbone(event)
        history = actions / self.config.action_scale if self.config.normalize_action_history else actions
        return obs, self.action_proj(history), event

    def recurrent_features(self, dynamics, patches, actions, hidden=None, valid_mask=None, reset_mask=None,
                           contour_tokens=None, contour_mask=None, relation_tokens=None, relation_mask=None):
        """One frame at collection; consecutive frames at training. Masked pads do not advance h."""
        # cuDNN needs its training kernel to retain backward intermediates. This
        # single-layer GRU has dropout=0, so using that kernel in BOTH collection
        # and updates preserves deterministic policy semantics while enabling gradients.
        self.gru.train(True)
        obs, action, events = self._encode_frames(
            dynamics, patches, actions, contour_tokens, contour_mask, relation_tokens, relation_mask,
        )
        inputs = torch.cat([obs, action], dim=-1)
        if hidden is None:
            hidden = inputs.new_zeros(1, inputs.shape[0], self.config.d_model)
        if valid_mask is None and reset_mask is None:
            output, hidden = self.gru(inputs, hidden)
        else:
            outputs = []
            for step in range(inputs.shape[1]):
                if reset_mask is not None:
                    hidden = hidden * (1 - reset_mask[:, step]).view(1, -1, 1)
                out, updated = self.gru(inputs[:, step:step+1], hidden)
                if valid_mask is not None:
                    active = valid_mask[:, step].bool().view(1, -1, 1)
                    updated = torch.where(active, updated, hidden)
                hidden = updated
                outputs.append(hidden[0].unsqueeze(1))
            output = torch.cat(outputs, dim=1)
        if events is not None:
            output = output + self.event_to_heads(events)
        return output, hidden

    def _backbone(self, dynamics, patches, actions, valid_mask=None,
                  contour_tokens=None, contour_mask=None,
                  relation_tokens=None, relation_mask=None):
        if self.config.policy_backbone == "gru":
            output, _ = self.recurrent_features(
                dynamics, patches, actions, valid_mask=valid_mask,
                contour_tokens=contour_tokens, contour_mask=contour_mask,
                relation_tokens=relation_tokens, relation_mask=relation_mask,
            )
            return output[:, -1]
        obs, action, event = self._encode_frames(
            dynamics, patches, actions, contour_tokens, contour_mask, relation_tokens, relation_mask,
        )
        b, t, d = obs.shape
        seq = obs.new_zeros(b, 2*t, d)
        seq[:, 0::2], seq[:, 1::2] = obs, action
        seq = seq + self.pos_embed(torch.arange(2*t, device=seq.device)).unsqueeze(0)
        out = self.transformer(seq, mask=build_attn_mask(2*t, valid_mask, self.config.nhead, seq.device))[:, -1]
        if event is not None:
            out = out + self.event_to_heads(event[:, -1])
        return out

    def distribution_from_features(self, features):
        mean = self.actor_mean(features)
        std = (self.actor_log_std + self.exploration_boost).clamp(
            self.config.ppo_log_std_min, self.config.ppo_log_std_max).exp().expand_as(mean)
        return Normal(mean, std), self.critic_head(features).squeeze(-1)

    def sample_features(self, features, noise_prev, noise_phi, deterministic=False, noise_scale=None):
        from .actor_critic import ActionValue
        distribution, value = self.distribution_from_features(features)
        mean, std = distribution.mean, distribution.stddev
        std_t = self._scaled_std(std, noise_scale)
        cm, cs = self._ou_conditional(mean, std_t, noise_prev, noise_phi)
        conditional = Normal(cm, cs)
        raw = mean if deterministic else conditional.sample()
        noise = ((raw - mean) / std_t.clamp(min=1e-8)).detach()
        lp = self._log_prob_squash(conditional.log_prob(raw).sum(-1), raw)
        return ActionValue(self._squash_action(raw), lp, value, noise, raw.detach())

    def evaluate_features(self, features, raw, noise_prev, noise_phi, noise_scale=None):
        distribution, values = self.distribution_from_features(features)
        std_t = self._scaled_std(distribution.stddev, noise_scale)
        mean, std = self._ou_conditional(distribution.mean, std_t, noise_prev, noise_phi)
        conditional = Normal(mean, std)
        lp = self._log_prob_squash(conditional.log_prob(raw).sum(-1), raw)
        return lp, conditional.entropy().sum(-1), values

    @torch.no_grad()
    def embedding_scales(self, dynamics, patches, actions,
                         contour_tokens=None, contour_mask=None,
                         relation_tokens=None, relation_mask=None):
        obs, action, _ = self._encode_frames(
            dynamics, patches, actions, contour_tokens, contour_mask, relation_tokens, relation_mask,
        )
        return {"observation_embedding_rms": float(obs.square().mean().sqrt()),
                "action_embedding_rms": float(action.square().mean().sqrt()),
                "action_history_rms": float(actions.square().mean().sqrt()),
                "normalized_action_history_rms": float((actions / self.config.action_scale).square().mean().sqrt())}

    def predict_deterministic(self, *args, **kwargs):
        if self.config.policy_backbone == "gru" or self.config.action_segment_steps > 1:
            raise RuntimeError("Use collect_ppo for stateful GRU / segment inference; a reset window is not equivalent")
        return super().predict_deterministic(*args, **kwargs)
