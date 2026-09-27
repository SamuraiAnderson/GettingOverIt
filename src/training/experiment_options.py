"""Shared CLI and explicit architecture/input migration policy for PPO experiments."""
import argparse
from .config import TrainConfig
from .policy_features import experimental_enabled, validate_experiments

EXPERIMENT_FIELDS = (
    "policy_backbone", "relation_features", "event_features", "event_branch", "geometry_features",
    "trend_features", "event_source", "normalize_action_history", "feature_position_scale",
    "feature_velocity_scale", "feature_angular_scale", "feature_clip", "feature_step_seconds",
    "geometry_query_radius", "action_segment_steps", "segment_interrupt_contacts",
    "sparse_geometry", "sparse_window_cells", "sparse_window_meters",
    "sparse_max_contour_tokens", "sparse_relation_k", "sparse_mlp_hidden",
)


def add_experiment_arguments(parser):
    parser.add_argument("--init-from", type=str, default=None,
                        help="Warm start compatible weights only; new architecture/features, fresh optimizer and iteration")
    parser.add_argument("--features", nargs="+", choices=("none", "relations", "events", "geometry", "trends"), default=None)
    parser.add_argument("--policy-backbone", choices=("transformer", "gru"), default=None)
    parser.add_argument("--event-source", choices=("native", "geometry"), default=None)
    parser.add_argument("--event-branch", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--normalize-action-history", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--segment-interrupt-contacts", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--action-segment-steps", type=int, default=None)
    parser.add_argument("--recurrent-seq-len", type=int, default=None)
    parser.add_argument("--feature-step-seconds", type=float, default=None)
    parser.add_argument("--sparse-geometry", action=argparse.BooleanOptionalAction, default=None)


def apply_experiment_arguments(config, args, checkpoint=None):
    if args.features is not None:
        if "none" in args.features and len(args.features) != 1:
            raise ValueError("--features none cannot be combined with other groups")
        for group, field in (("relations", "relation_features"), ("events", "event_features"),
                             ("geometry", "geometry_features"), ("trends", "trend_features")):
            setattr(config, field, group in args.features)
    for name in ("policy_backbone", "event_source", "event_branch", "normalize_action_history",
                 "segment_interrupt_contacts", "action_segment_steps", "recurrent_seq_len",
                 "feature_step_seconds", "sparse_geometry"):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    validate_experiments(config)
    if checkpoint:
        defaults = TrainConfig()
        saved = checkpoint.get("config", {})
        changed = [name for name in EXPERIMENT_FIELDS if getattr(config, name) != saved.get(name, getattr(defaults, name))]
        if changed:
            raise ValueError(f"Input/architecture contract changed ({changed}); use --init-from, not resume/checkpoint")
    if experimental_enabled(config):
        if checkpoint and "actor_mean_contact.weight" in checkpoint["model"]:
            raise ValueError("Experimental single head requires --init-from for contact-head checkpoints")
        config.contact_heads_enabled = False
        config.sil_enabled = False  # Legacy SIL cannot reconstruct native events / sequences / segment parameters.
        config.explore_boost_enabled = False
        config.phase_weighting_enabled = False
        config.sil_advantage_weighting_enabled = False


def restore_initialization_shape(config, checkpoint):
    """Retain compatible front-end dimensions; experiment choices remain explicit."""
    for name in ("d_model", "nhead", "num_layers", "context_len", "patch_size", "patch_channels",
                 "patch_mode", "wide_patch_resolution", "contact_patch_resolution", "wide_channels", "contact_channels",
                 "sparse_geometry", "sparse_window_cells", "sparse_window_meters",
                 "sparse_max_contour_tokens", "sparse_relation_k", "sparse_mlp_hidden"):
        if name in checkpoint.get("config", {}):
            setattr(config, name, checkpoint["config"][name])
