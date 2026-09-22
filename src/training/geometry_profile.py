"""Explicit geometry migration; old checkpoints retain their input semantics."""


def configure_geometry(config, checkpoint=None, profile=None):
    saved = (checkpoint or {}).get("config", {})
    fields = ("tip_angle_source", "tip_angle_offset", "tip_scale", "body_angle_offset", "body_scale")
    before = {name: getattr(config, name) for name in fields}
    if checkpoint:
        # Historical checkpoints without a config used the old visual estimates.
        for name, value in {"tip_angle_source": "pole", "tip_angle_offset": -90.,
                            "tip_scale": 1., "body_angle_offset": -90., "body_scale": 1.}.items():
            setattr(config, name, saved.get(name, value))
        before = {name: getattr(config, name) for name in fields}
    if profile is not None:
        if profile not in ("native-pose-v1", "legacy"):
            raise ValueError(f"Unknown geometry profile: {profile}")
        config.tip_angle_source = "pole"
        config.tip_scale = config.body_scale = 1.
        config.tip_angle_offset = -180. if profile == "native-pose-v1" else -90.
        config.body_angle_offset = -84.49833 if profile == "native-pose-v1" else -90.
    return before != {name: getattr(config, name) for name in fields}
