"""Configure the public player batch; GameRuntime owns its physics backend."""
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

MAX_PLAYERS = 64


def player_settings(settings: dict, num_agents: int, port: int | None = None) -> dict:
    if isinstance(num_agents, bool) or not isinstance(num_agents, int) or not 1 <= num_agents <= MAX_PLAYERS:
        raise ValueError(f"num_agents must be an integer in 1..{MAX_PLAYERS}")
    result = dict(settings)
    result.pop("numDuplicates", None)
    result["numAgents"] = num_agents
    if port is not None:
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("port must be an integer in 1..65535")
        result["tcpPort"] = port
    return result


def configure_players(game_root: Path, num_agents: int, port: int | None = None) -> None:
    """Atomically update count/port while retaining unrelated runtime settings."""
    path = Path(game_root) / "GoiData/runtime_config.json"
    settings = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
    settings = player_settings(settings, num_agents, port)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            json.dump(settings, output, ensure_ascii=False, indent=2)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
