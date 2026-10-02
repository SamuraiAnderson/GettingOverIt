"""Public launch entrypoint for a controlled batch of native players."""
import json
from pathlib import Path

from .game_launcher import GameLauncher
from .game_mode_controller import GameModeController
from .runtime_config import configure_players


def launch_players(num_agents: int, *, game_root=None, port=None, **launch_options):
    """Configure count/port and launch the same installation in GameRuntime mode.

    Process ownership remains with the caller. Pass desktop=None for a visible
    game; otherwise GameLauncher's isolated desktop default is preserved.
    """
    launcher = GameLauncher(str(Path(game_root) / "GettingOverIt.exe")) if game_root is not None else GameLauncher()
    if port is None:
        project = Path(__file__).resolve().parents[1] / "config/project.json"
        port = json.loads(project.read_text(encoding="utf-8-sig"))["tcp_port"]
    configure_players(launcher.game_root, num_agents, port)
    GameModeController(str(launcher.game_root)).set_game_runtime_mode()
    return launcher.launch(**launch_options)
