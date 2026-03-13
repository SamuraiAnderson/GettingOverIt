"""
游戏启动模块
包含游戏模式控制和游戏启动功能
"""

from .game_mode_controller import (
    GameMode,
    GameModeController,
    setup_data_collection,
    setup_game_runtime,
    setup_game_testing
)

from .game_launcher import (
    GameLauncher,
    launch_game
)

__all__ = [
    'GameMode',
    'GameModeController',
    'GameLauncher',
    'setup_data_collection',
    'setup_game_runtime',
    'setup_game_testing',
    'launch_game',
]

