"""Frozen 8-player evaluation on the current desktop. No training updates."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import start.game_launcher as game_launcher

_orig_launch = game_launcher.GameLauncher.launch


def _launch_here(self, wait=False, *, desktop=game_launcher._ISOLATED_DESKTOP):
    return _orig_launch(self, wait=wait, desktop=None)


game_launcher.GameLauncher.launch = _launch_here

from training.local_evaluation import main

if __name__ == "__main__":
    main()
