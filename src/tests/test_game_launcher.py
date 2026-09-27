"""独立桌面启动：结构检查，不启动游戏。"""
import ctypes
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from start.game_launcher import _STARTUPINFOW, desktop_path


class DesktopLaunchContractTests(unittest.TestCase):
    def test_desktop_path_stays_on_the_interactive_window_station(self):
        self.assertEqual(desktop_path("GoiRuntime"), "WinSta0\\GoiRuntime")

    def test_startupinfo_matches_64bit_windows_layout(self):
        self.assertEqual(ctypes.sizeof(_STARTUPINFOW), 104)
        self.assertEqual(_STARTUPINFOW.lpDesktop.offset, 16)


if __name__ == "__main__":
    unittest.main()
