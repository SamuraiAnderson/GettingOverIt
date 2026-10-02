"""Offline configuration compatibility and failure atomicity checks."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from start.runtime_config import configure_players, player_settings
from start.multi_player import launch_players


class MultiPlayerConfigurationChecks(unittest.TestCase):
    def test_legacy_count_migrates_without_losing_other_settings(self):
        original = {"numDuplicates": 8, "tcpPort": 12345, "other": {"keep": True}}
        self.assertEqual(player_settings(original, 2), {"numAgents": 2, "tcpPort": 12345, "other": {"keep": True}})
        self.assertEqual(original["numDuplicates"], 8)

    def test_invalid_count_does_not_create_or_modify_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for count in (0, 65, -1, True, 2.5, "8"):
                with self.assertRaises(ValueError): configure_players(root, count)
            self.assertFalse((root / "GoiData").exists())

    def test_bom_configuration_and_custom_port(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "GoiData").mkdir()
            path = root / "GoiData/runtime_config.json"
            path.write_text('{"numDuplicates": 8, "diagnostic": true}', encoding="utf-8-sig")
            configure_players(root, 64, 12345)
            self.assertEqual(json.loads(path.read_text()), {"numAgents": 64, "tcpPort": 12345, "diagnostic": True})

    def test_failed_replace_retains_previous_file_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configure_players(root, 2)
            path = root / "GoiData/runtime_config.json"
            previous = path.read_bytes()
            with patch.object(Path, "replace", side_effect=PermissionError):
                with self.assertRaises(PermissionError): configure_players(root, 8)
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_launch_configures_requested_installation_and_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("start.multi_player.GameLauncher") as constructor, patch("start.multi_player.GameModeController") as mode:
                launcher = constructor.return_value
                launcher.game_root = root
                launch_players(8, game_root=root, port=12345, desktop=None)
                constructor.assert_called_once_with(str(root / "GettingOverIt.exe"))
                mode.assert_called_once_with(str(root))
                mode.return_value.set_game_runtime_mode.assert_called_once_with()
                launcher.launch.assert_called_once_with(desktop=None)
                self.assertEqual(json.loads((root / "GoiData/runtime_config.json").read_text()), {"numAgents": 8, "tcpPort": 12345})


if __name__ == "__main__": unittest.main()
