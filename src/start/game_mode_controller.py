"""
游戏模式控制器
用于从 Python 侧控制 Unity 游戏的运行模式
"""
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "project.json"


def _load_game_root() -> Path:
    """从 project.json 读取游戏根目录"""
    try:
        import json
        with open(_CONFIG_PATH) as f:
            cfg = json.load(f)
        return Path(cfg["game"]["executable_path"]).parent
    except Exception:
        return None


class GameMode(Enum):
    DATA_COLLECTION = "data_collection"
    GAME_RUNTIME    = "game_runtime"
    GAME_TESTING    = "game_testing"


class GameModeController:
    """游戏模式控制器"""

    def __init__(self, game_root: str = None):
        if game_root is not None:
            self.game_root = Path(game_root)
        else:
            self.game_root = _load_game_root()
            if self.game_root is None:
                raise ValueError("无法从 project.json 读取游戏路径，请显式传入 game_root")

        self.signal_dir   = self.game_root / "GoiData" / "ControlSignals"
        self.collider_dir = self.game_root / "GoiData" / "Colliders"

        self.signal_dir.mkdir(parents=True, exist_ok=True)
        self.collider_dir.mkdir(parents=True, exist_ok=True)

        logger.info("游戏根目录: %s", self.game_root)

    def set_mode(self, mode: GameMode):
        self.clear_all_mode_signals()
        signal_file = self.signal_dir / f"mode_{mode.value}.signal"
        signal_file.write_text("")
        logger.info("已设置模式: %s  (信号: %s)", self._get_mode_name(mode), signal_file.name)

    def set_data_collection_mode(self):
        logger.info("设置数据采集模式")
        self.set_mode(GameMode.DATA_COLLECTION)

    def set_game_runtime_mode(self):
        logger.info("设置游戏运行模式")
        self.set_mode(GameMode.GAME_RUNTIME)

    def set_game_testing_mode(self):
        logger.info("设置游戏测试模式")
        self.set_mode(GameMode.GAME_TESTING)

    def clear_all_mode_signals(self):
        for mode in GameMode:
            signal_file = self.signal_dir / f"mode_{mode.value}.signal"
            if signal_file.exists():
                signal_file.unlink()

    def get_current_mode_signal(self):
        for mode in GameMode:
            if (self.signal_dir / f"mode_{mode.value}.signal").exists():
                return mode
        return None

    def get_latest_collider_file(self):
        if not self.collider_dir.exists():
            return None
        txt_files = list(self.collider_dir.glob("*.txt"))
        return max(txt_files, key=lambda p: p.stat().st_mtime) if txt_files else None

    def check_collider_files(self):
        if not self.collider_dir.exists():
            logger.info("碰撞箱目录不存在")
            return []
        txt_files = list(self.collider_dir.glob("*.txt"))
        if txt_files:
            logger.info("找到 %d 个碰撞箱文件:", len(txt_files))
            for f in sorted(txt_files):
                logger.info("  - %s (%d bytes)", f.name, f.stat().st_size)
        else:
            logger.info("未找到碰撞箱文件")
        return txt_files

    def print_status(self):
        current = self.get_current_mode_signal()
        logger.info("当前模式信号: %s", self._get_mode_name(current) if current else "无（使用配置默认）")
        self.check_collider_files()

    @staticmethod
    def _get_mode_name(mode: GameMode) -> str:
        names = {
            GameMode.DATA_COLLECTION: "数据采集模式",
            GameMode.GAME_RUNTIME:    "游戏运行模式",
            GameMode.GAME_TESTING:    "游戏测试模式",
        }
        return names.get(mode, "未知模式")


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    controller = GameModeController()

    if len(sys.argv) > 1:
        mode_arg = sys.argv[1].lower()
        if mode_arg in ("data", "collection", "data_collection"):
            controller.set_data_collection_mode()
        elif mode_arg in ("runtime", "game", "game_runtime"):
            controller.set_game_runtime_mode()
        elif mode_arg in ("test", "testing", "game_testing"):
            controller.set_game_testing_mode()
        elif mode_arg in ("status", "check"):
            controller.print_status()
        else:
            logger.error("未知模式: %s  可用: data, runtime, test, status", mode_arg)
            sys.exit(1)
    else:
        controller.set_data_collection_mode()
