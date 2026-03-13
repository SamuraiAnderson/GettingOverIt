"""
游戏启动器
用于启动 Getting Over It 游戏
"""
from pathlib import Path
import subprocess
import logging

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "project.json"


def _load_exe_path() -> Path:
    """从 project.json 读取游戏可执行文件路径"""
    try:
        import json
        with open(_CONFIG_PATH) as f:
            cfg = json.load(f)
        return Path(cfg["game"]["executable_path"])
    except Exception:
        return None


class GameLauncher:
    """游戏启动器"""

    def __init__(self, game_exe: str = None):
        if game_exe is not None:
            self.game_exe = Path(game_exe)
        else:
            self.game_exe = _load_exe_path()

        if self.game_exe is None or not self.game_exe.exists():
            raise FileNotFoundError(f"游戏执行文件不存在: {self.game_exe}")

        self.game_root = self.game_exe.parent
        logger.info(f"游戏路径: {self.game_exe}")

    def launch(self, wait: bool = False):
        """
        启动游戏

        Returns:
            wait=False: subprocess.Popen 对象
            wait=True:  返回码 int
        """
        logger.info("启动游戏: %s", self.game_exe.name)

        try:
            process = subprocess.Popen(
                [str(self.game_exe)],
                cwd=str(self.game_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            logger.info("游戏已启动 (PID: %d)", process.pid)

            if wait:
                returncode = process.wait()
                logger.info("游戏已退出 (返回码: %d)", returncode)
                return returncode
            return process

        except Exception as e:
            logger.error("启动游戏失败: %s", e)
            raise


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        launcher = GameLauncher()
        wait = "--wait" in sys.argv or "-w" in sys.argv
        launcher.launch(wait=wait)
        if not wait:
            logger.info("游戏已在后台运行（使用 --wait 等待退出）")
    except Exception as e:
        logger.error("错误: %s", e)
        sys.exit(1)
