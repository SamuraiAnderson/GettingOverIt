"""
游戏启动器
用于启动 Getting Over It 游戏
"""
from pathlib import Path
import subprocess
import logging
import time
import sys

logger = logging.getLogger(__name__)


class GameLauncher:
    """游戏启动器"""
    
    def __init__(self, game_root: str = None):
        """
        初始化启动器
        
        Args:
            game_root: 游戏根目录
        """
        if game_root is None:
            game_root = r"C:\Users\Symbol\software\game_store\steam\steamapps\common\Getting Over It"
        
        self.game_root = Path(game_root)
        self.game_exe = self.game_root / "GettingOverIt.exe"
        
        if not self.game_exe.exists():
            raise FileNotFoundError(f"游戏执行文件不存在: {self.game_exe}")
        
        logger.info(f"游戏路径: {self.game_exe}")
    
    def launch(self, wait: bool = False):
        """
        启动游戏
        
        Args:
            wait: 是否等待游戏退出
            
        Returns:
            subprocess.Popen 对象（如果 wait=False）
            返回码（如果 wait=True）
        """
        logger.info("=" * 50)
        logger.info("启动游戏")
        logger.info("=" * 50)
        logger.info(f"执行文件: {self.game_exe.name}")
        
        try:
            # 启动游戏
            process = subprocess.Popen(
                [str(self.game_exe)],
                cwd=str(self.game_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            logger.info(f"✅ 游戏已启动 (PID: {process.pid})")
            
            if wait:
                logger.info("等待游戏退出...")
                returncode = process.wait()
                logger.info(f"游戏已退出 (返回码: {returncode})")
                return returncode
            else:
                return process
                
        except Exception as e:
            logger.error(f"❌ 启动游戏失败: {e}")
            raise
    
    def launch_and_wait(self, timeout: int = None):
        """
        启动游戏并等待退出
        
        Args:
            timeout: 超时时间（秒），None 表示无限等待
            
        Returns:
            返回码
        """
        process = self.launch(wait=False)
        
        try:
            if timeout:
                logger.info(f"等待游戏退出（最多 {timeout} 秒）...")
                returncode = process.wait(timeout=timeout)
            else:
                logger.info("等待游戏退出...")
                returncode = process.wait()
            
            logger.info(f"✅ 游戏已退出 (返回码: {returncode})")
            return returncode
            
        except subprocess.TimeoutExpired:
            logger.warning(f"⏰ 游戏运行超过 {timeout} 秒，强制终止")
            process.kill()
            return -1


# ========== 便捷函数 ==========

def launch_game(game_root: str = None, wait: bool = False):
    """
    启动游戏的便捷函数
    
    Args:
        game_root: 游戏根目录
        wait: 是否等待游戏退出
        
    Returns:
        subprocess.Popen 对象或返回码
    """
    launcher = GameLauncher(game_root)
    return launcher.launch(wait=wait)


# ========== 主函数（用于独立运行） ==========

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    try:
        launcher = GameLauncher()
        
        # 检查命令行参数
        wait = "--wait" in sys.argv or "-w" in sys.argv
        
        if wait:
            launcher.launch_and_wait()
        else:
            launcher.launch()
            logger.info("\n💡 游戏已在后台运行")
            logger.info("  使用 --wait 参数等待游戏退出")
    
    except Exception as e:
        logger.error(f"错误: {e}")
        sys.exit(1)

