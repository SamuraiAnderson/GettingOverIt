"""
游戏模式控制器
用于从 Python 侧控制 Unity 游戏的运行模式
"""
from pathlib import Path
from enum import Enum
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class GameMode(Enum):
    """游戏运行模式"""
    DATA_COLLECTION = "data_collection"  # 数据采集模式
    GAME_RUNTIME = "game_runtime"        # 游戏运行模式
    GAME_TESTING = "game_testing"        # 游戏测试模式


class GameModeController:
    """游戏模式控制器"""
    
    def __init__(self, game_root: str = None):
        """
        初始化控制器
        
        Args:
            game_root: 游戏根目录，如果为 None 则使用默认路径
        """
        if game_root is None:
            game_root = r"C:\Users\Symbol\software\game_store\steam\steamapps\common\Getting Over It"
        
        self.game_root = Path(game_root)
        self.signal_dir = self.game_root / "GoiData" / "ControlSignals"
        self.collider_dir = self.game_root / "GoiData" / "Colliders"
        
        # 确保目录存在
        self.signal_dir.mkdir(parents=True, exist_ok=True)
        self.collider_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"游戏根目录: {self.game_root}")
        logger.info(f"信号目录: {self.signal_dir}")
    
    def set_mode(self, mode: GameMode):
        """
        设置游戏模式
        
        Args:
            mode: 游戏模式
        """
        # 清除所有模式信号
        self.clear_all_mode_signals()
        
        # 创建新的模式信号
        signal_file = self.signal_dir / f"mode_{mode.value}.signal"
        signal_file.write_text("")
        
        logger.info(f"✅ 已设置模式: {self._get_mode_name(mode)}")
        logger.info(f"📁 信号文件: {signal_file.name}")
    
    def set_data_collection_mode(self):
        """设置为数据采集模式"""
        logger.info("=" * 50)
        logger.info("设置数据采集模式")
        logger.info("=" * 50)
        self.set_mode(GameMode.DATA_COLLECTION)
    
    def set_game_runtime_mode(self):
        """设置为游戏运行模式"""
        logger.info("=" * 50)
        logger.info("设置游戏运行模式")
        logger.info("=" * 50)
        self.set_mode(GameMode.GAME_RUNTIME)
    
    def set_game_testing_mode(self):
        """设置为游戏测试模式"""
        logger.info("=" * 50)
        logger.info("设置游戏测试模式")
        logger.info("=" * 50)
        self.set_mode(GameMode.GAME_TESTING)
    
    def clear_all_mode_signals(self):
        """清除所有模式信号文件"""
        for mode in GameMode:
            signal_file = self.signal_dir / f"mode_{mode.value}.signal"
            if signal_file.exists():
                signal_file.unlink()
                logger.debug(f"已删除信号: {signal_file.name}")
    
    def get_current_mode_signal(self):
        """
        获取当前存在的模式信号
        
        Returns:
            GameMode 或 None
        """
        for mode in GameMode:
            signal_file = self.signal_dir / f"mode_{mode.value}.signal"
            if signal_file.exists():
                return mode
        return None
    
    def get_latest_collider_file(self):
        """
        获取最新的碰撞箱数据文件
        
        Returns:
            Path 或 None
        """
        if not self.collider_dir.exists():
            return None
        
        # 查找所有 txt 文件
        txt_files = list(self.collider_dir.glob("*.txt"))
        if not txt_files:
            return None
        
        # 返回最新的文件
        latest = max(txt_files, key=lambda p: p.stat().st_mtime)
        return latest
    
    def check_collider_files(self):
        """检查已存在的碰撞箱文件"""
        if not self.collider_dir.exists():
            logger.info("碰撞箱目录不存在")
            return []
        
        txt_files = list(self.collider_dir.glob("*.txt"))
        if txt_files:
            logger.info(f"找到 {len(txt_files)} 个碰撞箱文件:")
            for f in sorted(txt_files):
                size = f.stat().st_size
                logger.info(f"  - {f.name} ({size} bytes)")
        else:
            logger.info("未找到碰撞箱文件")
        
        return txt_files
    
    @staticmethod
    def _get_mode_name(mode: GameMode) -> str:
        """获取模式显示名称"""
        names = {
            GameMode.DATA_COLLECTION: "数据采集模式",
            GameMode.GAME_RUNTIME: "游戏运行模式",
            GameMode.GAME_TESTING: "游戏测试模式",
        }
        return names.get(mode, "未知模式")
    
    def print_status(self):
        """打印当前状态"""
        logger.info("=" * 50)
        logger.info("当前状态")
        logger.info("=" * 50)
        
        current = self.get_current_mode_signal()
        if current:
            logger.info(f"当前模式信号: {self._get_mode_name(current)}")
        else:
            logger.info("当前模式信号: 无（将使用配置文件默认值）")
        
        self.check_collider_files()
        logger.info("=" * 50)


# ========== 便捷函数 ==========

def setup_data_collection(game_root: str = None):
    """
    设置数据采集模式的便捷函数
    
    Args:
        game_root: 游戏根目录
        
    Returns:
        GameModeController 实例
    """
    controller = GameModeController(game_root)
    controller.set_data_collection_mode()
    return controller


def setup_game_runtime(game_root: str = None):
    """
    设置游戏运行模式的便捷函数
    
    Args:
        game_root: 游戏根目录
        
    Returns:
        GameModeController 实例
    """
    controller = GameModeController(game_root)
    controller.set_game_runtime_mode()
    return controller


def setup_game_testing(game_root: str = None):
    """
    设置游戏测试模式的便捷函数
    
    Args:
        game_root: 游戏根目录
        
    Returns:
        GameModeController 实例
    """
    controller = GameModeController(game_root)
    controller.set_game_testing_mode()
    return controller


# ========== 主函数（用于独立运行） ==========

if __name__ == "__main__":
    import sys
    
    controller = GameModeController()
    
    if len(sys.argv) > 1:
        mode_arg = sys.argv[1].lower()
        if mode_arg in ["data", "collection", "data_collection"]:
            controller.set_data_collection_mode()
        elif mode_arg in ["runtime", "game", "game_runtime"]:
            controller.set_game_runtime_mode()
        elif mode_arg in ["test", "testing", "game_testing"]:
            controller.set_game_testing_mode()
        elif mode_arg in ["status", "check"]:
            controller.print_status()
        else:
            logger.error(f"未知模式: {mode_arg}")
            logger.info("可用模式: data, runtime, test, status")
            sys.exit(1)
    else:
        # 默认：数据采集模式
        controller.set_data_collection_mode()
        logger.info("\n💡 提示: 可以通过命令行参数指定模式")
        logger.info("  python game_mode_controller.py data      # 数据采集")
        logger.info("  python game_mode_controller.py runtime   # 游戏运行")
        logger.info("  python game_mode_controller.py test      # 游戏测试")
        logger.info("  python game_mode_controller.py status    # 查看状态")
    
    logger.info("\n✅ 完成！现在可以启动游戏了。")

