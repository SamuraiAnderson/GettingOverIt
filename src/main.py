"""
Getting Over It - 数据采集主程序
自动设置模式、启动游戏、采集数据
"""
import sys
import time
import logging
from pathlib import Path

# 添加 src 目录到 Python 路径
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from start import GameModeController, GameLauncher

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """主函数：数据采集流程"""
    logger.info("=" * 60)
    logger.info("Getting Over It - 数据采集程序")
    logger.info("=" * 60)
    
    try:
        # 步骤 1: 设置数据采集模式
        logger.info("\n📋 步骤 1/3: 设置数据采集模式")
        logger.info("-" * 60)
        controller = GameModeController()
        controller.set_data_collection_mode()
        
        # 检查现有文件
        logger.info("\n📂 检查现有数据文件:")
        logger.info("-" * 60)
        controller.check_collider_files()
        
        # 步骤 2: 启动游戏
        logger.info("\n🎮 步骤 2/3: 启动游戏")
        logger.info("-" * 60)
        launcher = GameLauncher()
        process = launcher.launch(wait=False)
        
        logger.info("\n⏳ 游戏启动中...")
        logger.info("  - 等待进入 Main 场景（约 10-20 秒）")
        logger.info("  - 插件将自动检测数据采集模式")
        logger.info("  - 自动采集 Mountain 碰撞箱")
        logger.info("  - 导出数据后 2 秒自动退出")
        
        # 步骤 3: 等待游戏完成
        logger.info("\n⌛ 步骤 3/3: 等待数据采集完成")
        logger.info("-" * 60)
        logger.info("等待游戏退出...")
        
        returncode = process.wait()
        
        if returncode == 0:
            logger.info("✅ 游戏正常退出")
        else:
            logger.warning(f"⚠️  游戏退出码: {returncode}")
        
        # 等待一下确保文件写入完成
        time.sleep(1)
        
        # 检查采集结果
        logger.info("\n📊 检查采集结果:")
        logger.info("-" * 60)
        latest = controller.get_latest_collider_file()
        
        if latest:
            size = latest.stat().st_size
            logger.info(f"✅ 找到最新数据文件:")
            logger.info(f"  - 文件: {latest.name}")
            logger.info(f"  - 大小: {size:,} bytes")
            logger.info(f"  - 路径: {latest}")
            
            # 读取文件前几行
            logger.info("\n📄 文件内容预览:")
            logger.info("-" * 60)
            try:
                with open(latest, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:10]
                    for line in lines:
                        print(f"  {line.rstrip()}")
                    if len(lines) >= 10:
                        print(f"  ... (共 {len(lines)} 行)")
            except Exception as e:
                logger.warning(f"无法读取文件: {e}")
        else:
            logger.error("❌ 未找到数据文件")
            logger.info("  请检查游戏日志: BepInEx/LogOutput.log")
        
        # 完成
        logger.info("\n" + "=" * 60)
        logger.info("✅ 数据采集流程完成")
        logger.info("=" * 60)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"\n❌ 文件未找到: {e}")
        logger.info("  请检查游戏安装路径是否正确")
        return 1
        
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  用户中断")
        return 130
        
    except Exception as e:
        logger.error(f"\n❌ 发生错误: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

