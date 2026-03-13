"""
Getting Over It - 数据采集主程序
自动设置模式、启动游戏、采集数据
"""
import sys
import time
import logging
from pathlib import Path

src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from start import GameModeController, GameLauncher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Getting Over It - 数据采集程序")
    logger.info("=" * 60)

    try:
        logger.info("步骤 1/3: 设置数据采集模式")
        logger.info("-" * 60)
        controller = GameModeController()
        controller.set_data_collection_mode()

        logger.info("检查现有数据文件:")
        controller.check_collider_files()

        logger.info("步骤 2/3: 启动游戏")
        logger.info("-" * 60)
        launcher = GameLauncher()
        process = launcher.launch(wait=False)

        logger.info("游戏启动中...")
        logger.info("  - 等待进入 Main 场景（约 10-20 秒）")
        logger.info("  - 插件将自动检测数据采集模式")
        logger.info("  - 自动采集 Mountain 碰撞箱")
        logger.info("  - 导出数据后 2 秒自动退出")

        logger.info("步骤 3/3: 等待数据采集完成")
        logger.info("-" * 60)

        returncode = process.wait()

        if returncode == 0:
            logger.info("游戏正常退出")
        else:
            logger.warning("游戏退出码: %d", returncode)

        time.sleep(1)

        logger.info("检查采集结果:")
        logger.info("-" * 60)
        latest = controller.get_latest_collider_file()

        if latest:
            size = latest.stat().st_size
            logger.info("找到最新数据文件: %s (%d bytes)", latest.name, size)
            logger.info("路径: %s", latest)

            logger.info("文件内容预览:")
            try:
                with open(latest, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:10]
                    for line in lines:
                        logger.info("  %s", line.rstrip())
                    if len(lines) >= 10:
                        logger.info("  ... (共 %d 行)", len(lines))
            except Exception as e:
                logger.warning("无法读取文件: %s", e)
        else:
            logger.error("未找到数据文件，请检查游戏日志: BepInEx/LogOutput.log")

        logger.info("=" * 60)
        logger.info("数据采集流程完成")
        logger.info("=" * 60)

        return 0

    except FileNotFoundError as e:
        logger.error("文件未找到: %s  请检查游戏安装路径", e)
        return 1

    except KeyboardInterrupt:
        logger.info("用户中断")
        return 130

    except Exception as e:
        logger.error("发生错误: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
