# start - 游戏启动模块

## 文件说明

- `game_mode_controller.py` - 游戏模式控制器（设置数据采集/游戏运行/游戏测试模式）
- `game_launcher.py` - 游戏启动器（启动 Getting Over It 游戏）
- `__init__.py` - 包初始化文件

## 快速使用

### 1. 一键运行（推荐）

```bash
cd C:\Users\Symbol\aCodes\GettingOverIt\src
python main.py
```

### 2. 独立使用

```python
from start import GameModeController, GameLauncher

# 设置模式
controller = GameModeController()
controller.set_data_collection_mode()

# 启动游戏
launcher = GameLauncher()
launcher.launch_and_wait()

# 检查结果
latest = controller.get_latest_collider_file()
print(f"数据文件: {latest}")
```

### 3. 命令行工具

```bash
# 查看状态
python start/game_mode_controller.py status

# 设置模式
python start/game_mode_controller.py data      # 数据采集
python start/game_mode_controller.py runtime   # 游戏运行
python start/game_mode_controller.py test      # 游戏测试

# 启动游戏
python start/game_launcher.py --wait
```

## 便捷函数

```python
from start import setup_data_collection, setup_game_runtime, launch_game

# 一键设置并启动
controller = setup_data_collection()
process = launch_game(wait=True)
```

## 配置

默认游戏路径：`C:\Users\Symbol\software\game_store\steam\steamapps\common\Getting Over It`

自定义路径：
```python
controller = GameModeController(game_root="你的游戏路径")
launcher = GameLauncher(game_root="你的游戏路径")
```

