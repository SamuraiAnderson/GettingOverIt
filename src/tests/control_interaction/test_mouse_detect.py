"""
极简鼠标检测：启动游戏，发送零动作 30 步，观察 state 是否随鼠标移动变化。
运行期间请来回移动鼠标。
"""
import sys
sys.path.insert(0, r'C:\Users\Symbol\aCodes\GettingOverIt\src')

from env.goi_env import GoiEnv
from start.game_launcher import GameLauncher
from start.game_mode_controller import GameModeController
import time
import numpy as np

# 先设置信号文件，再启动游戏
GameModeController().set_game_runtime_mode()
GameLauncher().launch(wait=False)
time.sleep(12)  # 等待游戏完成初始化并进入 RL 模式

env = GoiEnv()
env.connect()

obs = env.reset()
print(f'[Reset] player_x={obs[0]:.4f}  player_y={obs[1]:.4f}')
print('=== 开始检测，请来回移动鼠标 ===')

prev_x = obs[0]
for i in range(30):
    obs, _, _ = env.step([0.0, 0.0])
    dx = obs[0] - prev_x
    prev_x = obs[0]
    print(f'step {i:2d}  player_x={obs[0]:.4f}  vel_x={obs[2]:.4f}  Δx={dx:+.4f}')
    time.sleep(0.3)

print('=== 检测结束 ===')
env.close()
