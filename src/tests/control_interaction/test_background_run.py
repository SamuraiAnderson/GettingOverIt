"""
验证游戏后台运行：对比前台/最小化时的 step 响应速度
程序会自动最小化游戏窗口
"""
import sys, time, ctypes
sys.path.insert(0, r'C:\Users\Symbol\aCodes\GettingOverIt\src')

SW_MINIMIZE  = 6
SW_RESTORE   = 9
GAME_TITLE   = "Getting Over It"

def _find_game_hwnd():
    hwnd = ctypes.windll.user32.FindWindowW(None, GAME_TITLE)
    return hwnd

def minimize_game():
    hwnd = _find_game_hwnd()
    if hwnd:
        ctypes.windll.user32.ShowWindow(hwnd, SW_MINIMIZE)
        print(f'[窗口] 游戏已最小化 (hwnd={hwnd})', flush=True)
    else:
        print('[窗口] 未找到游戏窗口', flush=True)

def restore_game():
    hwnd = _find_game_hwnd()
    if hwnd:
        ctypes.windll.user32.ShowWindow(hwnd, SW_RESTORE)
        print(f'[窗口] 游戏已还原 (hwnd={hwnd})', flush=True)

import numpy as np
from start.game_mode_controller import GameModeController
from start.game_launcher import GameLauncher
from env.goi_env import GoiEnv

GameModeController().set_game_runtime_mode()
proc = GameLauncher().launch(wait=False)
print(f'游戏已启动 PID={proc.pid}', flush=True)

ZERO = np.array([[0.0, 0.0]], dtype=np.float32)

with GoiEnv(port=9000, num_agents=1, timeout=60) as env:
    print('已连接，执行 reset...', flush=True)
    env.reset()
    print('reset OK', flush=True)

    # --- 前台测试 ---
    N = 30
    print(f'\n[前台] 执行 {N} 步...', flush=True)
    t0 = time.perf_counter()
    for _ in range(N):
        env.step(ZERO)
    fg_time = time.perf_counter() - t0
    print(f'[前台] {N} 步耗时 {fg_time:.2f}s  ({fg_time/N*1000:.1f}ms/step)', flush=True)

    # --- 自动最小化 ---
    print('\n[窗口] 自动最小化游戏窗口...', flush=True)
    time.sleep(0.5)
    minimize_game()
    time.sleep(1)  # 等待最小化完成

    # --- 后台测试 ---
    print(f'\n[后台] 执行 {N} 步...', flush=True)
    t0 = time.perf_counter()
    for _ in range(N):
        env.step(ZERO)
    bg_time = time.perf_counter() - t0
    print(f'[后台] {N} 步耗时 {bg_time:.2f}s  ({bg_time/N*1000:.1f}ms/step)', flush=True)

    # --- 还原窗口 ---
    restore_game()

    ratio = bg_time / fg_time
    print(f'\n后台/前台速度比: {ratio:.2f}x', flush=True)
    if ratio < 1.5:
        print('结论: 后台运行正常 [OK] (速度未显著下降)', flush=True)
    else:
        print(f'结论: 后台被限速 ({ratio:.1f}x 变慢)', flush=True)
