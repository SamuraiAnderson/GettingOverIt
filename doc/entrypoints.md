# 项目执行入口清单

本文档汇总项目中所有可执行入口（带 `if __name__ == "__main__"` 的脚本），按用途分类。

## 一、核心业务入口（`src/` 根目录）

| 入口 | 命令 | 作用 |
|------|------|------|
| `src/main.py` | `python src/main.py` | 数据采集主程序：设置数据采集模式 → 启动游戏 → 自动采集 Mountain 碰撞箱 → 导出后退出 |
| `src/explore_env.py` | `python src/explore_env.py --agents 3 --steps 100` | 环境能力探测：训练前验证帧同步、步进延迟、确定性、多 agent 独立性、reset 保真性 |

## 二、训练入口（`src/training/`）

| 入口 | 命令 | 作用 |
|------|------|------|
| `main_train.py` | `python -m src.training.main_train` | BC 行为克隆迭代训练（冷启动随机采集 → 效率图 → 筛选轨迹 → fine-tune）。`--resume` 跳过冷启动 |
| `main_ppo.py` | `python -m src.training.main_ppo` | PPO 在线强化学习。支持 `--num-agents` `--max-iterations` `--steps-per-rollout` `--resume` `--resume-bc` `--random-deploy` |

## 三、启动 / 模式控制（`src/start/`，可单独运行）

| 入口 | 作用 |
|------|------|
| `game_launcher.py` | 从 `project.json` 启动游戏进程 |
| `game_mode_controller.py` | 写信号文件切换 GameRuntime 模式（数据采集 / RL 运行时） |

## 四、控制交互测试 L1–L9（`src/tests/control_interaction/`）

| 入口 | 层次 | 作用 |
|------|------|------|
| `test_l1_l2_protocol_causality.py` | L1-L2 | TCP 协议因果性 |
| `test_l3_repeatability.py` | L3 | 轨迹可重复性 |
| `test_l4_dose_response.py` | L4 | 动作-响应剂量关系 |
| `test_l5_agent_isolation.py` / `test_l5_1_parallel_input.py` | L5 | 多 agent 物理隔离 / 并行输入 |
| `test_l6_collider_visual.py` | L6 | 碰撞体可视化 |
| `test_l7_surface_airdrop.py` | L7 | 可着陆面分析 + 投放点生成（`--physics-filter`） |
| `test_l8_airdrop_deploy.py` | L8 | 批量 deploy + settle 落差统计 |
| `test_l9_model_eval.py` / `test_l9_water_reset.py` | L9 | 模型评估 / 落水重置 |
| `drop_point_physics.py` | 辅助 | 物理稳定性过滤引擎 |
| `visualize_cached_drops.py` | 辅助 | 空投超参可视化 |
| `analyze_ppo.py` | 辅助 | PPO checkpoint 趋势分析 |

## 五、其他测试

- 环境交互（`src/tests/environment_interaction/test_scripts/`）：`test_state_consistency.py`、`test_state_checkpoint.py`、`test_full_reset.py`
- 输入特性（`src/tests/input_characteristics/`）：`test_max_input.py`、`test_max_refine.py`、`precision_test_50.py`、`precision_test_decimal.py`、`dose_response_test.py`

## 六、C# 侧入口

C# `GameRuntime_v2` 是 BepInEx 插件，无独立可执行入口，由游戏进程加载 `GameRuntimeManager` 后自动运行 TCP 服务端（默认端口 9000，从 `src/config/project.json` 读取）。

## 七、日常最常用

1. `src/main.py` — 采集数据 / 碰撞箱
2. `src/explore_env.py` — 训练前验证环境
3. `src/training/main_train.py` / `main_ppo.py` — 训练
