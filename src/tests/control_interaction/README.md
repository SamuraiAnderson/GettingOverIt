# 通道验收

游戏进程已经起来之后，检查步进、输入和物理前提是否成立。不训练。

- `test_l1_l2_protocol_causality.py` — 检查 TCP 步进和状态变化是否一一对应。
- `test_l3_repeatability.py` — 检查相同输入是否得到相同的状态变化。
- `test_l4_dose_response.py` — 测量动作幅度和位移响应的关系。
- `test_l5_agent_isolation.py` — 检查多个复制体是否各自响应、互不串输入。
- `test_l5_1_parallel_input.py` — 检查同一帧里不同复制体能不能吃到不同动作。
- `test_l6_collider_visual.py` — 在游戏窗口里描出碰撞箱，供目视确认它跟着玩家走。
- `test_l7_surface_airdrop.py` — 从地形抽出可着陆表面并筛空投落点。
- `test_l8_airdrop_deploy.py` — 检查空投投放是否把复制体放到选定落点。
- `test_l9_water_reset.py` — 检查落水之后能否重置回起点。
- `test_fakecursor_state.py` — 检查回包里的 fakeCursor 位置和速度是否有效。
- `test_mouse_detect.py` — 零动作步进时，看状态会不会跟着真实鼠标变。
- `test_background_run.py` — 比较游戏在前台和最小化时的步进耗时。
- `test_isolated_desktop_launch.py` — 检查游戏能否启动到另一个 Win32 桌面。
- `diag_snapshot_injection.py` — 对拍快照前后，同一动作的注入是否失效。
- `visualize_trajectory.py` — 给上述通道测试画轨迹、距离和剂量响应曲线。
- `test_planb_premise.py` — 检查「多个复制体当作并行世界模型」所需的物理前提。
