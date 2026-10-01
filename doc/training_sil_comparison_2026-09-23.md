# C + corridor-SIL 正式规格对照

2026-09-23 09:12:49（北京时间）启动，目录 `runs/corridor_sil_20260923`。单次后台控制器依次执行 seed 31、47，随后各自评估并写入对照报告；不需要对话继续派发。

- C：gamma=0.99、lambda=0.95；8 agents、1200 步、PPO batch=128、lr=3e-5、CPU threads=4。
- PPO+OU，B1 Transformer，归一化动作历史，native-pose-v1，固定起点；两轮 critic 预热计入训练样本。
- 初始化源与正式 run 相同：`runs/p2_contact/checkpoints/ppo_iter_0335.pt`，重建优化器。
- 增加 `--corridor-sil --sil-seed runs/best_motion_20260923/sil_seed_590.npz`。SIL 保留现有默认：第 6 轮起、优势加权、batch=64、每轮至多 40 minibatches；PPO batch 仍为 128。
- 以每个 seed 达到 96,000 个实际训练物理样本为终点，整轮结束后检查；最多超出不足 9,600 个样本。10 轮是满长轨迹的理想值，落水终止会减少样本，历史正式 C seed 31/47 分别用了 12/11 轮。
- 连续运行保留内存 SIL 精英池。新增 `--target-physical-samples` 只控制整轮停止位置；不改变采集、奖励或优化更新。20 轮及两小时为异常保护上限，达到样本目标后不追加训练。

评估与正式 96k 中期锚点相同：评估 seed=20260922、2 rounds × 8 agents，deterministic 和 exploration 各 16 回合，1200 步，jitter=0.01m、goal=5m、dwell=10，并重评分 120 步保持成功。每个 seed 保存独立比较与实际训练样本数，另汇总两个 seed 的回合结果。

原锚点 det 中位数：C seed 31 为 2.715m（102,100 样本），C seed 47 为 3.834m（97,205 样本）；2.72m 是 seed 31 的值。SIL 种子来自历史 T 组中期评估挑选的轨迹，因此本次是种子迁移效果对照。连续执行也不同于原正式 run 的分段重启与重新设随机种子。

启动前共 35 项不同的相关测试通过，覆盖现有 PPO/走廊/SIL、实际 CLI 配置与历史配置逐字段对齐、样本终点参数和原正式控制器。真实种子管线验证 credited=8.0991m、590 个监督窗口，走廊引导 patch 重建与在线路径的 wide 通道最大差为 0。

## 输出与停止

- 实时状态：`runs/corridor_sil_20260923/status.json`
- 配置、命令、截止：`runs/corridor_sil_20260923/manifest.json`
- 对照报告：`runs/corridor_sil_20260923/RESULT.md` 和 `comparison.json`
- 每轮权重、输入和精英池：`runs/corridor_sil_20260923_C_s31/`、`runs/corridor_sil_20260923_C_s47/`
- 启动记录：`runs/sil_launch_logs/`

控制器 PID=35440。创建 `runs/corridor_sil_20260923/STOP` 文件后，控制器在下一次检查时结束其子进程树，保留已完成 checkpoint。正常结束或异常停止均恢复启动前 `GoiData/runtime_config.json`。训练过程中不要编辑冻结的训练、环境、评估源文件或种子文件。
