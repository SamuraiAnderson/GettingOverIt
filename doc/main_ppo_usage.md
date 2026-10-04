# PPO 运行手册（`main_ppo`）

本文说明 `python -m src.training.main_ppo` **怎么跑**、各 CLI 参数做什么、哪些路径应视为废弃/慎用。
算法细节、观测/奖励口径见 [`training.md`](training.md)；优化优先级见 [`optimization_roadmap.md`](optimization_roadmap.md)。

环境：统一用 conda `getting-over-it-analysis`。

## 当前主线：GRU、精确出生状态

2026-10-03 的三个配对种子比较后，新建 PPO 默认使用 **单层 GRU128**、动作历史
归一化、TBPTT 32、固定出生点、单头 PPO+OU，停用旧 SIL。其余训练预算参数仍可
由 CLI 指定。以下命令沿用本次比较的采样与更新设置，开始新的主线训练：

```powershell
conda run --no-capture-output -n getting-over-it-analysis python -m src.training.main_ppo --run-dir runs/gru_mainline --reference-trajectory --recipe ppo-ou --seed 7 --num-agents 8 --steps-per-rollout 1200 --ppo-batch-size 32 --value-warmup-iters 2 --max-iterations 100 --persist-game --torch-threads 4
```

加 `--dry-run` 只查看解析后的配置，不启动游戏。需要 Transformer 对照时显式传
`--policy-backbone transformer`。`--resume PATH` 始终恢复原 checkpoint 的架构和输入
语义，不把 Transformer 权重当作 GRU；转换架构须使用 `--init-from PATH`，且会重新
开始迭代与优化器。旧 BC 入口仍保留其原结构。

GRU 使用独立的 `--gru-sil` 路径；`--corridor-sil` 和 `--recipe ppo-ou-sil` 仍会明确报错。
走廊 v2 任务可另加 `--elite-archive`，独立保存完整精英回合，SIL 仍关闭。
归档沿用 (到达时间 t, 确认距离 D) Pareto 与少量空间多样性准入，拒绝落水及无效轨迹，
默认容量 20（`--elite-archive-capacity`）。保留从 reset 起的全部状态、动作、接触、
奖励、终止及走廊字段，既不截掉历史前缀，也不删除末端回落；归档不参与 PPO 梯度。
快照与 checkpoint 同步保存到 `elites/full_episodes/elites_iter_NNNN.npz`，包括任务指纹
与完整回合格式标记。`--resume` 自动恢复源 checkpoint 同迭代的快照；
`--elite-archive-resume PATH` 可显式指定同任务的完整回合池，不接受旧剪枝 SIL 池。
旧 checkpoint 未启用归档时从空池开始；之前未保存的轨迹无法补回。
保存的是既有观测和执行动作，不保存作为训练真值的旧隐藏态。

### GRU SIL（显式启用）

在上述命令或 `--resume` 命令中加 `--gru-sil`，会自动启用完整精英归档。
`--no-gru-sil` 关闭模仿但保留已经启用的归档；续训默认恢复 checkpoint 的开关。
当前支持主线基础观测（含稀疏几何）、单头 GRU、逐帧动作与走廊 v2，额外事件/趋势
特征和动作段会拒绝。旧窗口 SIL 池不含可靠的完整 reset 前缀，不能混用。

每次更新均匀抽取一条精英轨迹及一个连续 TBPTT 片段。用**当前权重**从 reset 无梯度
重放全部前缀，再对片段反传。第 t 帧输入的是实际执行的第 t−1 帧动作，首帧为零。
只模仿最后一次确认进展之前、`R−V>0` 的动作；折扣回报 R 使用完整奖励尾段，真正
终止取零 bootstrap，时间截断用当前网络重建到末帧后的 V。归档本身仍保留回落尾段。

这是正优势加权的动作均值模仿：目标为执行动作的 inverse-tanh（限幅 ±3），优势限幅 5，
策略系数 0.1、正优势帧的 Huber 价值系数 0.05。SIL 不更新探索方差，不对负优势强行
提供监督；含 OU 噪声的精英动作不能解释为无噪声最优动作或 PPO 的离策略似然。

默认每轮最多 `min(4, floor(PPO实际更新数 × 0.25))` 个候选片段，预热期间与前 5 轮
不启用。每次更新后，对本轮全部在线决策重放当前网络，计算相对 **PPO 更新后** 策略的
累计 OU 条件 KL（包含各帧噪声倍率）。平均 KL 上限 0.01、单帧上限 0.1；超限同时回滚
模型与 Adam 状态并结束本轮 SIL。这只约束已采样历史上的漂移，不保证未知状态上的行为。
超参数存于 `config.py` 的 `gru_sil_*` 字段；指标为 `metrics.csv` 的 `gru_sil_*` 列，
逐片段起止、正优势数量、bootstrap、KL 和接受状态保存于 `sil_sampling/iter_NNNN.json`。

可加 `--stop-file PATH`；创建该文件后训练完成当前轮，再保存并正常退出。
重新启动前须移除该文件或换一个新路径。

固定出生评估默认不扰动；可显式写出完整验证命令：

```powershell
conda run --no-capture-output -n getting-over-it-analysis python -m src.training.local_evaluation --checkpoint runs/gru_mainline/checkpoints/ppo_iter_0100.pt --output runs/gru_mainline/eval_exact --steps 1200 --rounds 2 --num-agents 2 --jitter 0 --modes deterministic
```

比较归档为 `runs/backbone_compare_20261002/`；GRU 的路线进度领先，但保持成功仍为零。

新增可选的[视频参考走廊模式](reference_corridor_reward.md)：使用 `--reference-trajectory`
启用“走廊内前进优先”的奖励，`--reference-corridor-radius 1.5` 设置半宽。
该模式替换原 PPO 引导奖励和 wide 引导图；SIL 精英池在走廊模式下自动切换为
**credited 进度口径**（剪枝/排序/行为描述子/BC 截断全部基于逐帧记录的
`corridor_progress`，patch wide 通道按 `guidance_index` 重建引导段地图），
不再按高度筛选；不传时沿用本文原配方。

---

## 一、走廊命令里仍然有效的开关

下面这条 2 人、40 轮的「当前主线」命令已删除。续训用 `main_ppo --resume`，配方见后来的实验记录。

`--recipe ppo` 关闭 OU，`ppo-ou` 保留 OU；两者均关闭旧 SIL、停滞加噪、接触分头、阶段权重及旧 SIL 优势权重。`--gru-sil` 独立控制 GRU 模仿更新。历史 Transformer/SIL 路径另行保留，不适用于 GRU 主线。

`--seed` 固定 Python、NumPy、Torch 和投放采样 RNG；当前隔离物理运行时已通过精确出生状态的重复一致性验证。`--geometry-profile native-pose-v1` 使用原生姿态标定（锤头 −180°、身体约 −84.498°）；`legacy` 为原来的 −90°/−90°。不传时，新模型使用新标定，旧 checkpoint 恢复其配置，缺少配置的旧权重回退 legacy。BC 入口继续使用 legacy。

2026-10-03 一致性修复后，checkpoint 在完整迭代边界保存 Python、NumPy、Torch CPU/CUDA
以及投放器的随机状态，连同 critic 预热优化器、KL/梯度范数适配器及循环状态一起恢复。
`--resume` 默认继续原来的随机序列；省略 `--seed` 时沿用保存的初始 seed。确需换序列时
显式传 `--reset-rng --seed N`。旧 checkpoint 没有这些状态时仍可续训，但会记录 seed
重启提示，不能声称等同于不中断训练。跨设备/运行时的数值逐位复现不在此保证范围内。
策略概率审计使用独立 RNG，不再消耗训练的随机序列；GRU SIL 仍使用独立的 seed+迭代采样。
完整精英池恢复保留原顺序和 float64 分数，避免相同抽样索引在恢复后指向另一条轨迹。

全新模型按 `ppo_init_log_std` 初始化（默认 −1.6，raw std≈0.202）；`--resume` 保留学到的
方差，`--init-from` 默认也继承来源方差。`--lr` 在恢复 Adam 后仍生效。
GRU 不支持 `--phase-noise`，会明确报错；`--delayed-exploration` 则在续训时默认继承，
可用 `--no-delayed-exploration` 显式关闭。评估的 `--noise-phi` 默认沿用 checkpoint：
训练为 iid 则评估探索也为 iid；显式指定该参数才覆盖。

`--resume` 先恢复保存的配置，再应用显式 CLI 覆盖（本机 game_root/port 除外）。更换参考奖励或几何标定会从当前迭代重新预热 critic。`--max-iterations` 是累计迭代上限：从 335 恢复并传 337，只执行两轮。`--value-warmup-iters 0` 仅在明确需要跳过预热时使用。

`--save-rollout-inputs` 将每轮采样权重和完整输入保存到 `checkpoints/rollouts/`，便于冻结重放；体积较大，通常用于验收或短诊断。采集自动批推理，`TrainConfig.ppo_inference_batch_size=1` 可用于串行对照，`0` 批处理全部活跃 agent。patch 按帧共享，minibatch 才组装窗口，CNN 更新仍重新计算。

固定评估走 `local_evaluation`。评估分别执行无噪声与指定 OU 探索，默认要求同段有效前进 5 米、当前位置前进 5 米并在走廊内保持 10 步。`--goal-progress`、`--dwell` 可调整；`--jitter` 默认 0，所有轮次均从精确出生状态开始；显式设置非零值才测试位置扰动。记录成功、跌落、离开走廊、到达步数、输入和接触/动作事件，详情见[执行记录](optimization_execution_2026-09-19.md)。加 `--native-diagnostics` 需要新版插件，会增加诊断通信开销。

### P1.3 / P2 表示候选

GRU 已成为新建 PPO 的默认主线，仍使用 39D 动力学与逐步动作；Transformer 保留为显式对照。开关、单位、序列与段概率契约及验证边界见[表示实验](policy_representation_experiments.md)。

- `--features relations events geometry trends`：选择关系、事件、局部几何和多尺度趋势；可只选一组，`none` 清空这些组。
- `--event-branch`：小事件 MLP，同时送入 backbone 和 actor/critic，并自动启用事件输入；`--event-source native|geometry` 指定事件来源，默认 native。
- `--policy-backbone transformer|gru`：选择时序骨干；新任务默认 GRU，`--recurrent-seq-len 32` 控制训练块的决策数。
- `--normalize-action-history`：动作历史除以 100，执行动作不变。
- `--action-segment-steps 4`：两端点线性插值短段；默认接触边缘中断，`--no-segment-interrupt-contacts` 可关闭。
- `--feature-step-seconds`：特征计时所用物理步长，默认 1/120 秒；native 模式与游戏不一致时报错。
- `--init-from PATH`：只迁移兼容 PPO 权重，新优化器和新迭代；与 `--resume`/`--resume-bc` 互斥。结构或输入语义变化使用它，`--resume` 用于原契约续训。

例如只验证关系特征：

```powershell
conda run --no-capture-output -n getting-over-it-analysis python -m src.training.main_ppo --init-from runs/p2_contact/checkpoints/ppo_iter_0335.pt --features relations --normalize-action-history --reference-trajectory --geometry-profile native-pose-v1 --recipe ppo-ou --seed 7 --no-random-deploy --num-agents 2 --steps-per-rollout 96 --max-iterations 2 --value-warmup-iters 1 --run-dir relations_smoke --persist-game --save-rollout-inputs
```

这些开关同样适用于 `local_evaluation`。候选当前使用单头并停用旧 SIL；原生事件或原生接触中断会自动启用 D v2 诊断，不需要额外传 `--native-diagnostics`。段策略的 `buffer_size` 是决策数，比较样本预算应读 `rollout_environment_steps`。

### 原奖励最小任务（历史配方）

以下仅描述原高度奖励的历史实验。目标：固定起点、短 episode，观察是否翻过 Deadtree。先不要开 random deploy。

```bash
conda run -n getting-over-it-analysis python -m src.training.main_ppo \
  --no-random-deploy \
  --steps-per-rollout 400 \
  --max-iterations 40 \
  --run-dir my_experiment \
  --persist-game \
  --resume-bc <某 BC ckpt>   # 或 --resume <某 PPO ckpt>
```

`--run-dir` 会把这次实验的全部产物收敛到 `runs/my_experiment/`：权重、效率图、
控制台镜像、逐轮 `metrics.csv`、`run_meta.json`（CLI + config + git commit）以及
SIL 精英池快照。布局与保留策略见 [`runs/README.md`](../runs/README.md)。

观察信号：

- `runs/<tag>/logs/` 效率图：peak 是否开始出现 `y > 2`
- `runs/<tag>/metrics.csv` 的 `me_cells` 列（或日志 `[SIL/MAP-Elites] cells=`）：
  cell 数是否随迭代增长
- `runs/<tag>/metrics.csv` 的 `explore_pool_best_sdy` 列：精英池最佳竖直进展是否还在涨

CLI 未暴露的 OU / MAP-Elites 默认已开（`TrainConfig.ou_enabled=True`，`sil_map_elites_enabled=True`）。

---

## 二、CLI 速查

新模型未传的参数保持 `TrainConfig` 默认值（`src/training/config.py`）；恢复模型时优先保存的配置，显式 CLI 最后覆盖。

奖励/时间尺度对照可显式设置 `--reference-deviation-weight`（走廊外每物理步惩罚上限）、`--gamma`、`--gae-lambda`。默认值分别为 0.01、0.99、0.95，省略时续训保留 checkpoint 设置。gamma/lambda 不能在 `--resume` 时改变；改变时间配方请用 `--init-from` 新建运行。第二轮固定对照与自动终点评估见[训练方案](training_credit_plan_2026-09-19.md)。
`default=None` 的数值/路径类 flag =「不覆盖 config」。

| Flag | 作用 | 覆盖的 config 字段 | 备注 |
|------|------|-------------------|------|
| `--num-agents` | 并行 agent 数 | `num_agents` | |
| `--max-iterations` | PPO 迭代轮数 | `max_iterations` | |
| `--steps-per-rollout` | 每轮每 agent 步数 | `steps_per_rollout` | 最小任务建议 300–500 |
| `--y-max-cutoff` | 候选点高度上限（米） | `y_max_cutoff` | **仅**在 `random_deploy=True` 时过滤候选池；`y ≤ cutoff + drop_height` |
| `--run-dir` | 实验归档目录（**推荐**） | `checkpoint_dir` + `log_dir` | 裸名字落到 `runs/<名字>/`；额外开启元数据 / 指标 / 控制台 / 精英池落盘 |
| `--checkpoint-dir` | 权重输出目录 | `checkpoint_dir` | 旧式，与 `--run-dir` 互斥；不产出 `metrics.csv` 等记录 |
| `--log-dir` | 日志 / 效率图输出目录 | `log_dir` | 旧式，同上 |
| `--lr` | 学习率 | `lr` | |
| `--ent-coef` | 熵奖励系数 | `ent_coef` | |
| `--init-log-std` | 探索噪声初值上限 | `ppo_init_log_std` | tanh 前 raw 空间；越小噪声越小 |
| `--log-std-max` | 探索噪声 clamp 上限 | `ppo_log_std_max` | |
| `--value-warmup-iters` | 开局只训 critic 的轮数 | `value_warmup_iters` | `0` = 禁用；从 BC 迁移时建议保留默认 |
| `--resume PATH` | 从 PPO checkpoint 恢复 | （运行时） | 与 `--resume-bc` 互斥优先：有 `--resume` 则不走 BC |
| `--resume-bc PATH` | 从 BC `ActionPredictor` 迁移 backbone | （运行时） | 只初始化，不恢复 optimizer / iteration |
| `--random-deploy` | 开启候选池随机投放 | `random_deploy=True` | 课程后期再用 |
| `--no-random-deploy` | 全部留在初始起点 | `random_deploy=False` | **当前最小任务默认应显式传这个** |
| `--no-launch` | 不启动游戏（假定已在跑） | （运行时） | |
| `--persist-game` | 游戏跨轮常驻，循环内只 `reset` | （运行时） | 省每轮启停；与 `--no-launch` 可组合 |
| `--sil-seed NPZ [NPZ ...]` | 精英池冷启动种子轨迹 | （运行时） | `.npz` 含 `raw_states/actions/contact`；入池时按 `score_trajectory` 现算分数，后续与采集轨迹一同参与 MAP-Elites 竞争 |

### 未暴露为 CLI、但默认已生效的关键项

改这些请直接改 `TrainConfig` 或后续加 flag，不要以为「没传 CLI = 没开」：

| 字段 | 默认 | 含义 |
|------|------|------|
| `ou_enabled` / `ou_phi` | `True` / `0.85` | P1.1 时序连贯探索（OU AR(1)） |
| `sil_enabled` | `True` | Self-Imitation |
| `sil_map_elites_enabled` | `True` | P1.2 行为多样性精英池 |
| `sil_cell_size` / `sil_cold_start_min` | `5.0` / `5` | MAP-Elites 网格与冷启动 |
| `sil_prune_enabled` | `True` | 入池前物理剪枝：截到"最后有效推进点"（+`dwell_steps` 观察窗），峰后跌落/震荡段不进精英池存储。推进点口径自动分派：走廊轨迹（带 `corridor_progress`）取 credited 首次封顶；其余取 secured peak。score/bd/BC 窗口口径与完整轨迹一致（`dataset.prune_trajectory`，单测 `test_prune_trajectory.py`）。效率图 update 仍用完整轨迹（峰后段是信用记录）。附带效果：仅峰后段落水的轨迹不再被整条硬拒 |
| `explore_boost_enabled` | `True` | 精英池停滞时抬 log_std |
| `pbrs_gamma` | `1.0` | 伸缩式 PBRS（勿改回 0.99 除非明确要测漂移） |

完整字段注释以 `config.py` 为准。

---

## 三、常用配方

### B. 从 BC 冷启 PPO

```bash
python -m src.training.main_ppo \
  --no-random-deploy \
  --steps-per-rollout 400 \
  --resume-bc checkpoints/model_iter_XXXX.pt \
  --run-dir ppo_from_bc \
  --persist-game
```

### C. 续训已有 PPO

续训要写回同一个 `--run-dir`，这样 `metrics.csv` 会接着原来的轮次追加，
`console.log` 也是追加而非覆盖。

```bash
python -m src.training.main_ppo \
  --no-random-deploy \
  --steps-per-rollout 400 \
  --resume runs/my_experiment/checkpoints/ppo_iter_0020.pt \
  --run-dir my_experiment \
  --persist-game
```

### D. 低处课程 + 候选池投放（基座过树后再开）

```bash
python -m src.training.main_ppo \
  --random-deploy \
  --y-max-cutoff 125 \
  --steps-per-rollout 500 \
  --resume runs/my_experiment/checkpoints/ppo_iter_XXXX.pt \
  --run-dir lowcourse \
  --persist-game
```

前置：L7 已生成候选点文件（见 `training.md` / `entrypoints.md`）。

---

## 四、废弃 / 慎用 / 勿当默认

| 项 | 状态 | 说明 |
|----|------|------|
| 文档/口令把 `--random-deploy` 当 PPO 默认用法 | **过时** | 与 roadmap P0.1 冲突。需要关投放时传 `--no-random-deploy`。`TrainConfig.random_deploy` 仍默认 `True`，不传就会开投放 |
| 训练侧重算表面线段投放（`set_surface_segments` / `compute_landable_surfaces`） | **train 路径已移除** | 现为 L7 候选池 pool-only。`_sample_positions` 内线段采样仅作「无候选池」历史回退，新实验勿依赖 |
| SIL `add_trajectories(..., keep_ratio=)` top-K% 准入 | **PPO 侧降级为回退** | 默认走 `add_trajectories_map_elites`。仅当 `sil_map_elites_enabled=False` 时回到 top-K%。BC `main_train` 仍用 top-K%，不受影响 |
| `sil_keep_ratio` | **PPO 默认路径无效** | MAP-Elites 开启时不读该字段；改它不会改变入池行为 |
| OU 关回 iid（`--recipe ppo`） | **受控对照** | 原奖励下的结论不能代替轨迹走廊下的效果比较，保留相同 seed 和样本预算 |
| `pbrs_gamma=gamma(0.99)` | **勿当默认** | 会引入 `−(1−γ)Φ` 负漂移，实测压制攀爬；保持 `1.0` |
| `--resume` 与 `--resume-bc` 同时传 | **勿混用** | 代码优先 `--resume`；同时传等于忽略 BC |

仍合法、但**现在不该当主线**的：

- `--random-deploy` + `--y-max-cutoff`：分层课程，等最小任务点火后再用
- `sil_map_elites_enabled=False`：对比实验用

---

## 五、与 `TrainConfig` / 其他文档的分工

| 文档 | 管什么 |
|------|--------|
| **本文** | CLI、推荐命令、废弃清单 |
| [`training.md`](training.md) | 数据流、观测、奖励、效率图、TCP |
| [`optimization_roadmap.md`](optimization_roadmap.md) | 优化优先级与验收信号 |
| [`entrypoints.md`](entrypoints.md) | 启动 / 投放 / 候选点生成管线 |
| `src/training/config.py` | 全部超参的真源（字段默认值与注释） |

新增「经常要在命令行拧」的旋钮时：先加 CLI + 更新本文第二节；纯算法超参只改 `config.py` 并在 `training.md` 或 roadmap 交代即可。
