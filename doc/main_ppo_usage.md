# PPO 运行手册（`main_ppo`）

本文说明 `python -m src.training.main_ppo` **怎么跑**、各 CLI 参数做什么、哪些路径应视为废弃/慎用。
算法细节、观测/奖励口径见 [`training.md`](training.md)；优化优先级见 [`optimization_roadmap.md`](optimization_roadmap.md)。

环境：统一用 conda `getting-over-it-analysis`。

新增可选的[视频参考走廊模式](reference_corridor_reward.md)：使用 `--reference-trajectory`
启用“走廊内前进优先”的奖励，`--reference-corridor-radius 1.5` 设置半宽。
该模式替换原 PPO 引导奖励和 wide 引导图；SIL 精英池在走廊模式下自动切换为
**credited 进度口径**（剪枝/排序/行为描述子/BC 截断全部基于逐帧记录的
`corridor_progress`，patch wide 通道按 `guidance_index` 重建引导段地图），
不再按高度筛选；不传时沿用本文原配方。

---

## 一、当前主线：固定起点的轨迹走廊

用户已选择轨迹走廊引导。新实验使用固定配方，隔离 OU 与其他候选机制：

```powershell
conda run --no-capture-output -n getting-over-it-analysis python -m src.training.main_ppo --reference-trajectory --recipe ppo-ou --seed 7 --geometry-profile native-pose-v1 --no-random-deploy --num-agents 2 --steps-per-rollout 400 --max-iterations 40 --run-dir corridor_seed7 --persist-game
```

这是新训练命令，不代表 40 轮保证学会任务。`--recipe ppo` 关闭 OU，`ppo-ou` 保留 OU；两者均关闭 SIL、停滞加噪、接触分头、阶段权重及 SIL 优势权重。`ppo-ou-sil` 在走廊模式下同样可用——精英池已迁移到 credited 进度口径（见上文），不再是按高度筛选的旧实现。

`--seed` 固定 Python、NumPy、Torch 和投放采样 RNG；Unity 复位仍有物理差异，应重复评估。`--geometry-profile native-pose-v1` 使用原生姿态标定（锤头 −180°、身体约 −84.498°）；`legacy` 为原来的 −90°/−90°。不传时，新模型使用新标定，旧 checkpoint 恢复其配置，缺少配置的旧权重回退 legacy。BC 入口继续使用 legacy。

`--resume` 先恢复保存的配置，再应用显式 CLI 覆盖（本机 game_root/port 除外）。更换参考奖励或几何标定会从当前迭代重新预热 critic。`--max-iterations` 是累计迭代上限：从 335 恢复并传 337，只执行两轮。`--value-warmup-iters 0` 仅在明确需要跳过预热时使用。

`--save-rollout-inputs` 将每轮采样权重和完整输入保存到 `checkpoints/rollouts/`，便于冻结重放；体积较大，通常用于验收或短诊断。采集自动批推理，`TrainConfig.ppo_inference_batch_size=1` 可用于串行对照，`0` 批处理全部活跃 agent。patch 按帧共享，minibatch 才组装窗口，CNN 更新仍重新计算。

固定评估与离线重放：

```powershell
conda run --no-capture-output -n getting-over-it-analysis python -m src.training.local_evaluation --checkpoint runs/corridor_seed7/checkpoints/ppo_iter_0040.pt --output runs/corridor_seed7/checkpoints/eval --rounds 5 --steps 400 --num-agents 2
conda run --no-capture-output -n getting-over-it-analysis python -m src.training.local_evaluation --checkpoint runs/p0_corridor_training_smoke/checkpoints/rollouts/behavior_0337.pt --replay runs/p0_corridor_training_smoke/checkpoints/rollouts/iter_0337.npz --output runs/p0_corridor_training_smoke/checkpoints/replay
```

评估分别执行无噪声与指定 OU 探索，默认要求同段有效前进 5 米、当前位置前进 5 米并在走廊内保持 10 步。`--goal-progress`、`--dwell` 可调整；`--jitter` 默认 ±0.01 米，第一轮无扰动。记录成功、跌落、离开走廊、到达步数、输入和接触/动作事件，详情见[执行记录](optimization_execution_2026-09-19.md)。加 `--native-diagnostics` 需要新版插件，会增加诊断通信开销。

### P1.3 / P2 表示候选

候选已实现并通过工程 smoke，默认保持原 Transformer/39D/逐步动作。开关、单位、序列与段概率契约及验证边界见[表示实验](policy_representation_experiments.md)。

- `--features relations events geometry trends`：选择关系、事件、局部几何和多尺度趋势；可只选一组，`none` 清空这些组。
- `--event-branch`：小事件 MLP，同时送入 backbone 和 actor/critic，并自动启用事件输入；`--event-source native|geometry` 指定事件来源，默认 native。
- `--policy-backbone transformer|gru`：单帧 CNN+GRU 对照；`--recurrent-seq-len 16` 控制训练块的决策数。
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

以下仅描述原高度奖励的历史实验。目标：固定起点、短 episode，观察是否翻过 Deadtree。**先不要开 random deploy。** 当前走廊实验使用上面的命令和成功判据。

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

### A. 轨迹走廊（当前主线）

见第一节的新命令。关闭 deploy、固定配方、配套局部评估。

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
| 文档/口令把 `--random-deploy` 当 PPO 默认用法 | **过时** | 与 roadmap P0.1 冲突。当前 sprint 应 `--no-random-deploy`。`TrainConfig.random_deploy` 仍默认 `True`，故**必须显式关**，否则会开投放 |
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
