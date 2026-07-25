# PPO 运行手册（`main_ppo`）

本文说明 `python -m src.training.main_ppo` **怎么跑**、各 CLI 参数做什么、哪些路径应视为废弃/慎用。
算法细节、观测/奖励口径见 [`training.md`](training.md)；优化优先级见 [`optimization_roadmap.md`](optimization_roadmap.md)。

环境：统一用 conda `getting-over-it-analysis`。

---

## 一、当前推荐配方（P0.1 最小任务）

目标：固定起点、短 episode，只考核是否翻过 Deadtree。**先不要开 random deploy。**

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

未传的参数一律保持 `TrainConfig` 默认值（`src/training/config.py`）。  
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

### 未暴露为 CLI、但默认已生效的关键项

改这些请直接改 `TrainConfig` 或后续加 flag，不要以为「没传 CLI = 没开」：

| 字段 | 默认 | 含义 |
|------|------|------|
| `ou_enabled` / `ou_phi` | `True` / `0.85` | P1.1 时序连贯探索（OU AR(1)） |
| `sil_enabled` | `True` | Self-Imitation |
| `sil_map_elites_enabled` | `True` | P1.2 行为多样性精英池 |
| `sil_cell_size` / `sil_cold_start_min` | `5.0` / `5` | MAP-Elites 网格与冷启动 |
| `explore_boost_enabled` | `True` | 精英池停滞时抬 log_std |
| `pbrs_gamma` | `1.0` | 伸缩式 PBRS（勿改回 0.99 除非明确要测漂移） |

完整字段注释以 `config.py` 为准。

---

## 三、常用配方

### A. 最小任务（当前主线）

见第一节。关 deploy、短 rollout。

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
| OU 关回 iid（`ou_enabled=False`） | **仅 ablation** | 诊断已表明 50Hz iid 几乎撞不到勾—拉；日常训练保持默认开 |
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
