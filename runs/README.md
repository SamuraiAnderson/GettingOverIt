# 实验归档

一次训练一个目录，产物全部收敛在里面。**轻量记录入库，重资产忽略**——权重和图可以
重跑或丢弃，但"做了什么实验、结果如何"必须永久留在版本库里。

## 由来

早期实验把 checkpoint 丢进 `checkpoints_<tag>/`、效率图丢进 `logs_<tag>/`，两者都在
仓库根目录、都没进版本库，指标只打在控制台里，SIL 精英池只活在内存里。跑了 13 组
实验之后：

- 训练曲线**全部丢失**（只有 2 组手工 `Tee-Object` 存了 txt）
- 每组用了什么 CLI 参数、什么 commit，只能翻终端 scrollback
- `doc/optimization_roadmap.md` 引用的核心证据图 `logs_treefix2/elite_trajectories.png`
  已经不存在
- 根目录堆了 2.9 GB 未入库文件，`git status` 被淹没

归档时从终端记录里抢救回了 1030 轮训练指标，但这属于运气好。现在由
`src/training/run_logging.py` 在训练过程中直接落盘，不再依赖抢救。

## 目录布局

```
runs/<tag>/
  checkpoints/     ppo_iter_*.pt + cache/        重 · 忽略
  logs/            ppo_effmap_iter_*.png         重 · 忽略
  elites/          elites_iter_*.npz（SIL 精英池快照）  中 · 忽略
  console.log      全量控制台输出                 重 · 忽略
  run_meta.json    CLI + config 快照 + git commit + 起止状态   轻 · 入库
  metrics.csv      每轮指标（PPO + SIL + MAP-Elites + 探索）    轻 · 入库
  RESULT.md        意图 / 命令 / 客观结果 / 结论   轻 · 入库

runs/_eval/
  eval_*.json      模型评估原始数据               重 · 忽略
  summary.csv      逐 checkpoint 的评估汇总       轻 · 入库
```

## 怎么用

新实验用 `--run-dir`，其余目录参数不用给：

```bash
python -m src.training.main_ppo --run-dir my_experiment --no-random-deploy --persist-game
```

它会自动建好上面的目录、写元数据、逐轮追加 `metrics.csv`、把控制台镜像到
`console.log`（不必再手工 `Tee-Object`），并在每次存 checkpoint 时快照精英池。
训练被 Ctrl-C 或崩溃时，`run_meta.json` 的 `status` 会落成 `interrupted` / `failed`，
不会留下一个永远 "running" 的记录。

跑完后在 `RESULT.md` 的 `## 结论` 段写下判断——这是归档里唯一机器写不出来的部分。

相关工具：

| 命令 | 用途 |
|---|---|
| `python -m src.training.run_log_parser <console.log> -o metrics.csv` | 从控制台日志反解指标（崩溃后重建、或处理老日志） |
| `python -m src.training.archive_evals --collect logs` | 归档评估结果并重建 `_eval/summary.csv` |
| `python -m src.training.migrate_legacy_runs --refresh` | 按 `LEGACY_RUNS` 表重新渲染归档索引（保留手写结论） |
| `python -m src.tests.analysis.view_elite_trajectories --checkpoint runs/<tag>/checkpoints/xxx.pt` | 可视化精英轨迹 |

## 已归档实验

按时间顺序。"指标"列是 `metrics.csv` 里的轮数。

| 实验 | 意图 | 指标 | 保留权重 | 结果 |
|---|---|---|---|---|
| `default_randomdeploy` | 随机投放 PPO，跑在默认目录 | 100 | `ppo_iter_0100.pt` | mean_reward 全程为负；该目录被多次运行覆写，是引入 `--checkpoint-dir` 的起因 |
| `bcseed` | BC 种子 + 随机投放首试 | 17 | `ppo_iter_0015.pt` | 投放姿态与基座控制两个未解问题叠加，难归因，放弃该路线 |
| `bcseed2` / `bcseed3` / `bcseed4` | 同配方重跑 | 7 / 5 / 5 | 已删 | 均在 iter5~7 中止 |
| `start1` / `start2` | P0.1 固定起点最小任务首试 | 9 / 10 | 已删 | start1 发现 dropout 未关，start2 修复后交由 start_long 长跑 |
| `start_long` | 固定起点 baseline | 60 | `ppo_iter_0060.pt` | 未出现稳定攀爬，作为 SIL 的对照组 |
| `start_sil` | 固定起点 + SIL 精英池 | 190 | `ppo_iter_0190.pt` | 190 轮后价值仍只在起点盆地，与无 SIL 对照无质变 |
| `treefix` | 地形 mask 3x3 子采样修复 | 112 | `ppo_iter_0110.pt` | mask 修复本身未改变训练形态 |
| `treefix2` | mask 3x3 + PBRS gamma=1 | 190 | `ppo_iter_0190.pt` | **瓶颈判断主要证据**：精英轨迹 top-10 全在起点横向游走，无一翻越 Deadtree |
| `treefix2cont` | treefix2 续训到 260 | 70 | `ppo_iter_0260.pt` | 再给 70 轮仍无纵向价值传播，证否"训练时长不足" |
| `treefix3` | treefix2 配方重跑 | 5 | 已删 | iter5 中止 |
| `ou_me` | P1.1 OU 噪声 + P1.2 MAP-Elites | 250 | `ppo_iter_0290.pt` | **secured_dy 停在 4.28 m、cell 数停在 11**，排除"连贯探索不足"，转向 P2.1 接触信号 |

中止的实验只保留 `run_meta.json` / `metrics.csv` / `console.log` / `RESULT.md`，权重与
效率图已删除。归档总计约 150 MB（迁移前为 2.9 GB）。

各组实验的详细参数、完整指标与结论见各自目录下的 `RESULT.md` 与 `run_meta.json`。
