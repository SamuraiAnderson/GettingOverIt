# 视频参考走廊奖励

PPO 默认可读取 `src/Data/video_reference/37258923076-1-192/reference_trajectory_game_world.json`（游戏世界锅心，出生点锁到 `TransformPoint(b)`；旧地形拟合稿与 live-camera 中间稿仍保留），
把粗略视频轨迹作为有宽度、有顺序的空间路线。运行时使用游戏直接返回的位置，
不解码视频、不运行 CV，也不要求追上视频时间。它是路线奖励，不是鼠标动作模仿。

## 启用

`--reference-trajectory` 不带路径时使用游戏世界锅心那份 JSON；也可传另一个同格式文件。续训用 `main_ppo --resume`，见 [`training.md`](training.md) 第九节。
半宽默认 1.5 世界米，可用 `--reference-corridor-radius` 调整。该宽度是起始实验参数，
不是已由游戏实验确认的最优值。未指定参考轨迹的旧命令仍使用原奖励。

走廊模式默认固定起点；如显式传 `--random-deploy`，部署后会在最近参考段建立新的进度基线，
不奖励部署位置。当前路线有缺口，初期推荐固定起点。

## 奖励优先级

- 走廊内：只奖励超过本回合已达到进度的新路段。默认每米 `+1`；
  插值来源经过的边权重为 `0.5`。不处罚走廊内部偏离中心线。
- 走廊外：不给路线进度奖励，按超出宽度的距离给予弱惩罚，默认每步最多 `-0.01`。
  从外部返回的第一步也不补发进度奖励。走廊内停留、横移、往返不产生正奖励。
- 落水：仍使用原来的 `water_penalty` 覆盖当步奖励，并停止为该 agent 采集本回合的样本。
- 不分别归一化路线进度和偏离惩罚，避免把稀少的小偏差放大成主导奖励。

这是偏好参考路线的启发式任务奖励，不声称具有 PBRS 的最优策略不变性；
也不构成任意时长轨迹之间的严格字典序约束。长时间偏离仍会累计惩罚。

启用时，旧的高度创新高、只奖励向右、学习到的效率图奖励不参与 PPO 奖励。
效率图停止更新。新建走廊任务的精英池按下一节的 v2：距离是 D，剪枝和排序用 cap 与 D。
旧 checkpoint 没有 `reference_progress_mode` 时仍按 legacy credited。
patch wide 通道按 `guidance_index` 从预栅格化引导场重建，与在线采集逐点一致。
非走廊轨迹仍走高度口径。BC 评分函数本身不改动。

## 几何和连续性

加载时按 0.15m 容差简化折线，保留顺序及连续段边界。当前 JSON 的 628 个点简化后为
101 个点、6 条有效连续线段；另一个孤立检测点不能构成线，予以忽略。
折线的累计弧长预先计算。每步只在当前进度附近的弧长窗口内投影，同时受真实位移限制，
避免交叉处跳到后面的分支。单步位移超过 4m 时只重定位，不发进度奖励。

每个 agent 独立记录当前段、投影进度、已达到的最高进度；reset 后重新建立基线。
走廊外推进也会更新该基线但不领奖，因此绕路后返回不能补领跳过的进度。

到达一个连续段的终点区域（终点周围一个走廊半宽）后进入缺口状态，未走完的末尾不补奖。
缺口不连线、不发路线奖励，也不处罚距离未知路径的偏差。只有进入下一段的起点区域才
继续跟踪下一段。最后一个已知段结束不等于通关，之后没有参考奖励。
局部投影和半宽用于粗略引导，不是额外的碰撞检测器。

## 模型输入和恢复

当前连续段预先栅格化为路线引导场，复用现有 wide 通道；缺口中显示下一段。
沿线值随进度增加，离线逐渐衰减。其余地形/身体/锤头通道保持原构建方式，
张量仍为 `(4,32,32)`。每个 PPO 样本保存当时的输入，patch 按帧共享并按索引重建窗口；更新使用相同数值并重新计算 CNN；
bootstrap 也调用同一构建函数。SIL 精英池自 2026-09-23 起支持走廊模式：轨迹逐帧记录
引导段索引（`guidance_index`），重建时按索引从预栅格化引导场选图，与在线采集逐点一致
（单测 `test_prune_trajectory.py` 的 patch 一致性用例）。无需改变状态协议和模型层尺寸，
但引导通道语义已经改变，需要训练适应。

checkpoint 保存走廊配置、文件 SHA256 和 critic 预热结束迭代。
`--resume` 自动恢复走廊设置，若原文件内容改变会拒绝静默恢复；显式指定
`--reference-trajectory PATH` 可切换参考。`--no-reference-trajectory` 可切回原奖励。
从旧奖励切到路线奖励或改变路线配置时，恢复训练会重新预热 critic。

2026-09-19 起，PPO 保存 tanh 前 raw action、OU 条件及采样配置；更新前检查冻结策略概率一致性。新训练采用经过原生姿态对照的几何偏移，旧 checkpoint 保留旧值；显式 `--geometry-profile native-pose-v1` 才迁移已保存的策略，并重新预热 critic。形状兼容不代表输入语义相同。

局部评估入口为 `python -m src.training.local_evaluation`，与训练共用采集、接触及 patch 管线；默认考核连续段内 5 米有效进度和 10 步保持，不把参考缺口当作成功。完整命令、归档及实测限制见[运行手册](main_ppo_usage.md)和[执行记录](optimization_execution_2026-09-19.md)。

## 验证和观察

```powershell
conda run --no-capture-output -n getting-over-it-analysis python -m unittest src.tests.training.test_reference_corridor -v
```

覆盖内部推进优先、内部无中心线偏置、往返/静止不刷分、离开后返回不补奖、跳变、
交叉点、缺口、多 agent 隔离、reset、落水终止、共享输入构建、旧奖励回归和恢复配置。
包含模拟环境的实际 `collect_ppo` 调用，不启动游戏。

运行日志和 `metrics.csv` 记录：`corridor_progress_m`（置信权重后的新增进度总和）、
`corridor_deviation_penalty`、`corridor_inside_fraction`、`corridor_gap_fraction`。
这些是路线奖励诊断；不能替代实际通关率、攀爬结果和轨迹检查。

## 确认终点 `confirmed_endpoint_v2`

新建走廊任务的 `reference_progress_mode` 默认是 `confirmed_endpoint_v2`。旧 checkpoint 没有这个字段时识别为 `legacy_credit_v1`，恢复时不会被新默认值改掉。

v2 的距离 D 是最后一个有效入账动作之后、锅心在参考路线上的弧长（扣除本回合起点）。入账条件仍是原来的双端在走廊内、刷新历史最高投影、非跳变、置信权重大于 0。置信权重不再缩小 D。走廊外走过的已知弧长会在下一次入账时补进 D，不补未知缺口的直线距离。

PPO 每步奖励是 `reference_progress_weight * delta_D - 偏离惩罚`。落水仍覆盖当步奖励。位移超过 `reference_max_step` 的跳变使该 agent 整条轨迹失效，不把跳过的距离留到后面补奖。

`corridor_progress_m` 在 v2 里是各合法 agent 的 D 之和，不是旧 credit 之和。评估主指标是 `cap_progress_m`；`credited_progress_m` 保留为旧账诊断。

训练默认读的 `reference_trajectory_game_world.json` 已把出生点锁到 `TransformPoint(b)`。
旧地形拟合稿 `reference_trajectory.json` 仍带 `training_ready: false`，锅的可见中心没有标成 Player 原点。
宽走廊允许小误差。训练效果和宽度仍需游戏内实验检验。
