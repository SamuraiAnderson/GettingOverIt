"""训练超参数配置 — 集中管理所有可调参数。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "project.json"


def _default_game_root() -> str:
    try:
        with open(_PROJECT_CONFIG, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return str(Path(cfg["game"]["executable_path"]).parent)
    except Exception:
        return ""


def _default_port() -> int:
    try:
        with open(_PROJECT_CONFIG, "r", encoding="utf-8") as f:
            return json.load(f).get("tcp_port", 9000)
    except Exception:
        return 9000


@dataclass
class TrainConfig:
    # ── 环境 ──
    game_root: str = field(default_factory=_default_game_root)
    num_agents: int = 10
    port: int = field(default_factory=_default_port)
    warmup_steps: int = 100
    settle_steps: int = 30
    steps_per_rollout: int = 500

    # ── 模型 ──
    # 原始状态 33 维 = 基础 29 + fakeCursor 4（cursorX/Y, cursorVelX/Y，索引 29-32）。
    # 注意：state_dim 是 C# 回传的**原始状态维度**，与模型消费的 dynamics 特征维**解耦**。
    # build_dynamics 输出维度由 dataset.DYNAMICS_DIM 决定 = 15 速度/角速度(含 cursor 速度)
    # + 3 角度×2(sin/cos) + 6 部件相对坐标×2(含 cursor) + 1 绝对高度 = 34 (BASE_DYNAMICS_DIM)。
    # 另 + 5 维接触信号（contact_features.CONTACT_DIM = tip_contact / tip_grip /
    # body_contact / pot_contact / fall_distance_norm，复刻游戏原生 HammerCollisions +
    # PlayerSounds；body / pot 按游戏 IL 里 `rb.GetPoint().y > 0.4` 分层各自输出）
    # → DYNAMICS_DIM = 39。模型 dynamics_proj / 归一化 buffer 用 DYNAMICS_DIM，勿再用
    # state_dim 冒充特征维。
    state_dim: int = 33
    action_dim: int = 2
    # 归一化后动力学观测裁剪到 [-dynamics_clip, +dynamics_clip]，防止物理爆炸速度
    # 或微方差维度（如 hub 相对坐标 std≈0.008）放大出的极端值在 Transformer 内溢出成 NaN。
    dynamics_clip: float = 10.0
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    context_len: int = 32
    patch_size: int = 32
    patch_channels: int = 4          # dualscale: wide_channels + contact_channels
    patch_resolution: float = 0.5    # 仅 legacy 单尺度使用
    dropout: float = 0.1

    # ── patch 尺度模式 ──
    # legacy: 原单尺度 4ch（player 中心 0.5m/px）；dualscale: 双分支（见 doc/training.md）
    #   wide  : player 中心，大范围低精度（攀爬效率图，全局引导）
    #   contact: player 中心，中高精度统一接触图（高精度地形 + 锅/body 轮廓 + 锤头轮廓，co-registered）
    patch_mode: str = "dualscale"
    wide_patch_resolution: float = 2.0    # m/px，wide 分支（64m 窗口）
    # contact 分支 8.64m 窗口（半径 4.32m）。该半径由锤子几何硬上界反推，**勿随意下调**：
    #   |hub−player| = 0.3546m（刚性偏移）+ slider 全伸 |tip−hub| = 2.955m（关节硬限位）
    #   = tip 原点上界 3.31m；PlayerControl IL 里另有 |cursor−player| ≤ 3.5f 硬编码钳制，
    #   与求解器软性超调后的实测极值(≈3.49m)吻合，故按 3.5m 取设计上界。
    #   再加 tip 轮廓半径 0.279m（player_contour.json，16 顶点）→ 轮廓需 3.78m；
    #   再留 2 格地形上下文（锤头全伸时也要能看到它接触的那块地形）→ 需 ≈4.28m。
    # 旧值 0.2（6.4m 窗口 / 半径 3.2m）不足：约 0.46% 的帧里锤头贴窗边、其接触的地形被裁掉，
    # 而这些恰是「锤子伸到最远够远处落点」的高价值帧。
    # 改分辨率而非 patch_size，是为了保持 patch shape (4,32,32) 不变 → checkpoint 完全兼容、
    # wide 分支不受影响（否则 64m 窗口会被迫跟着变）、CNN 的 Linear(2048) 无需迁移。
    # 分辨率变粗的代价已被 PART_RASTER_SUBSAMPLES 的面积覆盖率栅格化抵消（见 dataset.py）：
    # 部件轮廓不再因亚像素宽而漏采，0.27m/px 下 ch3 平均非零像素反而比旧 0.2m/px 二值高约 7×。
    # 精确接触判定另由 P2.1 的向量接触信号承担（走精确几何，不受栅格分辨率限制）。
    # 核验：python -m src.tests.analysis.check_tip_patch_coverage
    #      python -m src.tests.analysis.diagnose_tip_rasterization
    contact_patch_resolution: float = 0.27  # m/px，contact 分支（8.64m 窗口，半径 4.32m）
    wide_channels: int = 1                # [0] eff
    contact_channels: int = 3             # [1] 高精度地形, [2] 锅/body 轮廓, [3] 锤头轮廓
    # 锅/body 轮廓世界朝向：player→hub 方向 + offset（body 局部 +y=躯干朝上）。
    # NOTE: offset=-90 为几何推断的粗标定，精确值待 verify_body_reconstruction sweep / C# 真值。
    body_angle_source: str = "hub"        # hub: player→hub 方向
    body_angle_offset: float = -90.0
    body_scale: float = 1.0

    # ── 训练 ──
    lr: float = 1e-4
    epochs: int = 10
    batch_size: int = 256
    action_scale: float = 100.0
    max_grad_norm: float = 0.5
    max_grad_norm_cap: float = 5.0   # GradNormAdapter 冻结值的硬上限 (default * cap_ratio)
    weight_decay: float = 1e-4
    val_ratio: float = 0.1
    early_stop_patience: int = 3
    num_workers: int = 4

    # ── 数据管理 ──
    max_good_trajectories: int = 300
    warmup_rollouts: int = 50
    keep_ratio: float = 0.5
    rollouts_per_iteration: int = 3

    # ── 评分 ──
    # climb_score 量纲自洽：summit[米] + efficiency_weight * (summit/peak_step * speed_ref_steps)[米]。
    # efficiency_weight 为无量纲相对权重(O(1))；speed_ref_steps 是把「米/步」换算回「米」的参考步数。
    # 默认 1.0 * 50 = 50，与历史 efficiency_weight=50 的数值行为完全一致，仅令量纲/语义清晰。
    efficiency_weight: float = 1.0
    speed_ref_steps: int = 50
    # waypoint_weight 现有两处用途（均非 BC 排序）：
    #   1. 效率图 update() 的跨迭代价值传播系数（fs = climb_score + waypoint_weight·(E_prev(peak)-E_prev(start))）；
    #   2. PPO step_reward 无 normalizer 回退路径里 PBRS 塑形项的常数系数。
    # 有 normalizer 时 PPO 塑形强度由 reward_alpha 掌控，与此值无关。
    waypoint_weight: float = 0.5
    # BC 轨迹排序里的 waypoint 贡献（score_trajectory）。效率图是 value-to-go，端点差 E(peak)-E(start)
    # 沿真实攀爬多为负，会把好轨迹排到后面；且 PBRS 是 RL 回报口径的塑形、不适用于监督排序。
    # 故默认 0.0：BC 纯按 base_score（secured progress + 速度 + 绝对高度 tie-break）排序，
    # 效率图对 BC 的影响只经 patch wide 通道输入，不再进评分。见 reward.score_trajectory。
    bc_waypoint_weight: float = 0.0
    # 绝对高度加成（轨迹级/排序）：base_score 每条轨迹加一次，用于 BC 跨轨迹筛选。
    # 与效率图的 height_prior_weight（网格级/空间价值）分工，勿混用，见 reward.climb_score。
    # 0.1→0.02：有效性分析 D 显示 0.1 时早期 bonus/climb≈1.67，BC top-K 退化为「按投放绝对高度排序」；
    # 下调到 0.02 使早期比值≈0.33，让相对攀爬重新主导筛选（绝对高度仅作轻微 tie-break）。
    abs_height_weight: float = 0.02

    # ── 攀升门控的进度势（progress_metric，取代单一 summit）──
    # progress = secured_dy + progress_wx * max(0, reach)，仅当 secured_dy > 0（不爬升不计横向、
    # 向左不计）。secured_dy 由 dwell 滑窗过滤甩飞尖峰后的「守住的净爬升」，reach 为到达 secured
    # 峰时的向右伸展（x_peak - x_start）。用于让「右侧真实攀爬路径」压过「爬树死路」。
    progress_wx: float = 0.5        # 横向 reach 相对纵向的门控权重（>0 即可修树，量级由标定微调）
    # dwell 驻留：一个高度需在 dwell_steps 步窗口内守住（窗口内跨度 ≤ dwell_drop_tol）才算 secured。
    # 滤除「甩到顶又滑回」的瞬时尖峰，使评分/截断/PPO 创新高都只认「守得住」的进展。
    dwell_steps: int = 15
    dwell_drop_tol: float = 1.0      # 窗口内允许的回落容差（米）：跨度≤此值则按窗口最高点计

    # ── 效率图 ──
    grid_resolution: float = 1.0
    # 50→80 / 0.2→0.3：有效性分析 A 显示 r*≈9m 仅覆盖 56% 上升对，44% 超引导半径；
    # 小幅增大迭代/alpha 以覆盖更多尾部（仍受几何断裂限制，远距连通靠轨迹价值回传 + deploy 课程）。
    diffusion_iterations: int = 80
    diffusion_alpha: float = 0.3
    # 绝对高度先验（网格级/空间价值）：给每个可通行格子按 world_y 注入 floor。
    # 与 base_score 的 abs_height_weight（轨迹级/排序）分工，勿混用，见 reward.climb_score。
    # 0.02→0.0：height_prior 无条件奖励「格子越高越值钱」，会给死路树顶白送 floor，与「攀升门控
    # 进度势」冲突（死路应因未来无处可去而贬值）。置 0 后格值完全由 secured-progress 沿轨迹回传决定；
    # 远距连通改由轨迹价值回传（reach≈120m）+ random_deploy 分层课程承担，而非绝对高度 floor。
    height_prior_weight: float = 0.0

    # ── 投放模式 ──
    random_deploy: bool = True     # True=随机投放到表面, False=所有 agent 留在初始位置
    drop_height: float = 2.0        # 表面上方投放高度
    surface_padding: float = 0.3    # Player 包围盒高度之上的间隙
    y_max_cutoff: float = 380.0     # 屏蔽高于此值的表面
    settle_drop_threshold: float = 5.0  # settle 后跌幅 > 此值的 agent 视为不稳定并跳过
    # settle 后锤头嵌进障碍物的 agent 视为无效投放并跳过（与 L7 离线筛选同一判据）
    deploy_tip_obstacle_check: bool = True
    tip_angle_source: str = "pole"      # pole: pole→tip；hub: hammerAngle
    tip_angle_offset: float = -90.0     # 锤杆朝向到锤头局部系的偏移（度）
    tip_scale: float = 1.0              # 锤头局部多边形缩放
    # 人工涂抹排除区：落点若落入 deploy_exclusion_zones.json 的圆内则剔除（离线+运行时同判据）
    deploy_exclusion_check: bool = True

    # ── 安全 ──
    water_y_threshold: float = -10.0  # y 低于此值视为落水，轨迹废弃（坐标阈值）
    water_penalty: float = -10.0      # 落水时 step_reward 覆盖的惩罚值（奖励量级，与阈值语义无关）

    # ── 探索 ──
    explore_noise_std: float = 0.1
    noise_decay: float = 0.95

    # ── PPO ──
    gamma: float = 0.99
    gae_lambda: float = 0.95
    # PBRS 塑形折扣：F = pbrs_gamma·Φ(s') − Φ(s)。默认 1.0 = 伸缩式（telescoping），
    # 沿轨迹求和 = Φ_end − Φ_start，纯进度信号、无漂移。用 gamma(0.99) 会带 −(1−γ)Φ 的
    # 每步负漂移：Φ 越大惩罚越重、静止即被罚，实测主导奖励并拖低 mean_reward、压制攀爬。
    pbrs_gamma: float = 1.0
    clip_epsilon: float = 0.2
    ppo_epochs: int = 4
    vf_coef: float = 0.25
    ent_coef: float = 0.001
    target_kl: float | None = 0.03
    ppo_batch_size: int = 64
    normalize_advantages: bool = True

    # ── PPO 探索噪声（tanh 前 raw 空间的 log_std）──
    # 动作 a = tanh(u)·action_scale, u~N(mean, exp(log_std))。中性动作(u≈0)处动作空间噪声
    # ≈ action_scale·exp(log_std)。旧默认由 BC 迁移标定为 log(output_scale)≈-0.65 → std≈0.52
    # → 中性处 ≈ ±45~52 单位，对精细连贯挥杆过大，会在 PPO 早期冲坏 BC 开局（实测 PPO-5 初始点落水）。
    # 下调初值 + 收紧 clamp：std≈0.20 → 中性噪声 ≈ ±20 单位（约 BC 参考 ±10 的 2 倍，够探索不失控）。
    ppo_init_log_std: float = -1.6      # BC 迁移/初始化时的 log_std 上限（std≈0.20 → 中性 ≈ ±20）
    ppo_log_std_min: float = -2.5       # forward clamp 下限（std≈0.082 → ±8，保底探索）
    ppo_log_std_max: float = 0.0        # forward clamp 上限（std≈1.0，封顶防探索过大）

    # ── PPO 探索噪声：时序连贯 OU / AR(1)（P1.1，roadmap 见 doc/optimization_roadmap.md）──
    # 从每帧 iid 高斯改为 AR(1)：z_t = φ·z_{t-1} + √(1-φ²)·η_t，η~N(0,1)，marginal z_t~N(0,1)。
    # 采样：raw = mean + std·z_t（tanh 前）。目标动作住在 4Hz 以下低频薄流形（见 hammer_physics /
    # optimization_roadmap P1.1），iid 白噪声几乎在流形外，OU 把 lag-1 自相关抬到 ~φ，功率
    # 谱下移到低频，让探索命中概率显著提高。log_prob 按**条件高斯**写出，new/old 用相同条件式，
    # PPO ratio 数学自洽（无近似）：
    #   raw_t | z_{t-1} ~ N(mean + std·φ·z_{t-1}, std²·(1-φ²))
    # 为此在 PPORolloutBuffer 每步额外存 noise_prev = z_{t-1}（(2,)）。ou_enabled=False 退化为
    # 原 iid 采样。φ=0.85 → lag-1 ≈0.85（贴近 pink 噪声 ~0.79），先从 OU 起（log_prob 好推）。
    ou_enabled: bool = True
    ou_phi: float = 0.85

    # ── 精英池停滞 → 自适应提方差（打破"贴地到树根"局部最优的发现瓶颈）──
    # 当 SIL 精英池"守得住的竖直高度"峰值连续 patience 轮无提升，临时给采样 log_std 叠 boost
    # （同时软化 SIL），制造额外探索；一旦 pool 峰值有实质提升就回落 boost，交回利用。boost 只在
    # forward 里叠加（采集+当轮更新一致，PPO 比率自洽），非可学习参数、不入 state_dict。
    explore_boost_enabled: bool = True
    explore_stall_patience: int = 8     # 竖直峰值无提升多少轮后抬 boost
    explore_stall_eps: float = 0.5      # secured_dy 提升超过该米数才算"有进展"
    explore_boost_step: float = 0.2     # 每次抬升/回落的 log_std 增量
    explore_boost_max: float = 0.8      # boost 上限（base≈-1.6 → std ×≈2.2，仍 < ppo_log_std_max）
    explore_sil_relax: float = 0.5      # boost 满时 SIL 损失系数 ×(1-0.5)=0.5，避免被 SIL 拽回

    # ── PPO 稳定性（防 BC 迁移后早期灾难性遗忘）──
    # 现象：从 BC 种子迁移后，critic_head 是随机初始化 → 优势值来自随机价值函数 → 第 1 轮 PPO
    # 更新就把均值策略推爆（实测 iter1 approx_kl=4.3、clip_fraction=0.91，开局从 +5.8m 退化到 +0.5m）。
    # 修复：① 先冻结 backbone+actor 只预热 critic 若干轮，让优势值有意义；② KL 早停改为逐 minibatch。
    value_warmup_iters: int = 5         # 开局冻结策略、只训 critic 的迭代数（仅对全新起步生效）
    value_warmup_epochs: int = 8        # critic 预热每轮的 epoch 数（只训 critic_head，开销小）
    ppo_kl_stop_factor: float = 1.5     # 逐 minibatch KL 早停阈值 = factor × target_kl

    # ── Self-Imitation Learning (SIL)：精英轨迹 BC 自模仿 ──
    # 问题：PPO 逐 minibatch KL 早停后每轮仅用 4/78 个 minibatch，96% 采集数据被丢弃，历史最优
    # 轨迹（如"过树好开局"）无法被复用/记住。修复：持久保存自身历史最优轨迹，每轮 PPO 更新后额外
    # 做几步 BC 监督（MSE-on-mean），把 actor 均值锚向精英动作。只作用 mean 不碰 log_std → 不压制探索。
    sil_enabled: bool = True
    sil_start_iter: int = 5             # 对齐 value_warmup_iters：critic 预热完再开 SIL
    sil_traj_cap: int = 20             # 精英池按分数保留的轨迹上限（非 FIFO，历史最优不淘汰）
    sil_keep_ratio: float = 0.3        # 每轮新轨迹入池的 top-K 比例
    sil_batch_size: int = 64
    sil_epochs: int = 1                # 每次 PPO 迭代对精英池遍历轮数
    sil_loss_coef: float = 0.5         # MSE-on-mean 损失权重（温和，防过度模仿压制探索）
    sil_max_batches: int = 40          # 每次 SIL 更新最多 minibatch 数（标准 SIL 采样式，0=遍历整池）
    sil_num_workers: int = 4           # SIL DataLoader 并行建 patch 的 worker 数（0=主进程串行）

    # ── SIL 精英池：MAP-Elites 行为多样性准入（P1.2，roadmap 见 doc/optimization_roadmap.md）──
    # 取代原 top-K% 相对门槛。行为描述子 bd = secured peak (x, y)（复用 progress_metric，
    # 已被 dwell 过滤，不会被瞬时尖峰/落水回落污染），离散化到 sil_cell_size m 网格；
    # 每 cell 只保留分数最高一条；is_water_trajectory 直接硬拒；|pool| < sil_cold_start_min
    # 时任何合法轨迹入池，避免早期全挤在同一 cell 塌缩到 1 条。
    # 相较 top-K%：绝对准入门槛 + 强制行为覆盖，防止"整批都烂时仍强化最不烂的乱晃"；
    # cell 数量随迭代增长即为探索健康信号（验收指标）。
    sil_map_elites_enabled: bool = True
    sil_cell_size: float = 5.0          # 网格边长（米），(peak_x, peak_y) 离散化格
    sil_cold_start_min: int = 5         # 池小于此数时任何合法轨迹入池

    # ── 非对称奖励缩放 ──
    # 0.1→0.0：速通口径「secured 进展」下跌落不再显式惩罚（时间隐式受罚：掉下去要重爬 → 步数变多 →
    # climb_speed 变差）。防甩飞改由确认式创新高（NewHighConfirmer，dwell 守住才结算）承担，而非扣分。
    neg_reward_scale: float = 0.0     # log 压缩系数: -scale * log(1 + |dy|)（置 0 = 不惩罚回落）
    # step_reward 归一化模式下 progress 通道 vs PBRS 势场塑形通道的无量纲比例
    # (α→progress, 1-α→PBRS F=γΦ'−Φ)。σ 冻结后 α、σ_h/σ_e 皆常数 → 等价「主奖励+常数·PBRS」，
    # 最优策略不变性成立（常数倍势场仍合法、全局缩放不改 argmax）。
    reward_alpha: float = 0.5

    # ── 训练控制 ──
    max_iterations: int = 100
    save_interval: int = 5
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
