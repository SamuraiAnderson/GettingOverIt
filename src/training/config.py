"""训练超参数配置 — 集中管理所有可调参数。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .reference_anchor import ANCHOR_POT_CALIBRATED, VERIFIED_CALIBRATION_SHA256

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
    @classmethod
    def ppo_mainline(cls) -> "TrainConfig":
        """Fresh PPO defaults; bare defaults remain the legacy BC/checkpoint contract."""
        return cls(policy_backbone="gru", normalize_action_history=True,
                   recurrent_seq_len=32, random_deploy=False,
                   contact_heads_enabled=False, sil_enabled=False,
                   explore_boost_enabled=False, phase_weighting_enabled=False,
                   sil_advantage_weighting_enabled=False,
                   gru_sil_sampling_mode="quality_dynamics_v3",
                   gru_sil_max_updates=16, gru_sil_budget_ratio=1.0)

    # ── 环境 ──
    game_root: str = field(default_factory=_default_game_root)
    num_agents: int = 10
    port: int = field(default_factory=_default_port)
    warmup_steps: int = 100
    settle_steps: int = 30
    steps_per_rollout: int = 500
    ppo_inference_batch_size: int = 0  # 0 = all active agents; 1 = serial profiling baseline

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
    state_dim: int = 33  # 协商前的默认线路宽度；实际 transport 以 HELLO 为准，禁止被旧 ckpt 覆盖
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

    # Legacy defaults for BC and checkpoints missing these fields.
    # New PPO runs use ppo_mainline(); explicit saved values always take precedence.
    policy_backbone: str = "transformer"  # transformer | gru
    relation_features: bool = False
    event_features: bool = False
    event_branch: bool = False
    geometry_features: bool = False
    trend_features: bool = False
    event_source: str = "native"  # native physics contacts | explicitly labelled geometry proxy
    normalize_action_history: bool = False
    feature_position_scale: float = 3.5
    feature_velocity_scale: float = 10.0
    feature_angular_scale: float = 10.0  # radians / second
    feature_clip: float = 4.0
    feature_step_seconds: float = 1.0 / 120.0  # verified against native fixedDeltaTime * stepFrames
    geometry_query_radius: float = 3.0
    recurrent_seq_len: int = 16  # decisions per TBPTT chunk; full prefix burn-in, no gradient
    action_segment_steps: int = 1  # >1 samples two endpoints, linearly interpolated per STEP
    segment_interrupt_contacts: bool = True

    # ── patch 尺度模式 ──
    # legacy: 原单尺度 4ch（player 中心 0.5m/px）；dualscale: 双分支（见 doc/training.md）
    #   wide  : player 中心，大范围低精度（攀爬效率图，全局引导）
    #   contact: player 中心，中高精度统一接触图（高精度地形 + 锅/body 轮廓 + 锤头轮廓，co-registered）
    patch_mode: str = "dualscale"
    wide_patch_resolution: float = 2.0    # m/px，wide 分支（32×2.0 = 64 m 窗口）
    # contact 粗图 8.64m 窗口（32×0.27，半径 4.32m）。半径由锤子几何硬上界反推，**勿缩小窗口**。
    # 细交互不靠把 CNN 加到 256²，而走 sparse_geometry（同窗连续轮廓 + 多候选接触关系）。
    # 核验：src/tests/training/test_patch_tip_coverage.py
    contact_patch_resolution: float = 0.27  # m/px，粗 contact 图（8.64m 窗口）
    wide_channels: int = 1                # [0] eff
    contact_channels: int = 3             # [1] 高精度地形, [2] 锅/body 轮廓, [3] 锤头轮廓
    # 稀疏局部几何：256 格精度的窗口只编码轮廓 token 与接触关系，不进 CNN。
    sparse_geometry: bool = True
    sparse_window_cells: int = 256
    sparse_window_meters: float = 8.64
    sparse_max_contour_tokens: int = 128
    sparse_relation_k: int = 8
    sparse_mlp_hidden: int = 32
    # 锅/body 轮廓世界朝向：player→hub 方向 + offset（body 局部 +y=躯干朝上）。
    # 2026-09-19 原生 Transform 对照标定；旧 checkpoint 的显式配置仍按原值恢复。
    body_angle_source: str = "hub"        # hub: player→hub 方向
    body_angle_offset: float = -84.49833
    body_scale: float = 1.0

    # ── 训练 ──
    lr: float = 1e-4
    epochs: int = 10
    batch_size: int = 256
    action_scale: float = 100.0
    max_grad_norm: float = 0.5
    max_grad_norm_cap: float = 5.0   # GradNormAdapter 冻结值的硬上限 (default * cap_ratio)
    # 冻结发生在进程内第 500 次更新，PPO 梯度 p95≈9 → 直接跳到 cap；Adam 二阶矩仍是按 0.5
    # 累积的，此后约千步有效步长偏大（4h 续训第 146 轮冻结后每轮只剩 1–2 次更新）。默认固定。
    max_grad_norm_adaptive: bool = False
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
    tip_angle_offset: float = -180.0    # 原生 Tip Transform 相对 pole→tip 的标定偏移
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
    # 自适应时早停轮喂入的 KL 已超过 ppo_kl_stop_factor × target，target 只降不升，
    # 2026-09 两次续训均被钉在下限 0.01、每轮仅 1–5 次更新。默认固定 target_kl。
    ppo_target_kl_adaptive: bool = False
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
    # σ：默认继承 checkpoint，不强制重设、不按轮递减。actor_log_std 仍走 PPO 梯度。
    # 历史正式 init-from 的基础 σ≈0.96–0.99（条件 σ≈0.51–0.52）。重置+退火会先砍到
    # ≈0.20 再继续降，跨批进度变差，不能单独怪退火速度。表现触发的降噪尚未实现。
    ppo_reset_log_std_on_init: bool = False
    ppo_log_std_anneal_rate: float = 0.0   # 每轮 log_std 递减量（log 空间线性；0=关闭）

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
    # 部分 player 晚开探索。默认关闭。开启后 8 个 player 为 0/200/400/600 四档，
    # 延迟期保留 floor 倍噪声，开启后再用 ramp_steps 线性升到满幅度。见 delayed_exploration.py。
    delayed_exploration_enabled: bool = False
    delayed_exploration_floor: float = 0.15
    delayed_exploration_ramp_steps: int = 50

    # Opt-in temporal noise windows: low/full/finite/suffix = 1/1/4/2 agents.
    exploration_windows_enabled: bool = False
    exploration_window_floor: float = 0.0
    exploration_window_peak: float = 1.0
    exploration_window_min_steps: int = 32
    exploration_window_max_steps: int = 480
    exploration_window_suffix_min_steps: int = 120
    exploration_window_transition_steps: int = 0
    # Legacy checkpoint field: only the old soft-loss normalization is retained.
    # It no longer imposes an action bound or rejects PPO/SIL updates.
    exploration_window_action_limit: float = 1.0
    exploration_window_anchor_scale: float | None = None  # Mouse units; None uses legacy soft-loss scale.
    exploration_window_anchor_coef: float = 1.0

    @property
    def deterministic_anchor_scale(self) -> float:
        return (self.exploration_window_action_limit if self.exploration_window_anchor_scale is None
                else self.exploration_window_anchor_scale)

    # ── 相位探索噪声（空中相放大，2026-09-24）──
    # 物理依据：空中相是弹道运动，动作只影响锤子为下一次接触的预定位——噪声不扰动
    # 当前轨迹，是"安全探索"通道；接触相是挥杆执行，其噪声是发现引擎
    # （ab3：全局降 σ 伤发现），故接触相保持基准、只放大空中相。
    # std(s) 是观测（接触特征）的确定性函数，采集/更新同经 _actor_parameters，
    # PPO 比率自洽；与 OU 条件式正交（std_t 逐步参与条件高斯）。
    # 默认关闭（opt-in）：未验证的探索侧机制不进默认配置，实验时用 --phase-noise 显式启用。
    phase_noise_enabled: bool = False
    phase_noise_airborne_log_bonus: float = 0.5   # 空中帧 log_std 加成（+0.5 ≈ σ×1.6）

    # P2.2：tip_contact 硬 gate；原 actor_mean / actor_log_std 是空中头。
    contact_heads_enabled: bool = True
    contact_gate_threshold: float = 0.5
    contact_log_std_offset: float = -0.5

    # P3.1：接触前后 K 帧保留完整进度奖励，纯空中弱化；PBRS / 落水惩罚不变。
    phase_weighting_enabled: bool = True
    phase_window: int = 7
    phase_airborne_weight: float = 0.25
    # P3.2（已接线）：SIL 窗口按优势加权。A = 折扣未来 credited（沿精英轨迹，
    # cap 后 credited≡0，无需 GAE 的 λ 折中）+ 轨迹末窗 V 自举（与 V 的无限时域
    # 口径对齐，否则 A 被系统性压负）− 当前 critic 的 V。
    # 门控为原始 SIL 的 (R−V)₊（Oh et al. 2018）：非正优势零权重、不模仿；
    # 正优势按相对大小加权（正部均值 1），cap 封顶；全池无正优势窗口时本轮跳过。
    # 损失为 raw 空间加权高斯 NLL（σ detach，不碰 log_std）。
    # 仅走廊模式生效（需 corridor_progress）；高度模式回退旧 MSE-on-mean 并告警。
    sil_advantage_weighting_enabled: bool = True
    # 【已弃用，保留仅为旧 run_meta 兼容】原"非正优势保留 floor 弱权重"语义偏离了
    # (R−V)₊ 的关键过滤（20260924 评审），现非正优势一律零权重。
    sil_advantage_floor: float = 0.1
    sil_advantage_cap: float = 5.0
    # 【已弃用，保留仅为旧 run_meta 兼容】pos_frac 门控。实测 critic 在精英窗口上
    # 系统性膨胀 ~5-25×（20260923 A/B 离线复算：V p50≈5.4m vs 真实折扣回报
    # ≈0.2-0.7m），A=R−V 恒负、pos_frac≈0，门控永久关闭。由 sil_gate_margin
    # 的行为门控（V-free）取代。
    sil_min_pos_frac: float = 0.1
    # 行为门控（V-free）：池 median 总 credited ≤ 当前轮 per-agent 平均 credited
    # × (1+margin) 时跳过本轮 SIL。同量纲（原始米、同一 tracker 口径），不依赖
    # critic 校准。池由 MAP-Elites 维持"历史最优"，门控只在池停滞/种子弱于当前
    # 行为时关闭。
    sil_gate_margin: float = 0.0
    # raw 空间模仿目标限幅（atanh 之后）：精英动作 = 策略均值 + OU 噪声，饱和动作
    # （种子轨迹 42% 满饱和）经 atanh 映射到 ±7.6，NLL 梯度 w·(μ−u)/σ² 会把均值钉进
    # tanh 饱和区（det 塌陷的直接机制）。clamp 保留方向、只限幅度；tanh(3)=0.995，
    # 动作空间 ±99.5 以内不受影响；0 = 关闭。
    sil_target_clamp: float = 3.0
    # 1.3 critic 精英回报校准：SIL 更新时对精英窗口追加 value 损失，把 frontier 区域
    # 系统性膨胀的 V（20260923 离线复算：V p50≈5.4m vs 真实折扣回报 0.2-0.7m）拉向
    # 精英轨迹的 MC 回报（末窗 V 自举口径，与优势计算一致）。小系数混合防反向高估；
    # Huber 损失压大误差的梯度。0 = 关闭。
    sil_value_coef: float = 0.1
    # P3.1-SIL（已接线）：精英轨迹学习的接触相位加权 + 空中帧稀疏采样。
    # 物理依据：GOI 空中相是弹道运动，动作与结果几乎无关（克隆目标≈纯噪声）；
    # 接触相是高杠杆决策（动作→结果信息量高）。smoke 实测 tip 接触帧仅 ~35%，
    # 等权 SIL 把 ~65% 梯度花在空中噪声上（ab3 负收益的主要机制之一）。
    # 接触帧（tip/body/pot 任一）前后 K 帧权重 1.0（含"即将接触"前瞻窗口）；
    # 纯空中帧权重 airborne_weight，且训练输入时按 airborne_sample 概率稀疏保留
    # （损失降权 × 采样稀疏 双重削弱）。与优势权重（P3.2）相乘合成。
    sil_phase_weighting_enabled: bool = True
    sil_phase_window: int = 7
    sil_phase_airborne_weight: float = 0.25
    sil_phase_airborne_sample: float = 0.25   # 空中帧稀疏保留率（1=不稀疏，0=全丢）

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
    # Stateful SIL uses complete reset-to-end archives, independently of legacy windows.
    gru_sil_enabled: bool = False
    # CPU observations only; 0 retains round-local caching. Entries are not checkpointed.
    gru_sil_cache_mb: int = 1024
    gru_sil_max_updates: int = 4       # historical fallback; fresh PPO uses 16
    gru_sil_budget_ratio: float = 0.25 # historical fallback; fresh PPO uses 1.0
    gru_sil_loss_coef: float = 0.1
    gru_sil_value_coef: float = 0.05
    gru_sil_target_kl: float = 0.01
    gru_sil_max_frame_kl: float = 0.1
    gru_sil_seed: int = 0
    gru_sil_sampling_mode: str = "trajectory_uniform"  # legacy fallback; fresh PPO uses quality/dynamics weighting
    gru_sil_phase_rules: dict | None = None  # versioned geometric proxy rules; no native branch claim
    recovery_archive_enabled: bool = False  # explicit opt-in; old checkpoints remain elite-only
    gru_sil_recovery_fraction: float = 0.25
    recovery_min_forward_m: float = 2.0
    recovery_start_bin_m: float = 1.0
    recovery_delta_bin_m: float = 1.0
    training_seed: int = 0  # Initial stream seed; resume restores RNG positions separately.
    sil_start_iter: int = 5             # 对齐 value_warmup_iters：critic 预热完再开 SIL
    sil_traj_cap: int = 20             # 精英池容量。map_elites 仍可按分数兜底；td_pareto 不按 score 截断
    sil_keep_ratio: float = 0.3        # 每轮新轨迹入池的 top-K 比例
    sil_batch_size: int = 64
    sil_epochs: int = 1                # 每次 PPO 迭代对精英池遍历轮数
    sil_loss_coef: float = 0.5         # MSE-on-mean 损失权重（温和，防过度模仿压制探索）
    sil_max_batches: int = 40          # 每次 SIL 更新最多 minibatch 数（标准 SIL 采样式，0=遍历整池）
    # SIL 梯度预算从属于 PPO：本轮 SIL minibatch 上限 = ratio × 当轮 PPO 实际更新数
    # （仍受 sil_max_batches 约束）。20260923 塌陷期 PPO 被 KL 早停到每轮 2-10 次而
    # SIL 固定 40 次，策略实质退化为纯 BC；PPO 饥饿时 SIL 同步停手，等 KL 预算恢复。
    # ratio < 0 = 不约束。
    sil_budget_ratio: float = 1.0
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
    # SIL 归档：map_elites 为对照；td_pareto 是终点 (t,D) 非支配前沿，不是唯一示范来源。
    # 前缀里程碑归档尚未实现。默认 map_elites，避免正式对照被改口径。
    sil_archive_mode: str = "map_elites"
    sil_td_bucket_m: float = 0.10       # 接近容量后才启用的 D 分桶宽度（米，待验证）
    sil_td_bucket_enable_at: int = 16   # 前沿条数达到该值才分桶；池小时关闭
    sil_td_behavior_slots: int = 3      # 前沿外另留的空间多样性条数
    # legacy_min_length：0=对齐到最短轨迹。balanced_stratified / coverage_share：
    # 正数是总计算预算的平均规模 K（不是每条实际固定窗口数），0 在新模式解析为 32。
    sil_td_windows_per_traj: int = 0
    # legacy_min_length 保持原采样与单一 D 门控。两个新模式使用完整 0..cap 候选，
    # 并关闭单一 D 门控；coverage_share 再按 (t, D) 覆盖面积分摊轨迹配额。
    sil_td_sampling_mode: str = "legacy_min_length"
    # 0 = 新模式创建时从 steps_per_rollout 解析并保存。恢复只用保存值，不随池内最大 t 改变。
    sil_td_sampling_horizon_steps: int = 0
    # 0 = 从当次训练 seed 解析。恢复使用已保存的采样种子。
    sil_td_sampling_seed: int = 0
    sil_td_sampling_rule_version: str = "coverage_v1"
    # 入池前物理剪枝：在 secured peak（+dwell 观察窗）处截断，去除峰后跌落/来回震荡段，
    # 精英池只存"有效推进 + 守住证据"，BC 窗口/score/bd 口径与完整轨迹一致（见 dataset.prune_trajectory）。
    sil_prune_enabled: bool = True

    # ── 非对称奖励缩放 ──
    # 0.1→0.0：速通口径「secured 进展」下跌落不再显式惩罚（时间隐式受罚：掉下去要重爬 → 步数变多 →
    # climb_speed 变差）。防甩飞改由确认式创新高（NewHighConfirmer，dwell 守住才结算）承担，而非扣分。
    neg_reward_scale: float = 0.0     # log 压缩系数: -scale * log(1 + |dy|)（置 0 = 不惩罚回落）
    # step_reward 归一化模式下 progress 通道 vs PBRS 势场塑形通道的无量纲比例
    # (α→progress, 1-α→PBRS F=γΦ'−Φ)。σ 冻结后 α、σ_h/σ_e 皆常数 → 等价「主奖励+常数·PBRS」，
    # 最优策略不变性成立（常数倍势场仍合法、全局缩放不改 argmax）。
    reward_alpha: float = 0.5

    # ── 视频参考走廊（显式启用；替代 PPO 高度/向右/效率图奖励）──
    reference_trajectory: str | None = None
    reference_corridor_radius: float = 1.5  # 半宽，世界米；走廊内不惩罚偏离中心线
    reference_guidance_sigma: float | None = None  # observation only; None preserves historical radius
    reference_progress_weight: float = 1.0  # 走廊内新增路线进度，每米奖励
    reference_deviation_weight: float = 0.01  # 走廊外每步弱惩罚的上限
    reference_projection_window: float = 4.0  # 局部投影搜索半径，沿线米
    reference_max_step: float = 4.0  # 单步位移超限视为跳变，仅重定位不领奖
    reference_simplify_tolerance: float = 0.15  # 折线简化容差，保留分段缺口
    reference_interpolated_weight: float = 0.5
    # 距离口径。新建走廊任务默认 confirmed_endpoint_v2。
    # 旧 checkpoint 没有该字段时必须识别为 legacy_credit_v1，不能靠这个默认值静默改口径。
    reference_progress_mode: str = "confirmed_endpoint_v2"
    # 路线比较点。默认已验证的实机可见锅心；旧 player_origin 实验须显式改回。
    reference_anchor_id: str = ANCHOR_POT_CALIBRATED
    reference_anchor_verified: bool = True
    reference_anchor_calibration_sha256: str | None = VERIFIED_CALIBRATION_SHA256
    reference_anchor_source_kind: str = "native"
    # 线路协议。默认 HELLO 进入 anchor35；必须协商成功，禁止自动降级 legacy33。
    tcp_protocol: str = "anchor35"

    # ── 训练控制 ──
    # Archive complete episodes independently of SIL; never feeds PPO updates.
    elite_archive_enabled: bool = False
    elite_archive_capacity: int = 20
    max_iterations: int = 100
    save_interval: int = 5
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
