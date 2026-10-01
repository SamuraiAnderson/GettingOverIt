# 训练库

这里是会被训练和评估调用的实现，加上三个入口。一次性战役脚本不在这里。

- `main_ppo.py` — PPO 训练入口：采集、更新、存档。
- `main_train.py` — 旧的行为克隆入口：冷启动采集、打分、微调。
- `local_evaluation.py` — 按训练同一条观测管线做固定走廊评估，不更新权重。
- `config.py` — 集中保存训练可调参数。
- `experiment_options.py` — 解析实验用 CLI，并规定换网络或换输入时如何迁移 checkpoint。
- `actor_critic.py` — PPO 的策略与价值网络。
- `experimental_policy.py` — 表征、GRU 和动作段的实验网络，checkpoint 语义与正式策略分开。
- `model.py` — 行为克隆用的动作预测网络。
- `ppo_trainer.py` — 对一批 rollout 做 PPO 更新，并接上 SIL。
- `trainer.py` — 对示范动作做监督学习微调。
- `rollout.py` — 在游戏里按物理步采集 PPO 样本。
- `experimental_rollout.py` — 为表征实验和动作段采集变长决策。
- `rollout_pipeline.py` — 把采集放到单独进程，训练进程消费落后不超过一步的批次。
- `ppo_buffer.py` — 存放逐物理步的 PPO 样本，并切成训练 batch。
- `decision_buffer.py` — 存放跨多个物理帧的一条决策。
- `dataset.py` — 把轨迹编成模仿和 SIL 用的窗口。
- `reward.py` — 计算一步奖励和轨迹进度分。
- `reference_corridor.py` — 用参考折线做走廊进度和偏离计分。
- `reference_anchor.py` — 规定计分用的锅心点，不改观测特征。
- `reference_fingerprint.py` — 用参考线、锚点和计分口径标识一次任务，避免只凭文件哈希认错。
- `anchor_fit.py` — 拟合身体局部锅锚点，不把结果标成已验证。
- `anchor_projection.py` — 在 Unity 画面坐标和身体局部坐标之间投影。
- `tas_pot_detector.py` — 在单帧画面上按 TAS 的颜色轮廓找锅心。
- `delayed_exploration.py` — 按玩家档位推迟探索噪声的开启。
- `policy_features.py` — 组装在线采集和离线重放共用的观测特征。
- `policy_audit.py` — 用冻结权重重放，检查行为策略的概率是否对得上。
- `sparse_geometry.py` — 构造以玩家为中心的轮廓 token 和表面关系。
- `contact_features.py` — 用几何近似锤头和锅的地形接触。
- `geometry_profile.py` — 在加载旧 checkpoint 时保持原来的几何标定，不默默换成新标定。
- `deploy_sampling.py` — 从可着陆表面抽空投落点，并应用排除区。
- `state_schema.py` — 规定原始状态每一维的含义；线上宽度和网络输入宽度分开。
- `sil_archive.py` — 维护 SIL 精英池，可选 MAP-Elites 或 (t, D) Pareto。
- `run_logging.py` — 把一次实验的指标、元数据和精英池写进 `runs/<tag>/`。
- `run_log_parser.py` — 从控制台文本还原 `metrics.csv`。
- `migrate_legacy_runs.py` — 把旧的 checkpoint 和日志目录收成 `runs/<tag>/`。
- `archive_evals.py` — 把评估 JSON 里的逐回合统计收成可入库的 `summary.csv`。
