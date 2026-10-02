# 0269 权重的 8 独立世界冷启动测试

使用 `runs/optimizer_momentum_fix_20261001/checkpoints/ppo_iter_0269.pt`，确定性实时推理，
8 个完整 Mian 场景、8 个不同 PhysicsScene2D，每个世界一个原始 Player。
输入、观测和推理复用既有评估器，1200 步，不训练，不额外 reset/预热/投放。
权重 SHA-256：`d29e3f764e1d3a0a506216fceac0912b5b88a5871ab793876cd7bac9bf9f37fe`。

- 9600 个 agent step 完整；8 份动作逐位一致，35D 观测排除 timestamp 后全程逐位一致。
- 第 31 和 1200 步完整诊断中的刚体、控制器及关节状态一致。
- 每个角色距离起点的最大位移约 11.099 米，后段均出现运动停滞，没有单独分离。
- 启动至策略就绪约 18.72 秒，采集约 283.07 秒，总计约 314.26 秒。
- 锤头描边、自由相机开启；运行后保留窗口。生产 DLL/配置及模型权重未改变。

边界：这些结果只验证此策略和轨迹。多世界起点相对原单世界参考仍存在同步时机差异，
没有据此宣布完整动力学等价或把实验实现接入生产 PPO 流程。

收尾问题：旧实验接口在 1200 步完成、Python 正常关闭连接之后，将 CLOSE 当作未支持
命令，覆盖了 `finished.json`。原始文件保留，`consistency_check.json` 明确标为
`TRAJECTORY_PASS_RUNTIME_RECORD_ERROR`，轨迹一致性单列 PASS。提交版本修正 CLOSE，
等待原子写入的 `closed.json` 才报告完成；已编译，未关闭留开的窗口再做在线重跑。

当前源码与复现说明：`src/tests/control_interaction/world_isolation/README.md`。
本目录只将元数据、结果和检查摘要入库；原始逐帧数据、权重、DLL 保留在本地。
