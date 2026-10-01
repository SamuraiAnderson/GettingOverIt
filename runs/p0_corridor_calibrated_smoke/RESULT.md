# 切换几何标定后的 critic 预热

2026-09-19。检验旧策略恢复后，显式切换走廊目标和几何标定会重新预热 critic。

```powershell
conda run --no-capture-output -n getting-over-it-analysis python -m src.training.main_ppo --resume runs/p2_contact/checkpoints/ppo_iter_0335.pt --reference-trajectory --geometry-profile native-pose-v1 --recipe ppo-ou --seed 7 --num-agents 2 --steps-per-rollout 128 --max-iterations 336 --value-warmup-iters 1 --run-dir p0_corridor_calibrated_smoke --persist-game --save-rollout-inputs
```

256 transitions，迭代 336 的 value_only=1，critic 执行 32 个 minibatch。直接比较采样前权重与最终 checkpoint，仅四个 critic_head 张量改变，backbone 和 actor 均逐位不变；预热结束迭代正确保存为 336，几何偏移保存为 −180°/−84.49833°，见 warmup_verification.json。

完整 256 样本离线重放最大 ratio 误差 7.99e-5、预更新 KL 1.45e-10。指标见 metrics.csv，命令与配置见 run_meta.json。

## 结论

几何语义切换、critic 预热及输入归档恢复通过。只有一轮工程验证，尚未训练或评价任务成功率。
