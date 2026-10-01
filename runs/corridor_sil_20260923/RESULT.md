# C + corridor SIL 对照

状态：completed；更新：2026-09-23T10:06:16+08:00。

8 agents × 1200 步，batch 128，lr 3e-5，gamma .99 / lambda .95，native-pose-v1，固定起点。
同一历史源权重初始化；两轮 critic 预热计入样本。SIL 使用 sil_seed_590.npz，默认第 6 轮起执行优势加权更新。
每个 seed 连续训练保留内存精英池；达到约定终点后自动评估。评估与正式中期一致：seed 20260922，2 rounds × 8 agents × det/exploration。

## Seed 31

已训练 97,939 个实际样本，13 轮。

- 正式 C 96k 锚点：实际样本 102,100；det 中位数 2.715m，exploration 4.065m；det 保持 1 秒 0/16。
- 本次 C + SIL：实际样本 97,939；det 中位数 0.004m，exploration 1.208m；det 保持 1 秒 0/16。
- 同 seed det 中位数变化：-2.711m。

## Seed 47

已训练 99,088 个实际样本，12 轮。

- 正式 C 96k 锚点：实际样本 97,205；det 中位数 3.834m，exploration 3.611m；det 保持 1 秒 0/16。
- 本次 C + SIL：实际样本 99,088；det 中位数 0.000m，exploration 2.427m；det 保持 1 秒 0/16。
- 同 seed det 中位数变化：-3.834m。

2.72m 对应原 C seed 31。完整分 seed、两个 seed 的回合汇总及实际样本量见 comparison.json。
种子轨迹来自历史 T 组中期评估；本次衡量优质轨迹冷启动的增益，不是未经挑选数据的独立保留集结论。
训练连续执行，不复刻历史 run 分段恢复时的随机数重置。仅两个训练 seed，不将复制体回合数当独立复验。

在本目录创建 STOP 文件可终止自有进程树；每轮已完成 checkpoint 保留。结束时恢复启动前游戏运行配置。
