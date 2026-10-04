现有数据的有限复用迁移包

范围：本次讨论的 confirmed_endpoint_v2_formal_20260928_S_s47 与
delayed_exploration_8h_20260930 两组运行。采用复制迁移，原目录原样保留。

warm_start_candidates/
  保存这两组运行现有的 checkpoint，作为模型初始化候选。
  新实验采用 --init-from <checkpoint>，不使用 --resume，暂不加载旧精英池。
  文件仍是原始 checkpoint，可能含 optimizer 等字段；使用方式决定是否恢复这些字段。
  --init-from 不保证自动清除旧 critic、归一化参数或旧策略影响。
  先验证新起点协议，再评估迁移效果；目录内模型尚未通过此验证，不代表最佳模型。

analysis_only/
  保存历史精英池及原始 fingerprint sidecar，供状态、动作、接触及复位问题分析。
  不传给 --sil-resume-pool / --sil-init-pool / --sil-seed，也不作为干净起点的 BC 数据。
  不把旧分数与修复后的成绩混排。
  不同迭代的池有重叠，不能直接拼接当作独立样本。

provenance/
  保存每组运行的 run_meta.json、metrics.csv、console.log，用于追溯采集条件。
  delayed_exploration 本次进程实际覆盖迭代 164–251，共 88 轮。

evidence/
  保存复位实验数据、脚本和当前关键实现，供复核。
  代码是迁移时的版本，不代表已确认的历史部署 DLL。

目前已批准直接用于新训练的轨迹：0 条。
尚无证据支持按固定 40 步截断后就视为干净数据，因此没有擅自裁剪轨迹、清零速度或改写奖励。
后续局部片段需结合历史输入窗口及新环境中的验证，再创建独立训练集。
本包不是独立运行环境，模型训练仍依赖原仓库中的配置、地图和参考路线。

manifest.json 列出每个文件的原路径、新路径、用途、大小及 SHA-256。
迁移校验：复制前后与源文件哈希一致；NPZ 可读取，状态/动作有限且形状对应。
这些校验不等价于游戏重放成功，也不认证复位质量。

复核或补全相同来源的复制：
conda run -n getting-over-it-analysis python runs/reusable_data_20261001/migrate.py
脚本遇到目标同名但内容不同时会停止，不覆盖已有不同数据。
