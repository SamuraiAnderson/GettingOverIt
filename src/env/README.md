# 环境客户端

Python 侧与游戏内 TCP 服务说话。不决定训什么、不打分。

- `goi_env.py` — 发送 RESET、STEP、TELEPORT，并读回每个复制体的状态。
- `protocol.py` — 打包和解包 legacy33 与 anchor35 帧。
