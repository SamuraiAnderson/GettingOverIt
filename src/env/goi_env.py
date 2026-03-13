"""
Getting Over It 强化学习环境接口

与 C# TcpStepServer 通信，实现帧级别同步步进。

线路协议（小端二进制）：
  Python → C#  RESET: [cmd:1B='R']
  Python → C#  STEP:  [cmd:1B='S'][n:1B][actions: n×2×4B]
  Python → C#  CLOSE: [cmd:1B='X']

  C# → Python  STATE: [n:1B][states: n×29×4B][dones: n×1B]
"""
import socket
import struct
import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

STATE_DIM  = 29
ACTION_DIM = 2

CMD_RESET = b'R'
CMD_STEP  = b'S'
CMD_CLOSE = b'X'


class GoiEnv:
    """
    Getting Over It 帧级别 RL 环境

    C# 侧在 GameRuntime 模式下启动 TcpStepServer，
    本类作为 TCP 客户端连接并驱动步进。

    参数：
        host:       C# 服务器地址（默认 localhost）
        port:       TCP 端口（默认 9000，与 RuntimeConfig.tcpPort 一致）
        num_agents: agent 数量（与 C# StepController 配置一致）
        timeout:    连接超时（秒）
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        num_agents: int = 1,
        timeout: float = 60.0,
    ):
        self.host       = host
        self.port       = port
        self.num_agents = num_agents
        self.timeout    = timeout

        self._sock: Optional[socket.socket] = None
        self._connected = False

        # 统计
        self.step_count   = 0
        self.total_time   = 0.0
        self.step_times: list[float] = []

    # ── 连接管理 ─────────────────────────────────────────────────

    def connect(self):
        """连接到 C# TcpStepServer，含重试等待"""
        deadline = time.time() + self.timeout
        last_err = None
        while time.time() < deadline:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.settimeout(5.0)
                s.connect((self.host, self.port))
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._sock = s
                self._connected = True
                logger.info("[GoiEnv] 已连接到 %s:%d", self.host, self.port)
                return
            except (ConnectionRefusedError, OSError) as e:
                s.close()
                last_err = e
                time.sleep(0.5)
        raise ConnectionError(f"[GoiEnv] 连接超时，最后错误: {last_err}")

    def close(self):
        """发送 CLOSE 命令并断开连接"""
        if self._connected:
            try:
                self._sock.sendall(CMD_CLOSE)
            except Exception:
                pass
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = None
        self._connected = False
        logger.info("[GoiEnv] 连接已关闭")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()

    # ── 核心 API ─────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """
        重置所有 agent，返回初始观测。
        shape: (num_agents, STATE_DIM)
        """
        self._send_all(CMD_RESET)
        states, _ = self._recv_response()
        self.step_count = 0
        self.step_times.clear()
        logger.debug("[GoiEnv] reset() 完成")
        return states

    def step(self, actions: np.ndarray):
        """
        执行一步，actions shape: (num_agents, ACTION_DIM)。
        返回 (obs, dones) ：
          obs   shape (num_agents, STATE_DIM)
          dones shape (num_agents,)  bool
        """
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        t0 = time.perf_counter()

        # 编码 STEP 命令
        n = self.num_agents
        buf = bytearray()
        buf += CMD_STEP
        buf += struct.pack("B", n)
        buf += actions.flatten().astype("<f4").tobytes()
        self._send_all(bytes(buf))

        obs, dones = self._recv_response()

        elapsed = time.perf_counter() - t0
        self.step_count  += 1
        self.total_time  += elapsed
        self.step_times.append(elapsed)

        return obs, dones

    # ── 低级 IO ──────────────────────────────────────────────────

    def _send_all(self, data: bytes):
        if not self._connected:
            raise RuntimeError("[GoiEnv] 未连接")
        self._sock.sendall(data)

    def _recv_response(self):
        """
        读取 C# 回包，解析为 (states, dones)。
        states shape: (num_agents, STATE_DIM)
        dones  shape: (num_agents,) bool
        """
        # 第一字节：agent 数量
        n_byte = self._recv_exact(1)
        n = struct.unpack("B", n_byte)[0]

        state_bytes = n * STATE_DIM * 4
        done_bytes  = n

        raw_states = self._recv_exact(state_bytes)
        raw_dones  = self._recv_exact(done_bytes)

        states = np.frombuffer(raw_states, dtype="<f4").reshape(n, STATE_DIM).copy()
        dones  = np.frombuffer(raw_dones,  dtype=np.uint8).astype(bool).copy()

        return states, dones

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("[GoiEnv] 连接断开")
            buf.extend(chunk)
        return bytes(buf)

    # ── 诊断 ─────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """返回步进统计信息"""
        if not self.step_times:
            return {"steps": 0}
        arr = np.array(self.step_times) * 1000  # ms
        return {
            "steps":    self.step_count,
            "mean_ms":  float(arr.mean()),
            "std_ms":   float(arr.std()),
            "min_ms":   float(arr.min()),
            "max_ms":   float(arr.max()),
            "total_s":  self.total_time,
        }
