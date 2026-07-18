"""
Getting Over It 强化学习环境接口

与 C# TcpStepServer 通信，实现帧级别同步步进。

线路协议（小端二进制）：
  Python → C#  RESET:            [cmd:1B='R']
  Python → C#  STEP:             [cmd:1B='S'][n:1B][actions: n×2×4B]
  Python → C#  NEW_SNAPSHOT:     [cmd:1B='N']  重新拍摄快照（warmup 后调用）
  Python → C#  CONFIG:           [cmd:1B='C'][active:1B][mouseXId:4B][mouseYId:4B]  RewiredMouseOverride
  Python → C#  EXPORT_COLLIDERS: [cmd:1B='E']  导出碰撞体几何到文件
  Python → C#  TELEPORT:         [cmd:1B='T'][agentIndex:1B][x:4B][y:4B]  传送 agent 到目标坐标
  Python → C#  CAMERA_FREE:      [cmd:1B='F'][enabled:1B]  启用/禁用自由相机
  Python → C#  CLOSE:            [cmd:1B='X']

  C# → Python  STATE: [n:1B] 然后对每个 agent: [state_i: 33×4B][done_i: 1B]

状态维度 33 = 基础 29 维（PlayerState.ToFloatArray）+ fakeCursor 4 维
（cursorX, cursorY, cursorVelX, cursorVelY，均为绝对量，索引 29-32）。
"""
import socket
import struct
import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

STATE_DIM  = 33
ACTION_DIM = 2

CMD_RESET       = b'R'
CMD_STEP        = b'S'
CMD_NEW_SNAPSHOT = b'N'
CMD_CONFIG      = b'C'
CMD_VISUALIZE        = b'V'
CMD_EXPORT_COLLIDERS = b'E'
CMD_TELEPORT         = b'T'
CMD_CAMERA_FREE      = b'F'
CMD_CLOSE            = b'X'


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
                # 连接成功后切换为阻塞模式（无超时），避免 C# 协程处理延迟触发 TimeoutError
                s.settimeout(None)
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

    def config_rewired_mouse(
        self,
        active: bool = True,
        mouse_x_action_id: int = -1,
        mouse_y_action_id: int = -1,
    ) -> None:
        """
        配置 RewiredMouseOverride（L3 可重复性测试用）。
        启用后屏蔽真实鼠标输入，仅使用注入值，避免测试期间鼠标移动影响轨迹。
        mouse_x_action_id / mouse_y_action_id：诊断得到的 Rewired actionId，-1 表示未诊断。
        """
        buf = bytearray()
        buf += CMD_CONFIG
        buf += struct.pack("B", 1 if active else 0)
        buf += struct.pack("<i", mouse_x_action_id)
        buf += struct.pack("<i", mouse_y_action_id)
        self._send_all(bytes(buf))
        self._recv_response()

    def new_snapshot(self) -> np.ndarray:
        """
        让 C# 以当前物理状态为基准重新拍快照（warmup 结束后调用）。
        之后每次 reset() 都将回到此时刻的状态。
        返回当前状态 shape: (num_agents, STATE_DIM)
        """
        self._send_all(CMD_NEW_SNAPSHOT)
        states, _ = self._recv_response()
        logger.info("[GoiEnv] new_snapshot() 完成，新基准状态已拍摄")
        return states

    def toggle_collider_visual(self, enabled: bool = True) -> None:
        """
        开启/关闭游戏内碰撞箱描边可视化。
        C# 侧 ColliderVisualizer 用 GL 绘制 Player 所有 Collider2D 轮廓。
        """
        buf = CMD_VISUALIZE + struct.pack("B", 1 if enabled else 0)
        self._send_all(buf)
        self._recv_response()
        logger.info("[GoiEnv] collider visual %s", "ON" if enabled else "OFF")

    def export_colliders(self) -> None:
        """
        请求 C# 导出碰撞体几何到文件。
        导出完成后文件位于 GoiData/Colliders/:
          - environment.json    环境（Mountain）世界坐标顶点
          - player_contour.json Player 轮廓本地坐标顶点
        """
        self._send_all(CMD_EXPORT_COLLIDERS)
        self._recv_response()
        logger.info("[GoiEnv] 碰撞体几何已导出到 GoiData/Colliders/")

    def teleport(self, x: float, y: float, agent_index: int = 0) -> np.ndarray:
        """
        传送指定 agent 到目标世界坐标 (x, y)。
        Player 是铰接体，C# 侧会整体平移所有子刚体并清零速度。
        返回传送后的状态 shape: (num_agents, STATE_DIM)
        """
        buf = bytearray()
        buf += CMD_TELEPORT
        buf += struct.pack("B", agent_index)
        buf += struct.pack("<ff", x, y)
        self._send_all(bytes(buf))
        states, _ = self._recv_response()
        logger.info("[GoiEnv] teleport agent %d → (%.2f, %.2f)", agent_index, x, y)
        return states

    def set_camera_free(self, enabled: bool = True) -> None:
        """
        启用/禁用自由相机模式。
        启用后解除原相机跟随锁定，允许滚轮缩放和鼠标中键拖动平移。
        """
        buf = CMD_CAMERA_FREE + struct.pack("B", 1 if enabled else 0)
        self._send_all(buf)
        self._recv_response()
        logger.info("[GoiEnv] camera free %s", "ON" if enabled else "OFF")

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

        C# 发送格式（TcpStepServer.SendResponse）：
          [n: 1B] 然后对每个 agent: [state_i: STATE_DIM×4B][done_i: 1B]
        注意：state 与 done 是按 agent 交错发送的，不能一次性读取所有 state 再读所有 done。
        """
        n = struct.unpack("B", self._recv_exact(1))[0]

        all_states: list[np.ndarray] = []
        all_dones:  list[bool]       = []
        for _ in range(n):
            raw_state = self._recv_exact(STATE_DIM * 4)
            raw_done  = self._recv_exact(1)
            all_states.append(np.frombuffer(raw_state, dtype="<f4").copy())
            all_dones.append(bool(raw_done[0]))

        if all_states:
            states = np.stack(all_states)          # (n, STATE_DIM)
        else:
            states = np.zeros((0, STATE_DIM), dtype="<f4")
        dones = np.array(all_dones, dtype=bool)    # (n,)

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
