using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;
using GoiRuntime.Core.Interfaces;

namespace GoiRuntime.Communication
{
	/// <summary>
	/// TCP 阻塞步进服务器
	///
	/// 线路协议（小端二进制）：
	///   Python → C#  RESET:    [cmd:1B='R']
	///   Python → C#  STEP:     [cmd:1B='S'][n:1B][actions: n×2×4B]
	///   Python → C#  TELEPORT:    [cmd:1B='T'][agentIndex:1B][x:4B][y:4B]
	///   Python → C#  CAMERA_FREE: [cmd:1B='F'][enabled:1B]
	///   Python → C#  CLOSE:       [cmd:1B='X']
	///
	///   C# → Python  STATE: [n:1B][states: n×29×4B][dones: n×1B]
	///
	/// 线程模型：
	///   - 后台线程（_bgThread）：阻塞等待 Python 命令，写入 _pendingCommand，设置 _commandReady
	///   - Unity 主线程协程：轮询 TryGetCommand()，执行物理步进后调用 SendResponse()
	///   - 两方向通过 ManualResetEvent 和 volatile 变量交换，不需要锁
	/// </summary>
	public class TcpStepServer : IStepServer
	{
		private readonly int _stateDim;

		private TcpListener _listener;
		private TcpClient  _client;
		private NetworkStream _stream;
		private Thread _bgThread;

		private volatile bool _running;
		private volatile bool _connected;

		public TcpStepServer(int stateDim = 29)
		{
			_stateDim = stateDim > 0 ? stateDim : 29;
		}

	// 主线程 ↔ 后台线程共享的命令槽
	private StepCommand _pendingCommand;
	private readonly ManualResetEvent _commandReady  = new ManualResetEvent(false);
	private volatile bool _commandReadyFlag = false;

	// 发送缓冲区（在主线程写好，后台线程读取发出）
	private byte[] _responseBytes;
	private readonly ManualResetEvent _responseReady = new ManualResetEvent(false);

		public bool IsRunning   => _running;
		public bool IsConnected => _connected;

		// ── 公共 API ──────────────────────────────────────────────

		public void StartListening(int port)
		{
			_running = true;
			_listener = new TcpListener(IPAddress.Loopback, port);
			_listener.Start();
			Debug.Log($"[TcpStepServer] 监听 127.0.0.1:{port}，等待 Python 连接...");

			_bgThread = new Thread(BackgroundLoop) { IsBackground = true, Name = "TcpStepServer-BG" };
			_bgThread.Start();
		}

	public void Stop()
	{
		_running = false;

		// 解除后台线程可能的阻塞
		_commandReadyFlag = true;
		_commandReady.Set();
		_responseReady.Set();

			try { _stream?.Close(); } catch { }
			try { _client?.Close(); } catch { }
			try { _listener?.Stop(); } catch { }

			_bgThread?.Join(1000);
			_connected = false;
			Debug.Log("[TcpStepServer] 已停止");
		}

	/// <summary>
	/// Unity 主线程（协程）调用：非阻塞检查是否有新命令
	/// </summary>
	public bool TryGetCommand(out StepCommand command)
	{
		if (_commandReadyFlag)
		{
			command = _pendingCommand;
			_commandReadyFlag = false;
			_commandReady.Reset();
			return true;
		}
		command = default(StepCommand);
		return false;
	}

		/// <summary>
		/// Unity 主线程（协程）调用：在物理步进完成后，将状态打包发出
		/// </summary>
		public void SendResponse(StepResponse response)
		{
			if (!_connected) return;

		// 编码：[n:1B][states: n×stateDim×4B][dones: n×1B]
		int n = response.Dones?.Length ?? 0;
		byte[] buf = new byte[1 + n * _stateDim * 4 + n];
			int offset = 0;

			buf[offset++] = (byte)n;

			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < _stateDim; j++)
				{
					float v = (response.States != null && i * _stateDim + j < response.States.Length)
						? response.States[i * _stateDim + j] : 0f;
					byte[] fb = BitConverter.GetBytes(v);
					if (!BitConverter.IsLittleEndian) Array.Reverse(fb);
					Buffer.BlockCopy(fb, 0, buf, offset, 4);
					offset += 4;
				}
				buf[offset++] = (response.Dones != null && i < response.Dones.Length && response.Dones[i])
					? (byte)1 : (byte)0;
			}

		// 传给后台线程发送（避免在主线程阻塞 socket write）
		_responseBytes = buf;
		_responseReady.Set();
	}


		// ── 后台线程 ──────────────────────────────────────────────

		private void BackgroundLoop()
		{
			try
			{
				// 等待 Python 连接
				_client = _listener.AcceptTcpClient();
				_client.NoDelay = true;
				_stream = _client.GetStream();
				_connected = true;
				Debug.Log("[TcpStepServer] Python 已连接");

				while (_running && _connected)
				{
					// --- 读取命令 ---
					int cmdByte = _stream.ReadByte();
					if (cmdByte < 0) break;  // 连接关闭

					CommandType cmdType = (CommandType)cmdByte;
					StepCommand cmd = new StepCommand { Type = cmdType };

					if (cmdType == CommandType.Step)
					{
						// 读 n（agent 数量）
						int n = _stream.ReadByte();
						if (n < 0) break;

						// 读 n×2 个 float（actions）
						int actionBytes = n * 2 * 4;
						byte[] actionBuf = ReadExact(actionBytes);
						if (actionBuf == null) break;

						float[] actions = new float[n * 2];
						for (int i = 0; i < actions.Length; i++)
						{
							actions[i] = BitConverter.ToSingle(actionBuf, i * 4);
						}
						cmd.Actions = actions;
					}
					else if (cmdType == CommandType.Config)
					{
						// 读 [active:1B][mouseXActionId:4B][mouseYActionId:4B]
						byte[] configBuf = ReadExact(9);
						if (configBuf == null) break;
						cmd.ConfigRewiredMouseActive = configBuf[0] != 0;
						cmd.ConfigMouseXActionId = BitConverter.ToInt32(configBuf, 1);
						cmd.ConfigMouseYActionId = BitConverter.ToInt32(configBuf, 5);
					}

					else if (cmdType == CommandType.Teleport)
					{
						// [agentIndex:1B][x:4B][y:4B] = 9 bytes
						byte[] teleBuf = ReadExact(9);
						if (teleBuf == null) break;
						cmd.TeleportAgentIndex = teleBuf[0];
						cmd.TeleportX = BitConverter.ToSingle(teleBuf, 1);
						cmd.TeleportY = BitConverter.ToSingle(teleBuf, 5);
					}

					else if (cmdType == CommandType.Visualize)
					{
						byte[] vizBuf = ReadExact(1);
						if (vizBuf == null) break;
						cmd.VisualizeEnabled = vizBuf[0] != 0;
					}

					else if (cmdType == CommandType.CameraFree)
					{
						byte[] camBuf = ReadExact(1);
						if (camBuf == null) break;
						cmd.CameraFreeEnabled = camBuf[0] != 0;
					}

				// --- 通知主线程 ---
				_pendingCommand = cmd;
				_responseReady.Reset();
				_commandReadyFlag = true;
				_commandReady.Set();

				if (cmdType == CommandType.Close) break;

				// --- 等待主线程执行完物理步进并准备好回包 ---
				_responseReady.WaitOne();
					if (!_running) break;

					// --- 发送回包 ---
					byte[] resp = _responseBytes;
					if (resp != null)
					{
						_stream.Write(resp, 0, resp.Length);
						_stream.Flush();
					}
				}
			}
			catch (Exception e) when (_running)
			{
				Debug.LogError($"[TcpStepServer] 后台线程异常: {e.Message}");
			}
			finally
			{
				_connected = false;
				Debug.Log("[TcpStepServer] 后台线程退出");
			}
		}

		/// <summary>
		/// 从 stream 精确读取 count 字节，连接断开时返回 null
		/// </summary>
		private byte[] ReadExact(int count)
		{
			byte[] buf = new byte[count];
			int received = 0;
			while (received < count)
			{
				int n = _stream.Read(buf, received, count - received);
				if (n <= 0) return null;
				received += n;
			}
			return buf;
		}
	}
}
