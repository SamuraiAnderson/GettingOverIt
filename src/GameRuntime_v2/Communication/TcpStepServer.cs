using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using GoiRuntime.Core.Interfaces;
using GoiRuntime.Core.Protocol;

namespace GoiRuntime.Communication
{
	/// <summary>
	/// TCP 阻塞步进服务器。每个连接默认 legacy33；HELLO 成功后才进入 anchor35。
	/// </summary>
	public class TcpStepServer : IStepServer
	{
		private readonly int _internalStateDim;
		private readonly int _legacyWireDim;
		private readonly HelloSnapshot _helloSnapshot;

		private TcpListener _listener;
		private TcpClient  _client;
		private NetworkStream _stream;
		private Thread _bgThread;

		private volatile bool _running;
		private volatile bool _connected;
		private volatile bool _anchor35;

		public TcpStepServer(int internalStateDim, int legacyWireDim, HelloSnapshot snapshot)
		{
			_internalStateDim = internalStateDim > 0 ? internalStateDim : 35;
			_legacyWireDim = legacyWireDim > 0 ? legacyWireDim : 33;
			_helloSnapshot = snapshot ?? new HelloSnapshot();
		}

		public TcpStepServer(int stateDim = 33)
			: this(stateDim <= 33 ? 35 : stateDim, 33, new HelloSnapshot())
		{
		}

	private StepCommand _pendingCommand;
	private readonly ManualResetEvent _commandReady  = new ManualResetEvent(false);
	private volatile bool _commandReadyFlag = false;

	private byte[] _responseBytes;
	private readonly ManualResetEvent _responseReady = new ManualResetEvent(false);

		public bool IsRunning   => _running;
		public bool IsConnected => _connected;

		public void StartListening(int port)
		{
			_running = true;
			_anchor35 = false;
			_listener = new TcpListener(IPAddress.Loopback, port);
			_listener.Start();
			Debug.Log($"[TcpStepServer] 监听 127.0.0.1:{port}，等待 Python 连接（默认 legacy33）...");

			_bgThread = new Thread(BackgroundLoop) { IsBackground = true, Name = "TcpStepServer-BG" };
			_bgThread.Start();
		}

	public void Stop()
	{
		_running = false;
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

		public void SendResponse(StepResponse response)
		{
			if (!_connected) return;
			if (response.DiagnosticJson != null)
			{
				_responseBytes = _anchor35
					? PackV2Diagnostic(response.Sequence, response.DiagnosticJson)
					: PackLegacyDiagnostic(response.DiagnosticJson);
				_responseReady.Set();
				return;
			}
			if (_anchor35 && response.ErrorCode != 0)
			{
				_responseBytes = PackV2Error(response.Sequence, response.ErrorCode,
					string.IsNullOrEmpty(response.ErrorMessage) ? "state_error" : response.ErrorMessage);
				_responseReady.Set();
				return;
			}

			int n = response.Dones != null ? response.Dones.Length : 0;
			int internalDim = _internalStateDim;
			if (response.States != null && n > 0 && response.States.Length % n == 0)
				internalDim = response.States.Length / n;
			int wireDim = _anchor35 ? internalDim : Math.Min(_legacyWireDim, internalDim);
			_responseBytes = _anchor35
				? PackV2State(response, n, internalDim, wireDim)
				: PackLegacyState(response, n, internalDim, wireDim);
			_responseReady.Set();
		}

		private static byte[] PackLegacyDiagnostic(string jsonText)
		{
			byte[] json = Encoding.UTF8.GetBytes(jsonText);
			byte[] packet = new byte[5 + json.Length];
			packet[0] = 255;
			WriteInt(packet, 1, json.Length);
			Buffer.BlockCopy(json, 0, packet, 5, json.Length);
			return packet;
		}

		private static byte[] PackV2Diagnostic(int sequence, string jsonText)
		{
			byte[] json = Encoding.UTF8.GetBytes(jsonText);
			return PackV2(HelloProtocol.V2Diagnostic, sequence, json);
		}

		private static byte[] PackV2Error(int sequence, int code, string message)
		{
			byte[] msg = Encoding.UTF8.GetBytes(message ?? "");
			byte[] payload = new byte[6 + msg.Length];
			WriteInt(payload, 0, code);
			payload[4] = (byte)(msg.Length & 0xFF);
			payload[5] = (byte)((msg.Length >> 8) & 0xFF);
			Buffer.BlockCopy(msg, 0, payload, 6, msg.Length);
			return PackV2(HelloProtocol.V2Error, sequence, payload);
		}

		private static byte[] PackV2State(StepResponse response, int n, int internalDim, int wireDim)
		{
			byte[] payload = new byte[1 + n * (wireDim * 4 + 1)];
			payload[0] = (byte)n;
			int offset = 1;
			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < wireDim; j++)
				{
					float v = (response.States != null && i * internalDim + j < response.States.Length)
						? response.States[i * internalDim + j] : 0f;
					WriteFloat(payload, offset, v);
					offset += 4;
				}
				payload[offset++] = (response.Dones != null && i < response.Dones.Length && response.Dones[i])
					? (byte)1 : (byte)0;
			}
			return PackV2(HelloProtocol.V2State, response.Sequence, payload);
		}

		private static byte[] PackLegacyState(StepResponse response, int n, int internalDim, int wireDim)
		{
			byte[] buf = new byte[1 + n * wireDim * 4 + n];
			int offset = 0;
			buf[offset++] = (byte)n;
			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < wireDim; j++)
				{
					float v = (response.States != null && i * internalDim + j < response.States.Length)
						? response.States[i * internalDim + j] : 0f;
					WriteFloat(buf, offset, v);
					offset += 4;
				}
				buf[offset++] = (response.Dones != null && i < response.Dones.Length && response.Dones[i])
					? (byte)1 : (byte)0;
			}
			return buf;
		}

		private static byte[] PackV2(byte type, int sequence, byte[] payload)
		{
			byte[] body = new byte[4 + payload.Length];
			WriteInt(body, 0, sequence);
			Buffer.BlockCopy(payload, 0, body, 4, payload.Length);
			byte[] packet = new byte[5 + body.Length];
			packet[0] = type;
			WriteInt(packet, 1, body.Length);
			Buffer.BlockCopy(body, 0, packet, 5, body.Length);
			return packet;
		}

		private static void WriteInt(byte[] buf, int offset, int value)
		{
			byte[] bytes = BitConverter.GetBytes(value);
			if (!BitConverter.IsLittleEndian) Array.Reverse(bytes);
			Buffer.BlockCopy(bytes, 0, buf, offset, 4);
		}

		private static void WriteFloat(byte[] buf, int offset, float value)
		{
			byte[] bytes = BitConverter.GetBytes(value);
			if (!BitConverter.IsLittleEndian) Array.Reverse(bytes);
			Buffer.BlockCopy(bytes, 0, buf, offset, 4);
		}

		private void BackgroundLoop()
		{
			try
			{
				_client = _listener.AcceptTcpClient();
				_client.NoDelay = true;
				_stream = _client.GetStream();
				_connected = true;
				_anchor35 = false;
				Debug.Log("[TcpStepServer] Python 已连接（legacy33，等待 HELLO）");

				while (_running && _connected)
				{
					int cmdByte = _stream.ReadByte();
					if (cmdByte < 0) break;

					CommandType cmdType = (CommandType)cmdByte;
					if (cmdType == CommandType.Hello)
					{
						if (!HandleHelloOnBackgroundThread())
							break;
						continue;
					}

					StepCommand cmd = new StepCommand { Type = cmdType };

					if (cmdType == CommandType.Step)
					{
						int n = _stream.ReadByte();
						if (n < 0) break;
						int actionBytes = n * 2 * 4;
						byte[] actionBuf = ReadExact(actionBytes);
						if (actionBuf == null) break;
						float[] actions = new float[n * 2];
						for (int i = 0; i < actions.Length; i++)
							actions[i] = BitConverter.ToSingle(actionBuf, i * 4);
						cmd.Actions = actions;
					}
					else if (cmdType == CommandType.Config)
					{
						byte[] configBuf = ReadExact(9);
						if (configBuf == null) break;
						cmd.ConfigRewiredMouseActive = configBuf[0] != 0;
						cmd.ConfigMouseXActionId = BitConverter.ToInt32(configBuf, 1);
						cmd.ConfigMouseYActionId = BitConverter.ToInt32(configBuf, 5);
					}
					else if (cmdType == CommandType.Teleport)
					{
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

				_pendingCommand = cmd;
				_responseReady.Reset();
				_commandReadyFlag = true;
				_commandReady.Set();

				if (cmdType == CommandType.Close) break;

				_responseReady.WaitOne();
					if (!_running) break;

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
				_anchor35 = false;
				Debug.Log("[TcpStepServer] 后台线程退出");
			}
		}

		private bool HandleHelloOnBackgroundThread()
		{
			byte[] lenBuf = ReadExact(4);
			if (lenBuf == null) return false;
			int length = BitConverter.ToInt32(lenBuf, 0);
			if (length < 0 || length > 1024 * 1024) return false;
			byte[] body = ReadExact(length);
			if (body == null) return false;

			HelloRequest request = null;
			try
			{
				request = JsonUtility.FromJson<HelloRequest>(Encoding.UTF8.GetString(body));
			}
			catch (Exception e)
			{
				Debug.LogWarning("[TcpStepServer] HELLO JSON 解析失败: " + e.Message);
			}

			string reject;
			bool accepted = HelloProtocol.TryAccept(request, _helloSnapshot, out reject);
			if (accepted)
			{
				_anchor35 = true;
				Debug.Log("[TcpStepServer] HELLO 接受，进入 anchor35");
			}
			else
			{
				_anchor35 = false;
				Debug.LogWarning("[TcpStepServer] HELLO 拒绝: " + reject);
			}

			string reply = HelloProtocol.ToJson(_helloSnapshot, accepted, reject);
			byte[] utf = Encoding.UTF8.GetBytes(reply);
			byte[] packet = new byte[5 + utf.Length];
			packet[0] = HelloProtocol.Marker;
			WriteInt(packet, 1, utf.Length);
			Buffer.BlockCopy(utf, 0, packet, 5, utf.Length);
			_stream.Write(packet, 0, packet.Length);
			_stream.Flush();
			return true;
		}

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
