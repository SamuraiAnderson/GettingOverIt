using System;

namespace GoiRuntime.Core.Interfaces
{
	/// <summary>
	/// UDP 接收服务接口
	/// 从 Python 接收动作数据
	/// </summary>
	public interface IUDPReceiveService
	{
		/// <summary>
		/// 启动服务
		/// </summary>
		void Start(string host, int port);

		/// <summary>
		/// 停止服务
		/// </summary>
		void Stop();

		/// <summary>
		/// 服务是否运行中
		/// </summary>
		bool IsRunning { get; }

		/// <summary>
		/// 尝试获取最新动作（非阻塞）
		/// </summary>
		/// <param name="actions">输出动作数组（num_duplis × action_dim）</param>
		/// <returns>是否成功获取新动作</returns>
		bool TryGetActions(out float[] actions);

		/// <summary>
		/// 获取动作维度（action_dim × num_duplis）
		/// </summary>
		int GetActionDimension();

		/// <summary>
		/// 获取已接收的数据包数量
		/// </summary>
		long GetReceivedCount();

		/// <summary>
		/// 获取接收频率 (Hz)
		/// </summary>
		double GetReceiveFrequency();

		/// <summary>
		/// 获取最后接收时间戳
		/// </summary>
		float GetLastReceiveTimestamp();
	}

	/// <summary>
	/// UDP 发送服务接口
	/// 向 Python 发送状态数据
	/// </summary>
	public interface IUDPSendService
	{
		/// <summary>
		/// 启动服务
		/// </summary>
		void Start(string host, int port);

		/// <summary>
		/// 停止服务
		/// </summary>
		void Stop();

		/// <summary>
		/// 服务是否运行中
		/// </summary>
		bool IsRunning { get; }

		/// <summary>
		/// 发送所有复制体的状态
		/// </summary>
		void SendStates(float[] states);

		/// <summary>
		/// 发送单个复制体的状态
		/// </summary>
		void SendState(float[] state, int duplicateIndex);

		/// <summary>
		/// 发送字节数据
		/// </summary>
		void SendBytes(byte[] data);

		/// <summary>
		/// 获取已发送的数据包数量
		/// </summary>
		long GetSentCount();

		/// <summary>
		/// 获取发送频率 (Hz)
		/// </summary>
		double GetSendFrequency();

		/// <summary>
		/// 获取最后发送时间戳
		/// </summary>
		float GetLastSendTimestamp();
	}

	/// <summary>
	/// 通信协议接口（统一的通信抽象）
	/// </summary>
	public interface IProtocol
	{
		/// <summary>
		/// 初始化协议
		/// </summary>
		void Initialize();

		/// <summary>
		/// 关闭协议
		/// </summary>
		void Shutdown();

		/// <summary>
		/// 协议是否就绪
		/// </summary>
		bool IsReady { get; }

		/// <summary>
		/// 尝试接收动作
		/// </summary>
		bool TryReceiveActions(out float[] actions);

		/// <summary>
		/// 发送状态
		/// </summary>
		void SendStates(float[] states);

		/// <summary>
		/// 检查命令信号（用于文件信号控制）
		/// </summary>
		bool CheckCommand(string commandName, out string parameter);
	}
}

