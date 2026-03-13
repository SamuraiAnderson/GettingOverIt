namespace GoiRuntime.Core.Interfaces
{
	/// <summary>
	/// RL 步进服务器接口
	/// 实现 Python ↔ C# 的帧级别同步：Python 发 STEP 后阻塞等待，
	/// C# 执行 Physics2D.Simulate 后才回包，保证 1 action = 精确 N 物理帧
	/// </summary>
	public interface IStepServer
	{
		/// <summary>
		/// 开始监听指定端口，接受 Python 连接
		/// </summary>
		void StartListening(int port);

		/// <summary>
		/// 停止服务器并关闭连接
		/// </summary>
		void Stop();

		/// <summary>
		/// 服务器是否正在运行
		/// </summary>
		bool IsRunning { get; }

		/// <summary>
		/// Python 是否已连接
		/// </summary>
		bool IsConnected { get; }

		/// <summary>
		/// 非阻塞检查是否有新命令到达（由 Unity 主线程协程轮询）
		/// </summary>
		bool TryGetCommand(out StepCommand command);

		/// <summary>
		/// 发送状态回包给 Python（在 Physics2D.Simulate 完成后调用）
		/// </summary>
		void SendResponse(StepResponse response);
	}

	/// <summary>
	/// 命令类型
	/// </summary>
	public enum CommandType : byte
	{
		/// <summary>重置所有 agent 到初始状态</summary>
		Reset = (byte)'R',

		/// <summary>执行一个 RL step（携带 N 个 agent 的动作）</summary>
		Step  = (byte)'S',

		/// <summary>关闭连接，退出训练</summary>
		Close = (byte)'X',
	}

	/// <summary>
	/// 从 Python 收到的命令
	/// </summary>
	public struct StepCommand
	{
		/// <summary>命令类型</summary>
		public CommandType Type;

		/// <summary>
		/// 动作数组，长度 = numAgents × actionDim(2)
		/// 布局：[agent0_x, agent0_y, agent1_x, agent1_y, ...]
		/// </summary>
		public float[] Actions;
	}

	/// <summary>
	/// 发送给 Python 的状态回包
	/// </summary>
	public struct StepResponse
	{
		/// <summary>
		/// 状态数组，长度 = numAgents × stateDim(29)
		/// 布局：[agent0_s0..s28, agent1_s0..s28, ...]
		/// </summary>
		public float[] States;

		/// <summary>
		/// 每个 agent 是否已终止（done），长度 = numAgents
		/// </summary>
		public bool[] Dones;
	}
}
