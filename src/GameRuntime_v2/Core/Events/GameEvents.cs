namespace GoiRuntime.Core.Events
{
	/// <summary>
	/// 游戏事件常量定义
	/// 集中管理所有事件名称，避免硬编码字符串
	/// </summary>
	public static class GameEvents
	{
		#region Player 事件

		/// <summary>
		/// Player 初始化完成
		/// 数据: GameObject player
		/// </summary>
		public const string PlayerInitialized = "player.initialized";

		/// <summary>
		/// Player 输入改变
		/// 数据: Vector2 input
		/// </summary>
		public const string PlayerInputChanged = "player.input.changed";

		/// <summary>
		/// Player 状态更新
		/// 数据: PlayerState state
		/// </summary>
		public const string PlayerStateUpdated = "player.state.updated";

		/// <summary>
		/// 复制体初始化完成
		/// 数据: int duplicateCount
		/// </summary>
		public const string DuplicatesInitialized = "duplicates.initialized";

		#endregion

		#region 碰撞箱事件

		/// <summary>
		/// 碰撞箱采集完成
		/// 数据: ColliderData[] colliders
		/// </summary>
		public const string CollidersCollected = "colliders.collected";

		/// <summary>
		/// 碰撞箱数据导出
		/// 数据: string filePath
		/// </summary>
		public const string ColliderDataExported = "colliders.exported";

		#endregion

		#region 通信事件

		/// <summary>
		/// 接收到动作
		/// 数据: float[] actions
		/// </summary>
		public const string ActionReceived = "udp.action.received";

		/// <summary>
		/// 状态已发送
		/// 数据: float[] states
		/// </summary>
		public const string StateSent = "udp.state.sent";

		/// <summary>
		/// UDP 连接丢失
		/// 数据: string reason
		/// </summary>
		public const string ConnectionLost = "udp.connection.lost";

		/// <summary>
		/// UDP 接收服务启动
		/// 数据: string endpoint
		/// </summary>
		public const string ReceiveServiceStarted = "udp.receive.started";

		/// <summary>
		/// UDP 发送服务启动
		/// 数据: string endpoint
		/// </summary>
		public const string SendServiceStarted = "udp.send.started";

		#endregion

		#region 系统事件

		/// <summary>
		/// 系统初始化完成
		/// 数据: null
		/// </summary>
		public const string SystemInitialized = "system.initialized";

		/// <summary>
		/// 系统关闭
		/// 数据: null
		/// </summary>
		public const string SystemShutdown = "system.shutdown";

		/// <summary>
		/// 系统错误
		/// 数据: string errorMessage
		/// </summary>
		public const string ErrorOccurred = "system.error";

		/// <summary>
		/// 配置加载
		/// 数据: RuntimeConfig config
		/// </summary>
		public const string ConfigLoaded = "config.loaded";

		#endregion

		#region 测试事件

		/// <summary>
		/// 测试开始
		/// 数据: string testName
		/// </summary>
		public const string TestStarted = "test.started";

		/// <summary>
		/// 测试完成
		/// 数据: TestReport report
		/// </summary>
		public const string TestCompleted = "test.completed";

		/// <summary>
		/// 测试失败
		/// 数据: string errorMessage
		/// </summary>
		public const string TestFailed = "test.failed";

		#endregion
	}
}

