using UnityEngine;
using GoiRuntime.Core.Configuration;
using GoiRuntime.Core.Utilities;
using GoiRuntime.Core.Events;

namespace GoiRuntime.Core
{
	/// <summary>
	/// 模式管理器
	/// 负责管理游戏的三种互斥运行模式
	/// </summary>
	public class ModeManager
	{
		private GameMode currentMode;
		private RuntimeConfig config;

		// 信号文件名称
		private const string SIGNAL_DATA_COLLECTION = "mode_data_collection";
		private const string SIGNAL_GAME_RUNTIME = "mode_game_runtime";
		private const string SIGNAL_GAME_TESTING = "mode_game_testing";

		/// <summary>
		/// 获取当前模式
		/// </summary>
		public GameMode CurrentMode => currentMode;

		/// <summary>
		/// 初始化模式管理器
		/// </summary>
		public void Initialize(RuntimeConfig config)
		{
			this.config = config;
			
			// 从配置或信号文件确定模式
			currentMode = DetermineMode();

			Debug.Log($"🎮 模式管理器初始化: {GetModeDisplayName(currentMode)}");
			
			// 发布模式切换事件
			EventBus.Publish(GameEvents.SystemInitialized, currentMode);
		}

		/// <summary>
		/// 确定运行模式
		/// 优先级：信号文件 > 配置文件
		/// </summary>
		private GameMode DetermineMode()
		{
			// 1. 检查信号文件（优先级最高）
			if (SignalFileHelper.CheckSignal(SIGNAL_DATA_COLLECTION))
			{
				Debug.Log("📄 检测到数据采集模式信号文件");
				SignalFileHelper.DeleteSignal(SIGNAL_DATA_COLLECTION);
				return GameMode.DataCollection;
			}

			if (SignalFileHelper.CheckSignal(SIGNAL_GAME_RUNTIME))
			{
				Debug.Log("📄 检测到游戏运行模式信号文件");
				SignalFileHelper.DeleteSignal(SIGNAL_GAME_RUNTIME);
				return GameMode.GameRuntime;
			}

			if (SignalFileHelper.CheckSignal(SIGNAL_GAME_TESTING))
			{
				Debug.Log("📄 检测到游戏测试模式信号文件");
				SignalFileHelper.DeleteSignal(SIGNAL_GAME_TESTING);
				return GameMode.GameTesting;
			}

			// 2. 使用配置文件中的模式
			Debug.Log($"📋 使用配置文件模式: {config.mode}");
			return config.mode;
		}

		/// <summary>
		/// 切换模式（运行时切换，如果需要）
		/// </summary>
		public void SwitchMode(GameMode newMode)
		{
			if (currentMode == newMode)
			{
				Debug.LogWarning($"已经处于 {GetModeDisplayName(newMode)} 模式");
				return;
			}

			GameMode oldMode = currentMode;
			currentMode = newMode;

			Debug.Log($"🔄 模式切换: {GetModeDisplayName(oldMode)} → {GetModeDisplayName(newMode)}");

			// 发布模式切换事件（如果需要其他模块响应）
			EventBus.Publish("mode.switched", new { oldMode, newMode });
		}

		/// <summary>
		/// 是否为数据采集模式
		/// </summary>
		public bool IsDataCollectionMode()
		{
			return currentMode == GameMode.DataCollection;
		}

		/// <summary>
		/// 是否为游戏运行模式
		/// </summary>
		public bool IsGameRuntimeMode()
		{
			return currentMode == GameMode.GameRuntime;
		}

		/// <summary>
		/// 是否为游戏测试模式
		/// </summary>
		public bool IsGameTestingMode()
		{
			return currentMode == GameMode.GameTesting;
		}

		/// <summary>
		/// 获取模式显示名称
		/// </summary>
		public static string GetModeDisplayName(GameMode mode)
		{
			switch (mode)
			{
				case GameMode.DataCollection:
					return "数据采集模式";
				case GameMode.GameRuntime:
					return "游戏运行模式";
				case GameMode.GameTesting:
					return "游戏测试模式";
				default:
					return "未知模式";
			}
		}

		/// <summary>
		/// 获取模式描述
		/// </summary>
		public static string GetModeDescription(GameMode mode)
		{
			switch (mode)
			{
				case GameMode.DataCollection:
					return "前期数据采集（Mountain 碰撞箱等）";
				case GameMode.GameRuntime:
					return "AI 训练交互（UDP 通信）";
				case GameMode.GameTesting:
					return "游戏交互性测试";
				default:
					return "未知模式";
			}
		}

		/// <summary>
		/// 打印模式信息
		/// </summary>
		public void PrintModeInfo()
		{
			Debug.Log("=== 模式信息 ===");
			Debug.Log($"当前模式: {GetModeDisplayName(currentMode)}");
			Debug.Log($"描述: {GetModeDescription(currentMode)}");
			Debug.Log($"配置模式: {GetModeDisplayName(config.mode)}");
		}

		/// <summary>
		/// 创建模式信号文件（用于 Python 控制）
		/// </summary>
		public static void CreateModeSignal(GameMode mode)
		{
			string signalName = GetModeSignalName(mode);
			SignalFileHelper.CreateSignal(signalName);
			Debug.Log($"✅ 已创建模式信号: {signalName}");
		}

		/// <summary>
		/// 获取模式信号文件名
		/// </summary>
		private static string GetModeSignalName(GameMode mode)
		{
			switch (mode)
			{
				case GameMode.DataCollection:
					return SIGNAL_DATA_COLLECTION;
				case GameMode.GameRuntime:
					return SIGNAL_GAME_RUNTIME;
				case GameMode.GameTesting:
					return SIGNAL_GAME_TESTING;
				default:
					return "mode_unknown";
			}
		}

		/// <summary>
		/// 清除所有模式信号文件
		/// </summary>
		public static void ClearAllModeSignals()
		{
			SignalFileHelper.DeleteSignal(SIGNAL_DATA_COLLECTION);
			SignalFileHelper.DeleteSignal(SIGNAL_GAME_RUNTIME);
			SignalFileHelper.DeleteSignal(SIGNAL_GAME_TESTING);
			Debug.Log("🧹 已清除所有模式信号文件");
		}
	}
}

