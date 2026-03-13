using System;
using System.IO;
using UnityEngine;

namespace GoiRuntime.Core.Configuration
{
	/// <summary>
	/// 游戏运行模式
	/// </summary>
	public enum GameMode
	{
		/// <summary>
		/// 数据采集模式（前期采集 Mountain 碰撞箱等静态数据）
		/// </summary>
		DataCollection,

		/// <summary>
		/// 游戏运行模式（AI 训练交互）
		/// </summary>
		GameRuntime,

		/// <summary>
		/// 游戏测试模式（交互性测试）
		/// </summary>
		GameTesting
	}

	/// <summary>
	/// 运行时配置
	/// 包含所有可配置的参数
	/// </summary>
	[Serializable]
	public class RuntimeConfig
	{
		#region 模式配置

		/// <summary>
		/// 当前运行模式
		/// </summary>
		public GameMode mode = GameMode.DataCollection;

		#endregion

		#region 通信配置

		/// <summary>
		/// TCP 步进服务器监听端口（Python 连接此端口进行帧级别交互）
		/// </summary>
		public int tcpPort = 9000;

		#endregion

		#region 数据配置

		/// <summary>
		/// 状态维度（每个复制体的浮点数数量）
		/// </summary>
		public int stateDimension = 29;

		/// <summary>
		/// 动作维度（每个复制体的动作数量）
		/// </summary>
		public int actionDimension = 2;

		/// <summary>
		/// 每个 RL step 推进的物理帧数
		/// </summary>
		public int stepFrames = 1;

		#endregion

		#region Player 配置

		/// <summary>
		/// 复制体数量（从注册表读取，此为默认值）
		/// </summary>
		public int numDuplicates = 1;

		/// <summary>
		/// Player 对象名称
		/// </summary>
		public string playerObjectName = "Player";

		#endregion

		#region 输入配置

		/// <summary>
		/// 最小有效输入值
		/// </summary>
		public float minInput = 10f;

		/// <summary>
		/// 最大有效输入值
		/// </summary>
		public float maxInput = 100f;

		/// <summary>
		/// 是否自动限制输入
		/// </summary>
		public bool autoClampInput = true;

		#endregion

		#region 路径配置

		/// <summary>
		/// 数据根目录名称
		/// </summary>
		public string dataRoot = "GoiData";

		/// <summary>
		/// 是否在启动时初始化所有目录
		/// </summary>
		public bool initializeDirectoriesOnStart = true;

		#endregion

		#region 调试配置

		/// <summary>
		/// 是否启用调试日志
		/// </summary>
		public bool enableDebugLogs = true;

		#endregion

		#region 静态方法

		/// <summary>
		/// 获取配置文件路径
		/// </summary>
		private static string ConfigFilePath
		{
			get
			{
				return Path.Combine(
					GoiRuntime.Core.Utilities.PathManager.DataRootPath, 
					"runtime_config.json"
				);
			}
		}

		/// <summary>
		/// 加载配置
		/// </summary>
		public static RuntimeConfig Load()
		{
			string configPath = ConfigFilePath;

			// 如果配置文件存在，则加载
			if (File.Exists(configPath))
			{
				try
				{
					string json = File.ReadAllText(configPath);
					RuntimeConfig config = JsonUtility.FromJson<RuntimeConfig>(json);
					Debug.Log($"配置已加载: {configPath}");
					return config;
				}
				catch (Exception e)
				{
					Debug.LogError($"加载配置失败: {e.Message}，使用默认配置");
				}
			}
			else
			{
				Debug.Log("配置文件不存在，使用默认配置");
			}

			// 返回默认配置
			RuntimeConfig defaultConfig = new RuntimeConfig();
			defaultConfig.Save();  // 保存默认配置
			return defaultConfig;
		}

		/// <summary>
		/// 保存配置
		/// </summary>
		public void Save()
		{
			try
			{
				GoiRuntime.Core.Utilities.PathManager.EnsureDirectory(
					GoiRuntime.Core.Utilities.PathManager.DataRootPath
				);

				string json = JsonUtility.ToJson(this, true);
				File.WriteAllText(ConfigFilePath, json);
				Debug.Log($"配置已保存: {ConfigFilePath}");
			}
			catch (Exception e)
			{
				Debug.LogError($"保存配置失败: {e.Message}");
			}
		}

		/// <summary>
		/// 从注册表更新复制体数量
		/// </summary>
		public void UpdateDuplicateCountFromRegistry()
		{
			#if !UNITY_EDITOR && UNITY_STANDALONE_WIN
			try
			{
				using (Microsoft.Win32.RegistryKey key = Microsoft.Win32.Registry.CurrentUser.OpenSubKey(
					@"Software\Bennett Foddy\Getting Over It"))
				{
					if (key != null)
					{
						object value = key.GetValue("numDuplis_h1216609508");
						if (value != null)
						{
							numDuplicates = (int)value;
							Debug.Log($"从注册表读取复制体数量: {numDuplicates}");
						}
					}
				}
			}
			catch (Exception e)
			{
				Debug.LogWarning($"读取注册表失败: {e.Message}，使用默认值 {numDuplicates}");
			}
			#else
			Debug.Log($"非 Windows 平台或编辑器模式，使用默认复制体数量: {numDuplicates}");
			#endif
		}

		#endregion

		#region 辅助方法

		/// <summary>
		/// 获取总状态维度（所有复制体）
		/// </summary>
		public int GetTotalStateDimension()
		{
			return stateDimension * numDuplicates;
		}

		/// <summary>
		/// 获取总动作维度（所有复制体）
		/// </summary>
		public int GetTotalActionDimension()
		{
			return actionDimension * numDuplicates;
		}

		#endregion
	}
}

