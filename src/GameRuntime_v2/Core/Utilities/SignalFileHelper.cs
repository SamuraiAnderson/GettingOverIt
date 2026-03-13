using System;
using System.IO;
using UnityEngine;

namespace GoiRuntime.Core.Utilities
{
	/// <summary>
	/// 信号文件辅助工具
	/// 用于基于文件的进程间通信和测试控制
	/// </summary>
	public static class SignalFileHelper
	{
		/// <summary>
		/// 检查信号文件是否存在
		/// </summary>
		public static bool CheckSignal(string signalName)
		{
			string signalPath = GetSignalPath(signalName);
			return File.Exists(signalPath);
		}

		/// <summary>
		/// 检查信号文件并读取内容
		/// </summary>
		/// <param name="signalName">信号名称（如 "pause_input.signal"）</param>
		/// <param name="content">信号内容</param>
		/// <returns>信号是否存在</returns>
		public static bool CheckSignal(string signalName, out string content)
		{
			content = null;
			string signalPath = GetSignalPath(signalName);
			
			if (!File.Exists(signalPath))
			{
				return false;
			}

			try
			{
				content = File.ReadAllText(signalPath).Trim();
				return true;
			}
			catch (Exception e)
			{
				Debug.LogError($"读取信号文件失败 [{signalName}]: {e.Message}");
				return false;
			}
		}

		/// <summary>
		/// 创建信号文件
		/// </summary>
		public static void CreateSignal(string signalName, string content = "")
		{
			string signalPath = GetSignalPath(signalName);
			
			try
			{
				PathManager.EnsureDirectory(PathManager.ControlSignalsPath);
				File.WriteAllText(signalPath, content);
				Debug.Log($"创建信号: {signalName}");
			}
			catch (Exception e)
			{
				Debug.LogError($"创建信号文件失败 [{signalName}]: {e.Message}");
			}
		}

		/// <summary>
		/// 删除信号文件
		/// </summary>
		public static void DeleteSignal(string signalName)
		{
			string signalPath = GetSignalPath(signalName);
			
			if (File.Exists(signalPath))
			{
				try
				{
					File.Delete(signalPath);
					Debug.Log($"删除信号: {signalName}");
				}
				catch (Exception e)
				{
					Debug.LogError($"删除信号文件失败 [{signalName}]: {e.Message}");
				}
			}
		}

		/// <summary>
		/// 处理并删除信号（原子操作）
		/// </summary>
		public static bool ConsumeSignal(string signalName, out string content)
		{
			bool exists = CheckSignal(signalName, out content);
			if (exists)
			{
				DeleteSignal(signalName);
			}
			return exists;
		}

		/// <summary>
		/// 检查并解析浮点数信号
		/// </summary>
		public static bool CheckFloatSignal(string signalName, out float value)
		{
			value = 0f;
			if (CheckSignal(signalName, out string content))
			{
				return float.TryParse(content, out value);
			}
			return false;
		}

		/// <summary>
		/// 检查并解析整数信号
		/// </summary>
		public static bool CheckIntSignal(string signalName, out int value)
		{
			value = 0;
			if (CheckSignal(signalName, out string content))
			{
				return int.TryParse(content, out value);
			}
			return false;
		}

		/// <summary>
		/// 检查并解析布尔信号
		/// </summary>
		public static bool CheckBoolSignal(string signalName, out bool value)
		{
			value = false;
			if (CheckSignal(signalName, out string content))
			{
				content = content.ToLower();
				if (content == "true" || content == "1" || content == "yes")
				{
					value = true;
					return true;
				}
				else if (content == "false" || content == "0" || content == "no")
				{
					value = false;
					return true;
				}
			}
			return false;
		}

		/// <summary>
		/// 获取信号文件的完整路径
		/// </summary>
		private static string GetSignalPath(string signalName)
		{
			// 确保信号名称以 .signal 结尾
			if (!signalName.EndsWith(".signal"))
			{
				signalName += ".signal";
			}

			return Path.Combine(PathManager.ControlSignalsPath, signalName);
		}

		/// <summary>
		/// 清除所有信号文件
		/// </summary>
		public static void ClearAllSignals()
		{
			try
			{
				if (Directory.Exists(PathManager.ControlSignalsPath))
				{
					string[] signalFiles = Directory.GetFiles(PathManager.ControlSignalsPath, "*.signal");
					foreach (string file in signalFiles)
					{
						File.Delete(file);
					}
					Debug.Log($"清除了 {signalFiles.Length} 个信号文件");
				}
			}
			catch (Exception e)
			{
				Debug.LogError($"清除信号文件失败: {e.Message}");
			}
		}

		/// <summary>
		/// 列出所有活跃的信号
		/// </summary>
		public static string[] ListActiveSignals()
		{
			try
			{
				if (Directory.Exists(PathManager.ControlSignalsPath))
				{
					string[] files = Directory.GetFiles(PathManager.ControlSignalsPath, "*.signal");
					for (int i = 0; i < files.Length; i++)
					{
						files[i] = Path.GetFileName(files[i]);
					}
					return files;
				}
			}
			catch (Exception e)
			{
				Debug.LogError($"列出信号文件失败: {e.Message}");
			}

			return new string[0];
		}
	}
}

