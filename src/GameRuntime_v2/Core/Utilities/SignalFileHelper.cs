using System;
using System.IO;
using UnityEngine;

namespace GoiRuntime.Core.Utilities
{
	public static class SignalFileHelper
	{
		public static bool CheckSignal(string signalName)
		{
			string signalPath = GetSignalPath(signalName);
			return File.Exists(signalPath);
		}

		public static bool CheckSignal(string signalName, out string content)
		{
			content = null;
			string signalPath = GetSignalPath(signalName);

			if (!File.Exists(signalPath))
				return false;

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

		private static string GetSignalPath(string signalName)
		{
			if (!signalName.EndsWith(".signal"))
				signalName += ".signal";

			return Path.Combine(PathManager.ControlSignalsPath, signalName);
		}
	}
}
