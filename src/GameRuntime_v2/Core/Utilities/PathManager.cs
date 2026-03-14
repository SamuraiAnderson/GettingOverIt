using System;
using System.IO;
using UnityEngine;

namespace GoiRuntime.Core.Utilities
{
	public static class PathManager
	{
		public static string GameRootPath
		{
			get
			{
				#if UNITY_EDITOR
				return Application.dataPath.Replace("/Assets", "");
				#else
				return Path.GetDirectoryName(Application.dataPath);
				#endif
			}
		}

		public static string DataRootPath => Path.Combine(GameRootPath, "GoiData");

		public static string DebugPath => Path.Combine(DataRootPath, "Debug");

		public static string CollidersPath => Path.Combine(DataRootPath, "Colliders");

		public static string ControlSignalsPath => Path.Combine(DataRootPath, "ControlSignals");

		public static string EnvironmentStatesPath => Path.Combine(DataRootPath, "EnvironmentStates");

		public static void EnsureDirectory(string path)
		{
			if (!Directory.Exists(path))
			{
				Directory.CreateDirectory(path);
				Debug.Log($"创建目录: {path}");
			}
		}

		public static string GetTimestampedFileName(string baseName, string extension)
		{
			string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
			return $"{baseName}_{timestamp}.{extension}";
		}

		public static string GetLatestFilePath(string directory, string baseName, string extension)
		{
			EnsureDirectory(directory);
			return Path.Combine(directory, $"{baseName}_latest.{extension}");
		}

		public static string GetTimestampedFilePath(string directory, string baseName, string extension)
		{
			EnsureDirectory(directory);
			string fileName = GetTimestampedFileName(baseName, extension);
			return Path.Combine(directory, fileName);
		}
	}
}
