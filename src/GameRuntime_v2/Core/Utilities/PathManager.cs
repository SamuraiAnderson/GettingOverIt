using System;
using System.IO;
using UnityEngine;

namespace GoiRuntime.Core.Utilities
{
	/// <summary>
	/// 路径管理工具
	/// 统一管理所有数据路径
	/// </summary>
	public static class PathManager
	{
		/// <summary>
		/// 游戏根目录（Getting Over It.exe 所在目录）
		/// </summary>
		public static string GameRootPath
		{
			get
			{
				// Unity Editor: 返回项目根目录
				// 游戏运行时: 返回 exe 所在目录
				#if UNITY_EDITOR
				return Application.dataPath.Replace("/Assets", "");
				#else
				return Path.GetDirectoryName(Application.dataPath);
				#endif
			}
		}

		/// <summary>
		/// 游戏数据根目录（GoiData）
		/// </summary>
		public static string DataRootPath => Path.Combine(GameRootPath, "GoiData");

		/// <summary>
		/// 游戏结果目录（训练数据）
		/// </summary>
		public static string GameResultsPath => Path.Combine(DataRootPath, "GameResults");

		/// <summary>
		/// 调试数据目录
		/// </summary>
		public static string DebugPath => Path.Combine(DataRootPath, "Debug");

		/// <summary>
		/// 状态快照目录
		/// </summary>
		public static string StatesPath => Path.Combine(DebugPath, "States");

		/// <summary>
		/// 动作快照目录
		/// </summary>
		public static string ActionsPath => Path.Combine(DebugPath, "Actions");

		/// <summary>
		/// 碰撞体数据目录
		/// </summary>
		public static string CollidersPath => Path.Combine(DebugPath, "Colliders");

		/// <summary>
		/// 探索报告目录
		/// </summary>
		public static string ExplorationPath => Path.Combine(DataRootPath, "Exploration");

		/// <summary>
		/// 测试信号目录
		/// </summary>
		public static string TestSignalsPath => Path.Combine(DataRootPath, "TestSignals");

		/// <summary>
		/// 控制信号目录
		/// </summary>
		public static string ControlSignalsPath => Path.Combine(DataRootPath, "ControlSignals");

		/// <summary>
		/// 环境状态目录
		/// </summary>
		public static string EnvironmentStatesPath => Path.Combine(DataRootPath, "EnvironmentStates");

		/// <summary>
		/// 确保目录存在
		/// </summary>
		public static void EnsureDirectory(string path)
		{
			if (!Directory.Exists(path))
			{
				Directory.CreateDirectory(path);
				Debug.Log($"创建目录: {path}");
			}
		}

		/// <summary>
		/// 初始化所有必要的目录
		/// </summary>
		public static void InitializeAllDirectories()
		{
			EnsureDirectory(DataRootPath);
			EnsureDirectory(GameResultsPath);
			EnsureDirectory(DebugPath);
			EnsureDirectory(StatesPath);
			EnsureDirectory(ActionsPath);
			EnsureDirectory(CollidersPath);
			EnsureDirectory(ExplorationPath);
			EnsureDirectory(TestSignalsPath);
			EnsureDirectory(ControlSignalsPath);
			EnsureDirectory(EnvironmentStatesPath);
		}

		/// <summary>
		/// 获取带时间戳的文件名
		/// </summary>
		public static string GetTimestampedFileName(string baseName, string extension)
		{
			string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
			return $"{baseName}_{timestamp}.{extension}";
		}

		/// <summary>
		/// 获取最新文件路径（_latest 后缀）
		/// </summary>
		public static string GetLatestFilePath(string directory, string baseName, string extension)
		{
			EnsureDirectory(directory);
			return Path.Combine(directory, $"{baseName}_latest.{extension}");
		}

		/// <summary>
		/// 获取时间戳文件路径
		/// </summary>
		public static string GetTimestampedFilePath(string directory, string baseName, string extension)
		{
			EnsureDirectory(directory);
			string fileName = GetTimestampedFileName(baseName, extension);
			return Path.Combine(directory, fileName);
		}

		/// <summary>
		/// 保存文件并创建最新副本
		/// </summary>
		public static void SaveWithLatestCopy(string content, string directory, string baseName, string extension)
		{
			// 保存时间戳版本
			string timestampedPath = GetTimestampedFilePath(directory, baseName, extension);
			File.WriteAllText(timestampedPath, content);

			// 保存最新版本
			string latestPath = GetLatestFilePath(directory, baseName, extension);
			File.WriteAllText(latestPath, content);

			Debug.Log($"文件已保存: {timestampedPath}");
		}

		/// <summary>
		/// 获取相对路径（相对于游戏根目录）
		/// </summary>
		public static string GetRelativePath(string fullPath)
		{
			if (fullPath.StartsWith(GameRootPath))
			{
				return fullPath.Substring(GameRootPath.Length).TrimStart(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
			}
			return fullPath;
		}
	}
}

