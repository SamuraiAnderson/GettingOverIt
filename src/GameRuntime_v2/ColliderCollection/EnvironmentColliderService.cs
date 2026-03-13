using UnityEngine;
using System.IO;
using GoiRuntime.Core.Interfaces;
using GoiRuntime.Core.Utilities;
using GoiRuntime.Core.Events;

namespace GoiRuntime.ColliderCollection
{
	/// <summary>
	/// 环境碰撞箱服务
	/// 负责采集游戏环境中的静态碰撞箱（如 Mountain）
	/// </summary>
	public class EnvironmentColliderService : ColliderCollectorBase
	{
		private GameObject mountainObject;
		private const string MOUNTAIN_NAME = "Mountain";

		/// <summary>
		/// 初始化服务
		/// </summary>
		public override bool Initialize()
		{
			// 查找 Mountain 对象
			mountainObject = GameObject.Find(MOUNTAIN_NAME);
			
			if (mountainObject == null)
			{
				Debug.LogWarning($"未找到 {MOUNTAIN_NAME} 对象");
				return false;
			}

			Debug.Log($"EnvironmentColliderService 初始化成功，找到 {MOUNTAIN_NAME}");
			isInitialized = true;
			return true;
		}

		/// <summary>
		/// 采集环境碰撞箱（Mountain）
		/// </summary>
		public override ColliderData[] CollectEnvironmentColliders()
		{
			if (!IsReady)
			{
				Debug.LogWarning("EnvironmentColliderService 未初始化");
				return new ColliderData[0];
			}

			// 采集 Mountain 下的所有碰撞箱
			cachedColliders = CollectFromGameObject(mountainObject, includeChildren: true);

			// 发布事件
			EventBus.Publish(GameEvents.CollidersCollected, cachedColliders.ToArray());

			return cachedColliders.ToArray();
		}

		/// <summary>
		/// 导出环境碰撞箱到文件
		/// </summary>
		public string ExportEnvironmentColliders()
		{
			if (cachedColliders.Count == 0)
			{
				Debug.LogWarning("没有可导出的碰撞箱数据，请先调用 CollectEnvironmentColliders()");
				return null;
			}

			// 生成文件路径
			string fileName = PathManager.GetTimestampedFileName("MountainColliders", "txt");
			string filePath = Path.Combine(PathManager.CollidersPath, fileName);

			// 导出
			string exportedPath = ExportToFile(filePath);

			if (exportedPath != null)
			{
				// 同时保存最新版本
				string latestPath = PathManager.GetLatestFilePath(PathManager.CollidersPath, "MountainColliders", "txt");
				File.Copy(exportedPath, latestPath, true);

				// 发布事件
				EventBus.Publish(GameEvents.ColliderDataExported, exportedPath);
			}

			return exportedPath;
		}

		/// <summary>
		/// 采集并导出（一步完成）
		/// </summary>
		public string CollectAndExport()
		{
			CollectEnvironmentColliders();
			return ExportEnvironmentColliders();
		}

		/// <summary>
		/// 打印统计信息
		/// </summary>
		public void PrintStatistics()
		{
			if (cachedColliders.Count == 0)
			{
				Debug.Log("没有碰撞箱数据");
				return;
			}

			Debug.Log("=== Mountain 碰撞箱统计 ===");
			Debug.Log($"总数量: {cachedColliders.Count}");
			Debug.Log($"总顶点数: {GetTotalVertexCount(cachedColliders)}");
			Debug.Log($"平均顶点数: {GetAverageVertexCount(cachedColliders):F1}");
			Debug.Log($"触发器数量: {GetTriggerCount(cachedColliders)}");
			Debug.Log($"边界: {GetCombinedBounds()}");
		}
	}
}

