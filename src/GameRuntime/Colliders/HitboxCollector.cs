using UnityEngine;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace GoiHitboxLogger
{
	/// <summary>
	/// 碰撞体收集器 - 专门用于收集和管理游戏中的PolygonCollider2D
	/// </summary>
	public static class HitboxCollector
	{
		/// <summary>
		/// 碰撞体信息结构
		/// </summary>
		public struct ColliderInfo
		{
			public string name;
			public string fullPath;
			public Vector3 position;
			public PolygonCollider2D collider;
			public int vertexCount;
			public bool isTrigger;
			public Bounds bounds;
		}
		
		/// <summary>
		/// 获取所有Mountain下的PolygonCollider2D对象
		/// </summary>
		/// <returns>碰撞体信息列表</returns>
		public static List<ColliderInfo> GetAllMountainColliders()
		{
			List<ColliderInfo> colliders = new List<ColliderInfo>();
			
			// 查找Mountain父对象
			GameObject mountain = GameObject.Find("Mountain");
			if (mountain == null)
			{
				Debug.LogWarning("未找到Mountain对象");
				return colliders;
			}
			
			// 递归搜索所有子对象中的PolygonCollider2D
			SearchColliders(mountain, colliders);
			
			Debug.Log($"在Mountain下找到 {colliders.Count} 个PolygonCollider2D对象");
			return colliders;
		}
		
		/// <summary>
		/// 获取指定GameObject下的所有PolygonCollider2D
		/// </summary>
		/// <param name="parent">父对象</param>
		/// <returns>碰撞体信息列表</returns>
		public static List<ColliderInfo> GetCollidersInGameObject(GameObject parent)
		{
			List<ColliderInfo> colliders = new List<ColliderInfo>();
			if (parent == null) return colliders;
			
			SearchColliders(parent, colliders);
			return colliders;
		}
		
		/// <summary>
		/// 递归搜索PolygonCollider2D组件
		/// </summary>
		/// <param name="current">当前对象</param>
		/// <param name="colliders">结果列表</param>
		private static void SearchColliders(GameObject current, List<ColliderInfo> colliders)
		{
			// 检查当前对象
			PolygonCollider2D polygonCollider = current.GetComponent<PolygonCollider2D>();
			if (polygonCollider != null)
			{
				ColliderInfo info = new ColliderInfo
				{
					name = current.name,
					fullPath = GetGameObjectPath(current),
					position = current.transform.position,
					collider = polygonCollider,
					vertexCount = polygonCollider.GetTotalPointCount(),
					isTrigger = polygonCollider.isTrigger,
					bounds = polygonCollider.bounds
				};
				
				colliders.Add(info);
			}
			
			// 递归搜索子对象
			for (int i = 0; i < current.transform.childCount; i++)
			{
				GameObject child = current.transform.GetChild(i).gameObject;
				SearchColliders(child, colliders);
			}
		}
		
		/// <summary>
		/// 获取GameObject的完整路径
		/// </summary>
		/// <param name="obj">目标对象</param>
		/// <returns>完整路径字符串</returns>
		public static string GetGameObjectPath(GameObject obj)
		{
			string path = obj.name;
			Transform parent = obj.transform.parent;
			
			while (parent != null)
			{
				path = parent.name + "/" + path;
				parent = parent.parent;
			}
			
			return path;
		}
		
		/// <summary>
		/// 导出碰撞体数据到文件
		/// </summary>
		/// <param name="colliders">碰撞体列表</param>
		/// <param name="fileName">文件名（可选）</param>
		public static void ExportCollidersToFile(List<ColliderInfo> colliders, string fileName = null)
		{
			if (fileName == null)
			{
				fileName = $"MountainColliders_{System.DateTime.Now:yyyyMMdd_HHmmss}.txt";
			}
			
			try
			{
				StringBuilder exportData = new StringBuilder();
				exportData.AppendLine("=== Mountain PolygonCollider2D 数据导出 ===");
				exportData.AppendLine($"导出时间: {System.DateTime.Now}");
				exportData.AppendLine($"碰撞体总数: {colliders.Count}");
				exportData.AppendLine();
				
				for (int i = 0; i < colliders.Count; i++)
				{
					ColliderInfo info = colliders[i];
					exportData.AppendLine($"[{i + 1}] {info.name}");
					exportData.AppendLine($"  路径: {info.fullPath}");
					exportData.AppendLine($"  位置: {info.position}");
					exportData.AppendLine($"  顶点数: {info.vertexCount}");
					exportData.AppendLine($"  触发器: {info.isTrigger}");
					exportData.AppendLine($"  边界: 中心{info.bounds.center} 大小{info.bounds.size}");
					
					// 导出详细顶点数据
					for (int pathIndex = 0; pathIndex < info.collider.pathCount; pathIndex++)
					{
						Vector2[] path = info.collider.GetPath(pathIndex);
						exportData.AppendLine($"  路径[{pathIndex}] 顶点数: {path.Length}");
						
						for (int vertexIndex = 0; vertexIndex < path.Length; vertexIndex++)
						{
							Vector3 worldPos = info.collider.transform.TransformPoint(path[vertexIndex]);
							exportData.AppendLine($"    顶点[{vertexIndex}]: 本地{path[vertexIndex]} → 世界{worldPos}");
						}
					}
					
					exportData.AppendLine();
				}
				
				// 保存到文件
				string baseDir = Path.Combine(Application.dataPath, "..");
				string hitboxDir = Path.Combine(baseDir, "HitboxDump");
				string filePath = Path.Combine(hitboxDir, fileName);
				
				Directory.CreateDirectory(Path.GetDirectoryName(filePath));
				File.WriteAllText(filePath, exportData.ToString());
				
				Debug.Log($"✅ 碰撞体数据已导出到: {filePath}");
			}
			catch (System.Exception e)
			{
				Debug.LogError($"❌ 导出碰撞体数据失败: {e.Message}");
			}
		}
		
		/// <summary>
		/// 查找指定名称的碰撞体
		/// </summary>
		/// <param name="colliders">碰撞体列表</param>
		/// <param name="namePattern">名称模式（支持Contains匹配）</param>
		/// <returns>匹配的碰撞体列表</returns>
		public static List<ColliderInfo> FindCollidersByName(List<ColliderInfo> colliders, string namePattern)
		{
			List<ColliderInfo> matches = new List<ColliderInfo>();
			string pattern = namePattern.ToLower();
			
			foreach (ColliderInfo info in colliders)
			{
				if (info.name.ToLower().Contains(pattern))
				{
					matches.Add(info);
				}
			}
			
			return matches;
		}
		
		/// <summary>
		/// 按顶点数量排序碰撞体
		/// </summary>
		/// <param name="colliders">碰撞体列表</param>
		/// <param name="ascending">是否升序</param>
		/// <returns>排序后的列表</returns}
		public static List<ColliderInfo> SortCollidersByVertexCount(List<ColliderInfo> colliders, bool ascending = true)
		{
			List<ColliderInfo> sorted = new List<ColliderInfo>(colliders);
			
			if (ascending)
			{
				sorted.Sort((a, b) => a.vertexCount.CompareTo(b.vertexCount));
			}
			else
			{
				sorted.Sort((a, b) => b.vertexCount.CompareTo(a.vertexCount));
			}
			
			return sorted;
		}
		
		/// <summary>
		/// 打印碰撞体统计信息
		/// </summary>
		/// <param name="colliders">碰撞体列表</param>
		public static void PrintColliderStatistics(List<ColliderInfo> colliders)
		{
			if (colliders.Count == 0)
			{
				Debug.Log("没有找到任何碰撞体");
				return;
			}
			
			int totalVertices = 0;
			int triggerCount = 0;
			float averageVertices = 0;
			
			foreach (ColliderInfo info in colliders)
			{
				totalVertices += info.vertexCount;
				if (info.isTrigger) triggerCount++;
			}
			
			averageVertices = (float)totalVertices / colliders.Count;
			
			Debug.Log("=== 碰撞体统计信息 ===");
			Debug.Log($"总数量: {colliders.Count}");
			Debug.Log($"总顶点数: {totalVertices}");
			Debug.Log($"平均顶点数: {averageVertices:F1}");
			Debug.Log($"触发器数量: {triggerCount}");
			Debug.Log($"普通碰撞体: {colliders.Count - triggerCount}");
		}
	}
}
