using UnityEngine;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Globalization;
using GoiRuntime.Core.Interfaces;
using GoiRuntime.Core.Utilities;

namespace GoiRuntime.ColliderCollection
{
	/// <summary>
	/// 碰撞箱采集器基类
	/// 提供通用的碰撞箱采集功能
	/// </summary>
	public class ColliderCollectorBase : IColliderService
	{
		private static readonly CultureInfo InvariantCulture = CultureInfo.InvariantCulture;

		protected bool isInitialized = false;
		protected List<ColliderData> cachedColliders = new List<ColliderData>();

		#region IColliderService 实现

		public virtual bool Initialize()
		{
			isInitialized = true;
			return true;
		}

		public bool IsReady => isInitialized;

		public virtual ColliderData[] CollectEnvironmentColliders()
		{
			return new ColliderData[0];
		}

		public virtual ColliderData[] CollectPlayerColliders(GameObject player)
		{
			return new ColliderData[0];
		}

		public int GetColliderCount()
		{
			return cachedColliders.Count;
		}

		public Bounds GetCombinedBounds()
		{
			if (cachedColliders.Count == 0)
			{
				return new Bounds(Vector3.zero, Vector3.zero);
			}

			Bounds combined = cachedColliders[0].bounds;
			for (int i = 1; i < cachedColliders.Count; i++)
			{
				combined.Encapsulate(cachedColliders[i].bounds);
			}

			return combined;
		}

		public virtual string ExportToFile(string filePath)
		{
			return ExportColliders(cachedColliders, filePath);
		}

		#endregion

		#region 通用采集方法

		/// <summary>
		/// 从指定 GameObject 采集所有碰撞箱
		/// </summary>
		protected List<ColliderData> CollectFromGameObject(GameObject parent, bool includeChildren = true)
		{
			List<ColliderData> colliders = new List<ColliderData>();
			
			if (parent == null)
			{
				Debug.LogWarning("采集目标为 null");
				return colliders;
			}

			if (includeChildren)
			{
				SearchCollidersRecursive(parent, colliders);
			}
			else
			{
				CollectFromSingle(parent, colliders);
			}

			Debug.Log($"从 {parent.name} 采集到 {colliders.Count} 个碰撞箱");
			return colliders;
		}

		/// <summary>
		/// 递归搜索碰撞箱
		/// </summary>
		private void SearchCollidersRecursive(GameObject current, List<ColliderData> colliders)
		{
			// 采集当前对象
			CollectFromSingle(current, colliders);

			// 递归搜索子对象
			for (int i = 0; i < current.transform.childCount; i++)
			{
				GameObject child = current.transform.GetChild(i).gameObject;
				SearchCollidersRecursive(child, colliders);
			}
		}

		/// <summary>
		/// 从单个对象采集碰撞箱
		/// </summary>
		private void CollectFromSingle(GameObject obj, List<ColliderData> colliders)
		{
			PolygonCollider2D polygonCollider = obj.GetComponent<PolygonCollider2D>();
			if (polygonCollider != null)
			{
				ColliderData data = CreateColliderData(obj, polygonCollider);
				colliders.Add(data);
			}
		}

		/// <summary>
		/// 创建碰撞箱数据
		/// </summary>
		protected ColliderData CreateColliderData(GameObject obj, PolygonCollider2D collider)
		{
			// 获取世界坐标顶点
			Vector2[] worldVertices = GetWorldVertices(collider);

			ColliderData data = new ColliderData
			{
				name = obj.name,
				worldVertices = worldVertices,
				bounds = collider.bounds,
				vertexCount = collider.GetTotalPointCount(),
				isTrigger = collider.isTrigger,
				position = obj.transform.position
			};

			return data;
		}

		/// <summary>
		/// 获取世界坐标顶点
		/// </summary>
		protected Vector2[] GetWorldVertices(PolygonCollider2D collider)
		{
			List<Vector2> allVertices = new List<Vector2>();

			// 遍历所有路径
			for (int pathIndex = 0; pathIndex < collider.pathCount; pathIndex++)
			{
				Vector2[] path = collider.GetPath(pathIndex);
				
				// 转换为世界坐标
				foreach (Vector2 localVertex in path)
				{
					Vector3 worldPos = collider.transform.TransformPoint(localVertex);
					allVertices.Add(new Vector2(worldPos.x, worldPos.y));
				}
			}

			return allVertices.ToArray();
		}

		#endregion

		#region 数据导出

		/// <summary>
		/// 导出碰撞箱数据到文件（JSON 格式，方便程序读取）
		/// </summary>
		protected string ExportColliders(List<ColliderData> colliders, string filePath)
		{
			try
			{
				StringBuilder sb = new StringBuilder();
				sb.AppendLine("{");
				sb.AppendLine($"  \"exportedAt\": \"{System.DateTime.Now.ToString("yyyy-MM-ddTHH:mm:ss.fffZ", InvariantCulture)}\",");
				sb.AppendLine($"  \"colliderCount\": {colliders.Count},");
				sb.AppendLine("  \"colliders\": [");

				for (int i = 0; i < colliders.Count; i++)
				{
					ColliderData data = colliders[i];
					sb.AppendLine("    {");
					sb.AppendLine($"      \"index\": {i},");
					sb.AppendLine($"      \"name\": \"{EscapeJson(data.name)}\",");
					sb.AppendLine($"      \"position\": {Vector3ToJson(data.position)},");
					sb.AppendLine($"      \"vertexCount\": {data.vertexCount},");
					sb.AppendLine($"      \"isTrigger\": {data.isTrigger.ToString().ToLowerInvariant()},");
					sb.AppendLine("      \"bounds\": {");
					sb.AppendLine($"        \"center\": {Vector3ToJson(data.bounds.center)},");
					sb.AppendLine($"        \"size\": {Vector3ToJson(data.bounds.size)}");
					sb.AppendLine("      },");
					sb.AppendLine("      \"worldVertices\": [");

					Vector2[] vertices = data.worldVertices ?? new Vector2[0];
					for (int j = 0; j < vertices.Length; j++)
					{
						string comma = j < vertices.Length - 1 ? "," : "";
						sb.AppendLine($"        {Vector2ToJson(vertices[j])}{comma}");
					}

					sb.AppendLine("      ]");
					sb.Append("    }");
					sb.AppendLine(i < colliders.Count - 1 ? "," : "");
				}

				sb.AppendLine("  ],");
				sb.AppendLine("  \"stats\": {");
				sb.AppendLine($"    \"totalVertexCount\": {GetTotalVertexCount(colliders)},");
				sb.AppendLine($"    \"averageVertexCount\": {GetAverageVertexCount(colliders).ToString("F2", InvariantCulture)},");
				sb.AppendLine($"    \"triggerCount\": {GetTriggerCount(colliders)}");
				sb.AppendLine("  }");
				sb.AppendLine("}");

				PathManager.EnsureDirectory(Path.GetDirectoryName(filePath));
				File.WriteAllText(filePath, sb.ToString());

				Debug.Log($"碰撞箱数据已导出: {filePath}");
				return filePath;
			}
			catch (System.Exception e)
			{
				Debug.LogError($"导出碰撞箱数据失败: {e.Message}");
				return null;
			}
		}

		#endregion

		#region JSON 帮助方法

		protected string Vector2ToJson(Vector2 value)
		{
			return string.Format(InvariantCulture, "[{0:F6},{1:F6}]", value.x, value.y);
		}

		protected string Vector3ToJson(Vector3 value)
		{
			return string.Format(InvariantCulture, "[{0:F6},{1:F6},{2:F6}]", value.x, value.y, value.z);
		}

		protected string EscapeJson(string value)
		{
			if (string.IsNullOrEmpty(value)) return string.Empty;
			return value.Replace("\\", "\\\\").Replace("\"", "\\\"");
		}

		#endregion

		#region 统计方法

		protected int GetTotalVertexCount(List<ColliderData> colliders)
		{
			int total = 0;
			foreach (var c in colliders)
			{
				total += c.vertexCount;
			}
			return total;
		}

		protected float GetAverageVertexCount(List<ColliderData> colliders)
		{
			if (colliders.Count == 0) return 0f;
			return (float)GetTotalVertexCount(colliders) / colliders.Count;
		}

		protected int GetTriggerCount(List<ColliderData> colliders)
		{
			int count = 0;
			foreach (var c in colliders)
			{
				if (c.isTrigger) count++;
			}
			return count;
		}

		#endregion
	}
}

