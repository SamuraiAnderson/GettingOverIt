using UnityEngine;
using System.Text;
using GoiRuntime.Core.Interfaces;

namespace GoiRuntime.ColliderCollection
{
	/// <summary>
	/// 碰撞箱数据格式化工具
	/// 提供多种格式的碰撞箱数据输出
	/// </summary>
	public static class ColliderDataFormatter
	{
		/// <summary>
		/// 格式化为详细文本
		/// </summary>
		public static string FormatDetailed(ColliderData data)
		{
			StringBuilder sb = new StringBuilder();
			
			sb.AppendLine($"名称: {data.name}");
			sb.AppendLine($"位置: {data.position}");
			sb.AppendLine($"顶点数: {data.vertexCount}");
			sb.AppendLine($"触发器: {data.isTrigger}");
			sb.AppendLine($"边界: 中心 {data.bounds.center}, 大小 {data.bounds.size}");
			sb.AppendLine("顶点坐标:");
			
			for (int i = 0; i < data.worldVertices.Length; i++)
			{
				sb.AppendLine($"  [{i}]: {data.worldVertices[i]}");
			}

			return sb.ToString();
		}

		/// <summary>
		/// 格式化为简洁摘要
		/// </summary>
		public static string FormatSummary(ColliderData data)
		{
			return $"{data.name} (顶点: {data.vertexCount}, 位置: {data.position}, 触发器: {data.isTrigger})";
		}

		/// <summary>
		/// 格式化为 CSV 行（不包含顶点）
		/// </summary>
		public static string FormatCsvRow(ColliderData data)
		{
			return $"{data.name},{data.position.x},{data.position.y},{data.vertexCount},{data.isTrigger}";
		}

		/// <summary>
		/// 格式化为 JSON（简化版）
		/// </summary>
		public static string FormatJson(ColliderData data)
		{
			StringBuilder sb = new StringBuilder();
			sb.AppendLine("{");
			sb.AppendLine($"  \"name\": \"{data.name}\",");
			sb.AppendLine($"  \"position\": [{data.position.x}, {data.position.y}, {data.position.z}],");
			sb.AppendLine($"  \"vertexCount\": {data.vertexCount},");
			sb.AppendLine($"  \"isTrigger\": {data.isTrigger.ToString().ToLower()},");
			sb.AppendLine($"  \"bounds\": {{");
			sb.AppendLine($"    \"center\": [{data.bounds.center.x}, {data.bounds.center.y}, {data.bounds.center.z}],");
			sb.AppendLine($"    \"size\": [{data.bounds.size.x}, {data.bounds.size.y}, {data.bounds.size.z}]");
			sb.AppendLine($"  }},");
			sb.Append($"  \"vertices\": [");
			
			for (int i = 0; i < data.worldVertices.Length; i++)
			{
				sb.Append($"[{data.worldVertices[i].x}, {data.worldVertices[i].y}]");
				if (i < data.worldVertices.Length - 1)
					sb.Append(", ");
			}
			
			sb.AppendLine("]");
			sb.AppendLine("}");
			
			return sb.ToString();
		}

		/// <summary>
		/// 格式化多个碰撞箱
		/// </summary>
		public static string FormatMultiple(ColliderData[] colliders, bool detailed = false)
		{
			StringBuilder sb = new StringBuilder();
			sb.AppendLine($"=== 碰撞箱数据 ({colliders.Length} 个) ===");
			sb.AppendLine();

			for (int i = 0; i < colliders.Length; i++)
			{
				sb.AppendLine($"[{i + 1}] {(detailed ? FormatDetailed(colliders[i]) : FormatSummary(colliders[i]))}");
				if (!detailed)
					sb.AppendLine();
			}

			return sb.ToString();
		}

		/// <summary>
		/// 提取顶点坐标为浮点数组
		/// </summary>
		public static float[] ExtractVerticesAsFloats(ColliderData data)
		{
			float[] result = new float[data.worldVertices.Length * 2];
			
			for (int i = 0; i < data.worldVertices.Length; i++)
			{
				result[i * 2] = data.worldVertices[i].x;
				result[i * 2 + 1] = data.worldVertices[i].y;
			}

			return result;
		}

		/// <summary>
		/// 提取多个碰撞箱的所有顶点
		/// </summary>
		public static float[] ExtractAllVerticesAsFloats(ColliderData[] colliders)
		{
			int totalVertices = 0;
			foreach (var c in colliders)
			{
				totalVertices += c.worldVertices.Length;
			}

			float[] result = new float[totalVertices * 2];
			int offset = 0;

			foreach (var c in colliders)
			{
				float[] vertices = ExtractVerticesAsFloats(c);
				System.Array.Copy(vertices, 0, result, offset, vertices.Length);
				offset += vertices.Length;
			}

			return result;
		}

		/// <summary>
		/// 获取碰撞箱数据的摘要统计
		/// </summary>
		public static string GetStatistics(ColliderData[] colliders)
		{
			if (colliders.Length == 0)
			{
				return "无碰撞箱数据";
			}

			int totalVertices = 0;
			int triggerCount = 0;
			Bounds combinedBounds = colliders[0].bounds;

			foreach (var c in colliders)
			{
				totalVertices += c.vertexCount;
				if (c.isTrigger) triggerCount++;
				combinedBounds.Encapsulate(c.bounds);
			}

			float avgVertices = (float)totalVertices / colliders.Length;

			StringBuilder sb = new StringBuilder();
			sb.AppendLine("=== 碰撞箱统计 ===");
			sb.AppendLine($"总数量: {colliders.Length}");
			sb.AppendLine($"总顶点数: {totalVertices}");
			sb.AppendLine($"平均顶点数: {avgVertices:F1}");
			sb.AppendLine($"触发器数量: {triggerCount}");
			sb.AppendLine($"普通碰撞体: {colliders.Length - triggerCount}");
			sb.AppendLine($"组合边界: 中心 {combinedBounds.center}, 大小 {combinedBounds.size}");

			return sb.ToString();
		}
	}
}

