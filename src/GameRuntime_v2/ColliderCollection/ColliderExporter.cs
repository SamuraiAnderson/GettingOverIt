using UnityEngine;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using GoiRuntime.Core.Utilities;

namespace GoiRuntime.ColliderCollection
{
	/// <summary>
	/// 碰撞体几何导出工具
	/// 将环境中所有静态 PolygonCollider2D 和 Player 轮廓导出为 JSON 文件。
	/// 环境使用世界坐标（静态不动），Player 使用本地坐标（形状不变，位姿由 state 提供）。
	/// </summary>
	public static class ColliderExporter
	{
		private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

		// Player 关键碰撞部件的 GameObject 名称
		private static readonly Dictionary<string, string> PlayerPartNames = new Dictionary<string, string>
		{
			{ "body",      "Player" },
			{ "tip",       "Tip" },
			{ "pot",       "PotCollider" },
			{ "pot_sides", "Sides" },
		};

		// ── 公共 API ──────────────────────────────────────────────

		/// <summary>
		/// 导出环境碰撞体为世界坐标 JSON。
		/// 搜索场景中所有 PolygonCollider2D，排除有 Rigidbody2D 的动态对象（Player 等）。
		/// </summary>
		public static bool ExportEnvironment(string outputPath)
		{
			var allPolys = Object.FindObjectsOfType<PolygonCollider2D>();
			var envPolys = new List<PolygonCollider2D>();

			int skippedDynamic = 0, skippedTrigger = 0;
			foreach (var poly in allPolys)
			{
				if (poly.GetComponentInParent<Rigidbody2D>() != null)
				{ skippedDynamic++; continue; }
				if (poly.isTrigger)
				{ skippedTrigger++; continue; }
				envPolys.Add(poly);
			}

			if (envPolys.Count == 0)
			{
				Debug.LogWarning("[ColliderExporter] 未找到任何环境 PolygonCollider2D，跳过导出");
				return false;
			}

			Debug.Log($"[ColliderExporter] 场景中 {allPolys.Length} 个 PolygonCollider2D → " +
			          $"环境: {envPolys.Count}，跳过动态: {skippedDynamic}，跳过触发器: {skippedTrigger}");

			for (int d = 0; d < envPolys.Count; d++)
			{
				var p = envPolys[d];
				Debug.Log($"[ColliderExporter]   [{d}] {GetHierarchyPath(p.transform)}  " +
				          $"(paths={p.pathCount}, enabled={p.enabled})");
			}

			var polys = envPolys;

			var sb = new StringBuilder();
			sb.AppendLine("{");
			sb.AppendLine("  \"type\": \"environment\",");
			sb.AppendLine("  \"colliders\": [");

			for (int i = 0; i < polys.Count; i++)
			{
				var poly = polys[i];
				string fullPath = GetHierarchyPath(poly.transform);
				sb.AppendLine("    {");
				sb.AppendFormat(Inv, "      \"name\": \"{0}\",\n", EscapeJson(poly.gameObject.name));
				sb.AppendFormat(Inv, "      \"path\": \"{0}\",\n", EscapeJson(fullPath));
				sb.AppendLine("      \"paths\": [");
				WritePathsWorld(sb, poly, "        ");
				sb.AppendLine("      ]");
				sb.Append("    }");
				if (i < polys.Count - 1) sb.Append(',');
				sb.AppendLine();
			}

			sb.AppendLine("  ]");
			sb.AppendLine("}");

			return WriteFile(outputPath, sb.ToString(), "environment");
		}

		/// <summary>
		/// 导出 Player 轮廓（body / tip / pot / pot_sides）为本地坐标 JSON。
		/// </summary>
		public static bool ExportPlayerContour(GameObject player, string outputPath)
		{
			if (player == null)
			{
				Debug.LogWarning("[ColliderExporter] Player 对象为 null，跳过 Player 轮廓导出");
				return false;
			}

			var allPolys = player.GetComponentsInChildren<PolygonCollider2D>(true);
			var partMap = new Dictionary<string, PolygonCollider2D>();

			foreach (var poly in allPolys)
			{
				string objName = poly.gameObject.name;
				foreach (var kv in PlayerPartNames)
				{
					if (objName == kv.Value && !partMap.ContainsKey(kv.Key))
					{
						partMap[kv.Key] = poly;
						break;
					}
				}
			}

			Debug.Log($"[ColliderExporter] Player 碰撞部件匹配: {partMap.Count}/{PlayerPartNames.Count}" +
			          $" (body={partMap.ContainsKey("body")}, tip={partMap.ContainsKey("tip")}," +
			          $" pot={partMap.ContainsKey("pot")}, pot_sides={partMap.ContainsKey("pot_sides")})");

			var sb = new StringBuilder();
			sb.AppendLine("{");
			sb.AppendLine("  \"type\": \"player_contour\",");
			sb.AppendLine("  \"parts\": {");

			int written = 0;
			foreach (var kv in PlayerPartNames)
			{
				string partKey = kv.Key;
				if (!partMap.ContainsKey(partKey)) continue;

				var poly = partMap[partKey];
				if (written > 0) sb.AppendLine(",");

				sb.AppendFormat(Inv, "    \"{0}\": {{\n", partKey);
				sb.AppendLine("      \"paths\": [");
				WritePathsLocal(sb, poly, "        ");
				sb.AppendLine("      ]");
				sb.Append("    }");
				written++;
			}

			sb.AppendLine();
			sb.AppendLine("  }");
			sb.AppendLine("}");

			return WriteFile(outputPath, sb.ToString(), "player_contour");
		}

		/// <summary>
		/// 一次性导出环境 + Player 轮廓到指定目录。
		/// </summary>
		public static void ExportAll(GameObject player, string collidersDir)
		{
			PathManager.EnsureDirectory(collidersDir);
			ExportEnvironment(Path.Combine(collidersDir, "environment.json"));
			ExportPlayerContour(player, Path.Combine(collidersDir, "player_contour.json"));
		}

		// ── 内部方法 ──────────────────────────────────────────────

		/// <summary>
		/// 写入多 path 的世界坐标顶点（环境用）
		/// </summary>
		private static void WritePathsWorld(StringBuilder sb, PolygonCollider2D poly, string indent)
		{
			Transform t = poly.transform;
			for (int p = 0; p < poly.pathCount; p++)
			{
				Vector2[] path = poly.GetPath(p);
				sb.Append(indent).Append('[');
				for (int v = 0; v < path.Length; v++)
				{
					Vector3 world = t.TransformPoint(path[v]);
					sb.AppendFormat(Inv, "[{0:F4},{1:F4}]", world.x, world.y);
					if (v < path.Length - 1) sb.Append(',');
				}
				sb.Append(']');
				if (p < poly.pathCount - 1) sb.Append(',');
				sb.AppendLine();
			}
		}

		/// <summary>
		/// 写入多 path 的本地坐标顶点（Player 用）
		/// </summary>
		private static void WritePathsLocal(StringBuilder sb, PolygonCollider2D poly, string indent)
		{
			for (int p = 0; p < poly.pathCount; p++)
			{
				Vector2[] path = poly.GetPath(p);
				sb.Append(indent).Append('[');
				for (int v = 0; v < path.Length; v++)
				{
					sb.AppendFormat(Inv, "[{0:F4},{1:F4}]", path[v].x, path[v].y);
					if (v < path.Length - 1) sb.Append(',');
				}
				sb.Append(']');
				if (p < poly.pathCount - 1) sb.Append(',');
				sb.AppendLine();
			}
		}

		private static bool WriteFile(string path, string content, string label)
		{
			try
			{
				PathManager.EnsureDirectory(Path.GetDirectoryName(path));
				File.WriteAllText(path, content);
				Debug.Log($"[ColliderExporter] {label} 已导出: {path} ({content.Length} bytes)");
				return true;
			}
			catch (System.Exception e)
			{
				Debug.LogError($"[ColliderExporter] {label} 导出失败: {e.Message}");
				return false;
			}
		}

		private static string GetHierarchyPath(Transform t)
		{
			var parts = new List<string>();
			while (t != null)
			{
				parts.Add(t.name);
				t = t.parent;
			}
			parts.Reverse();
			return string.Join("/", parts.ToArray());
		}

		private static string EscapeJson(string value)
		{
			if (string.IsNullOrEmpty(value)) return string.Empty;
			return value.Replace("\\", "\\\\").Replace("\"", "\\\"");
		}
	}
}
