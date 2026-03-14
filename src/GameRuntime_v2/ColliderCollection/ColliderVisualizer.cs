using UnityEngine;

namespace GoiRuntime.ColliderCollection
{
	/// <summary>
	/// Player 碰撞箱 GL 描边可视化
	/// 附加到 Camera 上，在 OnRenderObject 中用 GL.LINES 绘制所有 Collider2D 轮廓。
	/// </summary>
	public class ColliderVisualizer : MonoBehaviour
	{
		public GameObject[] playerRoots;

		private Material lineMaterial;

		private const int CIRCLE_SEGMENTS = 32;

		private static readonly Color COLOR_POT    = new Color(0f, 1f, 0f, 0.9f);
		private static readonly Color COLOR_TIP    = new Color(1f, 0.2f, 0.2f, 0.9f);
		private static readonly Color COLOR_BODY   = new Color(1f, 1f, 0f, 0.9f);
		private static readonly Color COLOR_DEFAULT = new Color(0f, 1f, 1f, 0.9f);

		void Awake()
		{
			CreateLineMaterial();
		}

		void OnEnable()
		{
			LogDetectedColliders();
		}

		private void LogDetectedColliders()
		{
			if (playerRoots == null) return;

			for (int r = 0; r < playerRoots.Length; r++)
			{
				var root = playerRoots[r];
				if (root == null) continue;

				var colliders = root.GetComponentsInChildren<Collider2D>(true);
				Debug.Log($"[ColliderVisualizer] Root[{r}] \"{root.name}\" — {colliders.Length} 个 Collider2D:");
				foreach (var col in colliders)
				{
					if (col == null) continue;
					string path = GetHierarchyPath(col.transform, root.transform);
					string extra = col is PolygonCollider2D pc ? $" pathCount={pc.pathCount}" : "";
					Debug.Log($"  [{(col.enabled ? "ON" : "off")}] {col.GetType().Name} on \"{path}\"{extra}");
				}
			}
		}

		private static string GetHierarchyPath(Transform leaf, Transform root)
		{
			var parts = new System.Collections.Generic.List<string>();
			for (Transform t = leaf; t != null && t != root; t = t.parent)
				parts.Add(t.name);
			parts.Add(root.name);
			parts.Reverse();
			return string.Join("/", parts.ToArray());
		}

		private void CreateLineMaterial()
		{
			var shader = Shader.Find("Hidden/Internal-Colored");
			if (shader == null)
			{
				Debug.LogError("[ColliderVisualizer] Hidden/Internal-Colored shader not found");
				return;
			}
			lineMaterial = new Material(shader);
			lineMaterial.hideFlags = HideFlags.HideAndDontSave;
			lineMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
			lineMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
			lineMaterial.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
			lineMaterial.SetInt("_ZWrite", 0);
			lineMaterial.SetInt("_ZTest", (int)UnityEngine.Rendering.CompareFunction.Always);
		}

		void OnRenderObject()
		{
			if (Camera.current != Camera.main) return;
			if (playerRoots == null || lineMaterial == null) return;

			GL.PushMatrix();
			lineMaterial.SetPass(0);

			foreach (var root in playerRoots)
			{
				if (root == null) continue;

				var colliders = root.GetComponentsInChildren<Collider2D>(true);
				foreach (var col in colliders)
				{
					if (col == null || !col.enabled) continue;

					Color c = PickColor(col.gameObject.name);

					if (col is PolygonCollider2D poly)
						DrawPolygonCollider(poly, c);
					else if (col is CircleCollider2D circle)
						DrawCircleCollider(circle, c);
					else if (col is BoxCollider2D box)
						DrawBoxCollider(box, c);
					else if (col is EdgeCollider2D edge)
						DrawEdgeCollider(edge, c);
					else if (col is CapsuleCollider2D capsule)
						DrawCapsuleCollider(capsule, c);
				}
			}

			GL.PopMatrix();
		}

		// ── 各类碰撞体绘制 ───────────────────────────────────────

		private void DrawPolygonCollider(PolygonCollider2D poly, Color c)
		{
			Transform t = poly.transform;
			Vector2 offset = poly.offset;

			for (int p = 0; p < poly.pathCount; p++)
			{
				Vector2[] path = poly.GetPath(p);
				if (path.Length < 2) continue;

				GL.Begin(GL.LINES);
				GL.Color(c);
				for (int i = 0; i < path.Length; i++)
				{
					Vector3 a = t.TransformPoint(path[i] + offset);
					Vector3 b = t.TransformPoint(path[(i + 1) % path.Length] + offset);
					GL.Vertex3(a.x, a.y, a.z);
					GL.Vertex3(b.x, b.y, b.z);
				}
				GL.End();
			}
		}

		private void DrawCircleCollider(CircleCollider2D circle, Color c)
		{
			Transform t = circle.transform;
			Vector2 center = circle.offset;
			float radius = circle.radius;

			GL.Begin(GL.LINES);
			GL.Color(c);
			for (int i = 0; i < CIRCLE_SEGMENTS; i++)
			{
				float a0 = 2f * Mathf.PI * i / CIRCLE_SEGMENTS;
				float a1 = 2f * Mathf.PI * ((i + 1) % CIRCLE_SEGMENTS) / CIRCLE_SEGMENTS;

				Vector2 local0 = center + new Vector2(Mathf.Cos(a0), Mathf.Sin(a0)) * radius;
				Vector2 local1 = center + new Vector2(Mathf.Cos(a1), Mathf.Sin(a1)) * radius;

				Vector3 w0 = t.TransformPoint(local0);
				Vector3 w1 = t.TransformPoint(local1);
				GL.Vertex3(w0.x, w0.y, w0.z);
				GL.Vertex3(w1.x, w1.y, w1.z);
			}
			GL.End();
		}

		private void DrawBoxCollider(BoxCollider2D box, Color c)
		{
			Transform t = box.transform;
			Vector2 center = box.offset;
			Vector2 half = box.size * 0.5f;

			Vector2[] corners = new Vector2[]
			{
				center + new Vector2(-half.x, -half.y),
				center + new Vector2( half.x, -half.y),
				center + new Vector2( half.x,  half.y),
				center + new Vector2(-half.x,  half.y),
			};

			GL.Begin(GL.LINES);
			GL.Color(c);
			for (int i = 0; i < 4; i++)
			{
				Vector3 a = t.TransformPoint(corners[i]);
				Vector3 b = t.TransformPoint(corners[(i + 1) % 4]);
				GL.Vertex3(a.x, a.y, a.z);
				GL.Vertex3(b.x, b.y, b.z);
			}
			GL.End();
		}

		private void DrawEdgeCollider(EdgeCollider2D edge, Color c)
		{
			Transform t = edge.transform;
			Vector2 offset = edge.offset;
			Vector2[] pts = edge.points;
			if (pts.Length < 2) return;

			GL.Begin(GL.LINES);
			GL.Color(c);
			for (int i = 0; i < pts.Length - 1; i++)
			{
				Vector3 a = t.TransformPoint(pts[i] + offset);
				Vector3 b = t.TransformPoint(pts[i + 1] + offset);
				GL.Vertex3(a.x, a.y, a.z);
				GL.Vertex3(b.x, b.y, b.z);
			}
			GL.End();
		}

		private void DrawCapsuleCollider(CapsuleCollider2D capsule, Color c)
		{
			Transform t = capsule.transform;
			Vector2 center = capsule.offset;
			Vector2 size = capsule.size;
			float halfW = size.x * 0.5f;
			float halfH = size.y * 0.5f;
			bool vertical = capsule.direction == CapsuleDirection2D.Vertical;

			float rectHalf, capRadius;
			if (vertical)
			{
				capRadius = halfW;
				rectHalf = Mathf.Max(0f, halfH - capRadius);
			}
			else
			{
				capRadius = halfH;
				rectHalf = Mathf.Max(0f, halfW - capRadius);
			}

			GL.Begin(GL.LINES);
			GL.Color(c);

			if (vertical)
			{
				// left & right edges
				Vector3 bl = t.TransformPoint(center + new Vector2(-capRadius, -rectHalf));
				Vector3 tl = t.TransformPoint(center + new Vector2(-capRadius,  rectHalf));
				Vector3 br = t.TransformPoint(center + new Vector2( capRadius, -rectHalf));
				Vector3 tr = t.TransformPoint(center + new Vector2( capRadius,  rectHalf));
				GL.Vertex3(bl.x, bl.y, bl.z); GL.Vertex3(tl.x, tl.y, tl.z);
				GL.Vertex3(br.x, br.y, br.z); GL.Vertex3(tr.x, tr.y, tr.z);

				// top semicircle
				DrawArc(t, center + new Vector2(0, rectHalf), capRadius, 0f, Mathf.PI);
				// bottom semicircle
				DrawArc(t, center + new Vector2(0, -rectHalf), capRadius, Mathf.PI, 2f * Mathf.PI);
			}
			else
			{
				// top & bottom edges
				Vector3 lb = t.TransformPoint(center + new Vector2(-rectHalf, -capRadius));
				Vector3 lt = t.TransformPoint(center + new Vector2(-rectHalf,  capRadius));
				Vector3 rb = t.TransformPoint(center + new Vector2( rectHalf, -capRadius));
				Vector3 rt = t.TransformPoint(center + new Vector2( rectHalf,  capRadius));
				GL.Vertex3(lb.x, lb.y, lb.z); GL.Vertex3(lt.x, lt.y, lt.z);
				GL.Vertex3(rb.x, rb.y, rb.z); GL.Vertex3(rt.x, rt.y, rt.z);

				// right semicircle
				DrawArc(t, center + new Vector2(rectHalf, 0), capRadius, -Mathf.PI * 0.5f, Mathf.PI * 0.5f);
				// left semicircle
				DrawArc(t, center + new Vector2(-rectHalf, 0), capRadius, Mathf.PI * 0.5f, Mathf.PI * 1.5f);
			}

			GL.End();
		}

		/// <summary>
		/// GL.LINES 模式下绘制圆弧（已在 GL.Begin/End 内调用）
		/// </summary>
		private void DrawArc(Transform t, Vector2 center, float radius, float startAngle, float endAngle)
		{
			int segments = CIRCLE_SEGMENTS / 2;
			float step = (endAngle - startAngle) / segments;
			for (int i = 0; i < segments; i++)
			{
				float a0 = startAngle + step * i;
				float a1 = startAngle + step * (i + 1);
				Vector2 local0 = center + new Vector2(Mathf.Cos(a0), Mathf.Sin(a0)) * radius;
				Vector2 local1 = center + new Vector2(Mathf.Cos(a1), Mathf.Sin(a1)) * radius;
				Vector3 w0 = t.TransformPoint(local0);
				Vector3 w1 = t.TransformPoint(local1);
				GL.Vertex3(w0.x, w0.y, w0.z);
				GL.Vertex3(w1.x, w1.y, w1.z);
			}
		}

		// ── 颜色映射 ────────────────────────────────────────────

		private Color PickColor(string objName)
		{
			if (objName == null) return COLOR_DEFAULT;
			string lower = objName.ToLower();
			if (lower.Contains("pot")) return COLOR_POT;
			if (lower.Contains("tip")) return COLOR_TIP;
			if (lower == "player" || lower.Contains("body")) return COLOR_BODY;
			return COLOR_DEFAULT;
		}
	}
}
