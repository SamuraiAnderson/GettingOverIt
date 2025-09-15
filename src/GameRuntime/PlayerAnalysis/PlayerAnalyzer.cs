using UnityEngine;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace GoiHitboxLogger
{
	/// <summary>
	/// Player对象分析器 - 专门用于分析游戏中的Player对象
	/// </summary>
	public static class PlayerAnalyzer
	{
		/// <summary>
		/// Player对象详细信息结构
		/// </summary>
		public struct PlayerInfo
		{
			public string name;
			public string fullPath;
			public Vector3 position;
			public Vector3 rotation;
			public Vector3 scale;
			public bool isActive;
			public int layer;
			public string tag;
			public List<ComponentInfo> components;
			public List<ChildInfo> children;
			public PhysicsInfo physics;
		}
		
		/// <summary>
		/// 组件信息结构
		/// </summary>
		public struct ComponentInfo
		{
			public string typeName;
			public bool isEnabled;
			public Dictionary<string, string> properties;
		}
		
		/// <summary>
		/// 子对象信息结构
		/// </summary>
		public struct ChildInfo
		{
			public string name;
			public string fullPath;
			public Vector3 localPosition;
			public Vector3 worldPosition;
			public bool isActive;
			public int componentCount;
			public List<string> componentTypes;
			public int depth;  // 层级深度
			public List<ChildInfo> children;  // 递归子对象
		}
		
		/// <summary>
		/// 物理信息结构
		/// </summary>
		public struct PhysicsInfo
		{
			public bool hasRigidbody;
			public bool hasCollider;
			public float mass;
			public Vector3 velocity;
			public Vector3 angularVelocity;
			public bool useGravity;
			public bool isKinematic;
			public List<string> colliderTypes;
		}
		
		/// <summary>
		/// 分析Player对象
		/// </summary>
		/// <returns>Player信息结构</returns>
		public static PlayerInfo AnalyzePlayer()
		{
			Debug.Log("开始分析Player对象...");
			
			GameObject player = FindPlayerObject();
			if (player == null)
			{
				Debug.LogWarning("未找到Player对象");
				return new PlayerInfo { name = "NOT_FOUND" };
			}
			
			Debug.Log($"找到Player对象: {player.name}");
			return AnalyzePlayerDetails(player);
		}
		
		/// <summary>
		/// 查找Player对象（精确匹配）
		/// </summary>
		/// <returns>Player GameObject，如果未找到返回null</returns>
		private static GameObject FindPlayerObject()
		{
			// 方法1: 直接查找名为"Player"的对象
			GameObject player = GameObject.Find("Player");
			if (player != null)
			{
				Debug.Log("通过名称'Player'找到Player对象");
				return player;
			}
			
			// 方法2: 查找标签为"Player"的对象
			try
			{
				GameObject taggedPlayer = GameObject.FindWithTag("Player");
				if (taggedPlayer != null)
				{
					Debug.Log("通过标签'Player'找到Player对象");
					return taggedPlayer;
				}
			}
			catch (UnityException)
			{
				// Player标签不存在，继续其他方法
			}
			
			// 方法3: 精确匹配常见的Player对象名称
			string[] exactNames = {
				"PlayerController",
				"PlayerCharacter", 
				"MainPlayer",
				"GamePlayer"
			};
			
			foreach (string name in exactNames)
			{
				GameObject candidate = GameObject.Find(name);
				if (candidate != null)
				{
					Debug.Log($"通过精确名称'{name}'找到Player对象");
					return candidate;
				}
			}
			
			Debug.LogWarning("未找到Player对象");
			return null;
		}
		
		/// <summary>
		/// 分析Player对象的详细信息
		/// </summary>
		/// <param name="player">Player GameObject</param>
		/// <returns>详细的Player信息</returns>
		private static PlayerInfo AnalyzePlayerDetails(GameObject player)
		{
			PlayerInfo info = new PlayerInfo
			{
				name = player.name,
				fullPath = GetGameObjectPath(player),
				position = player.transform.position,
				rotation = player.transform.eulerAngles,
				scale = player.transform.localScale,
				isActive = player.activeInHierarchy,
				layer = player.layer,
				tag = player.tag,
				components = new List<ComponentInfo>(),
				children = new List<ChildInfo>(),
				physics = AnalyzePhysics(player)
			};
			
			// 分析组件
			Component[] components = player.GetComponents<Component>();
			foreach (Component comp in components)
			{
				if (comp != null)
				{
					info.components.Add(AnalyzeComponent(comp));
				}
			}
			
			// 分析子对象（递归）
			for (int i = 0; i < player.transform.childCount; i++)
			{
				Transform child = player.transform.GetChild(i);
				info.children.Add(AnalyzeChildRecursive(child.gameObject, 1));
			}
			
			return info;
		}
		
		/// <summary>
		/// 分析组件信息
		/// </summary>
		/// <param name="component">要分析的组件</param>
		/// <returns>组件信息</returns>
		private static ComponentInfo AnalyzeComponent(Component component)
		{
			ComponentInfo info = new ComponentInfo
			{
				typeName = component.GetType().Name,
				isEnabled = true, // 默认启用
				properties = new Dictionary<string, string>()
			};
			
			// 检查是否是Behaviour类型（可以启用/禁用）
			if (component is Behaviour behaviour)
			{
				info.isEnabled = behaviour.enabled;
			}
			
			// 根据组件类型分析特定属性
			switch (component)
			{
				case Transform transform:
					info.properties["Position"] = transform.position.ToString();
					info.properties["Rotation"] = transform.eulerAngles.ToString();
					info.properties["Scale"] = transform.localScale.ToString();
					info.properties["ChildCount"] = transform.childCount.ToString();
					break;
					
				case Rigidbody rigidbody:
					info.properties["Mass"] = rigidbody.mass.ToString();
					info.properties["Velocity"] = rigidbody.velocity.ToString();
					info.properties["AngularVelocity"] = rigidbody.angularVelocity.ToString();
					info.properties["UseGravity"] = rigidbody.useGravity.ToString();
					info.properties["IsKinematic"] = rigidbody.isKinematic.ToString();
					info.properties["Drag"] = rigidbody.drag.ToString();
					info.properties["AngularDrag"] = rigidbody.angularDrag.ToString();
					break;
					
				case Collider collider:
					info.properties["IsTrigger"] = collider.isTrigger.ToString();
					info.properties["Bounds"] = collider.bounds.ToString();
					info.properties["Material"] = collider.material ? collider.material.name : "None";
					break;
					
				case Renderer renderer:
					info.properties["Enabled"] = renderer.enabled.ToString();
					info.properties["IsVisible"] = renderer.isVisible.ToString();
					info.properties["MaterialCount"] = renderer.materials.Length.ToString();
					if (renderer.materials.Length > 0 && renderer.materials[0] != null)
					{
						info.properties["MainMaterial"] = renderer.materials[0].name;
					}
					break;
					
				case AudioSource audioSource:
					info.properties["Clip"] = audioSource.clip ? audioSource.clip.name : "None";
					info.properties["Volume"] = audioSource.volume.ToString();
					info.properties["IsPlaying"] = audioSource.isPlaying.ToString();
					info.properties["Loop"] = audioSource.loop.ToString();
					break;
					
				case Animation animation:
					info.properties["ClipCount"] = animation.GetClipCount().ToString();
					info.properties["IsPlaying"] = animation.isPlaying.ToString();
					break;
					
				case Animator animator:
					info.properties["Controller"] = animator.runtimeAnimatorController ? 
						animator.runtimeAnimatorController.name : "None";
					info.properties["ParameterCount"] = animator.parameterCount.ToString();
					info.properties["LayerCount"] = animator.layerCount.ToString();
					break;
			}
			
			return info;
		}
		
		/// <summary>
		/// 递归分析子对象信息
		/// </summary>
		/// <param name="child">子对象</param>
		/// <param name="depth">当前层级深度</param>
		/// <returns>子对象信息</returns>
		private static ChildInfo AnalyzeChildRecursive(GameObject child, int depth)
		{
			Component[] components = child.GetComponents<Component>();
			List<string> componentTypes = new List<string>();
			
			foreach (Component comp in components)
			{
				if (comp != null)
				{
					componentTypes.Add(comp.GetType().Name);
				}
			}
			
			ChildInfo childInfo = new ChildInfo
			{
				name = child.name,
				fullPath = GetGameObjectPath(child),
				localPosition = child.transform.localPosition,
				worldPosition = child.transform.position,
				isActive = child.activeSelf,
				componentCount = components.Length,
				componentTypes = componentTypes,
				depth = depth,
				children = new List<ChildInfo>()
			};
			
			// 递归分析子对象的子对象（限制深度避免无限递归）
			if (depth < 5 && child.transform.childCount > 0)
			{
				for (int i = 0; i < child.transform.childCount; i++)
				{
					Transform grandChild = child.transform.GetChild(i);
					childInfo.children.Add(AnalyzeChildRecursive(grandChild.gameObject, depth + 1));
				}
			}
			
			return childInfo;
		}
		
		/// <summary>
		/// 分析物理信息
		/// </summary>
		/// <param name="player">Player对象</param>
		/// <returns>物理信息</returns>
		private static PhysicsInfo AnalyzePhysics(GameObject player)
		{
			PhysicsInfo physics = new PhysicsInfo
			{
				colliderTypes = new List<string>()
			};
			
			Rigidbody rb = player.GetComponent<Rigidbody>();
			if (rb != null)
			{
				physics.hasRigidbody = true;
				physics.mass = rb.mass;
				physics.velocity = rb.velocity;
				physics.angularVelocity = rb.angularVelocity;
				physics.useGravity = rb.useGravity;
				physics.isKinematic = rb.isKinematic;
			}
			
			Collider[] colliders = player.GetComponents<Collider>();
			physics.hasCollider = colliders.Length > 0;
			
			foreach (Collider collider in colliders)
			{
				if (collider != null)
				{
					physics.colliderTypes.Add(collider.GetType().Name);
				}
			}
			
			return physics;
		}
		
		/// <summary>
		/// 获取GameObject的完整路径
		/// </summary>
		/// <param name="obj">目标对象</param>
		/// <returns>完整路径</returns>
		private static string GetGameObjectPath(GameObject obj)
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
		/// 导出Player分析结果到文件
		/// </summary>
		/// <param name="playerInfo">Player信息</param>
		/// <param name="fileName">文件名（可选）</param>
		public static void ExportPlayerAnalysis(PlayerInfo playerInfo, string fileName = null)
		{
			if (fileName == null)
			{
				fileName = $"PlayerAnalysis_{System.DateTime.Now:yyyyMMdd_HHmmss}.txt";
			}
			
			try
			{
				StringBuilder report = new StringBuilder();
				
				report.AppendLine("=== Player对象详细分析报告 ===");
				report.AppendLine($"分析时间: {System.DateTime.Now}");
				report.AppendLine($"场景名称: {UnityEngine.SceneManagement.SceneManager.GetActiveScene().name}");
				report.AppendLine();
				
				// 基本信息
				report.AppendLine("📋 基本信息:");
				report.AppendLine($"  名称: {playerInfo.name}");
				report.AppendLine($"  完整路径: {playerInfo.fullPath}");
				report.AppendLine($"  激活状态: {playerInfo.isActive}");
				report.AppendLine($"  图层: {playerInfo.layer} ({LayerMask.LayerToName(playerInfo.layer)})");
				report.AppendLine($"  标签: {playerInfo.tag}");
				report.AppendLine();
				
				// Transform信息
				report.AppendLine("🌍 Transform信息:");
				report.AppendLine($"  位置: {playerInfo.position}");
				report.AppendLine($"  旋转: {playerInfo.rotation}");
				report.AppendLine($"  缩放: {playerInfo.scale}");
				report.AppendLine();
				
				// 物理信息
				report.AppendLine("⚡ 物理信息:");
				report.AppendLine($"  有刚体: {playerInfo.physics.hasRigidbody}");
				if (playerInfo.physics.hasRigidbody)
				{
					report.AppendLine($"    质量: {playerInfo.physics.mass}");
					report.AppendLine($"    速度: {playerInfo.physics.velocity}");
					report.AppendLine($"    角速度: {playerInfo.physics.angularVelocity}");
					report.AppendLine($"    使用重力: {playerInfo.physics.useGravity}");
					report.AppendLine($"    运动学: {playerInfo.physics.isKinematic}");
				}
				report.AppendLine($"  有碰撞体: {playerInfo.physics.hasCollider}");
				if (playerInfo.physics.hasCollider)
				{
					report.AppendLine($"    碰撞体类型: {string.Join(", ", playerInfo.physics.colliderTypes.ToArray())}");
				}
				report.AppendLine();
				
				// 组件信息
				report.AppendLine("🧩 组件列表:");
				foreach (ComponentInfo comp in playerInfo.components)
				{
					report.AppendLine($"  - {comp.typeName} (启用: {comp.isEnabled})");
					foreach (var prop in comp.properties)
					{
						report.AppendLine($"      {prop.Key}: {prop.Value}");
					}
				}
				report.AppendLine();
				
				// 子对象信息（递归显示）
				report.AppendLine("👶 子对象层级结构:");
				if (playerInfo.children.Count > 0)
				{
					foreach (ChildInfo child in playerInfo.children)
					{
						AppendChildInfoRecursive(report, child, "");
					}
				}
				else
				{
					report.AppendLine("  无子对象");
				}
				
				// 保存到文件
				string baseDir = Path.Combine(Application.dataPath, "..");
				string analysisDir = Path.Combine(baseDir, "PlayerAnalysis");
				string filePath = Path.Combine(analysisDir, fileName);
				
				Directory.CreateDirectory(Path.GetDirectoryName(filePath));
				File.WriteAllText(filePath, report.ToString());
				
				Debug.Log($"✅ Player分析报告已保存到: {filePath}");
			}
			catch (System.Exception e)
			{
				Debug.LogError($"❌ 保存Player分析报告失败: {e.Message}");
			}
		}
		
		/// <summary>
		/// 递归添加子对象信息到报告
		/// </summary>
		/// <param name="report">报告字符串构建器</param>
		/// <param name="child">子对象信息</param>
		/// <param name="indent">缩进字符串</param>
		private static void AppendChildInfoRecursive(StringBuilder report, ChildInfo child, string indent)
		{
			string depthIndicator = new string('-', child.depth);
			report.AppendLine($"  {indent}{depthIndicator} {child.name} (深度: {child.depth}, 激活: {child.isActive})");
			report.AppendLine($"  {indent}    完整路径: {child.fullPath}");
			report.AppendLine($"  {indent}    本地位置: {child.localPosition}");
			report.AppendLine($"  {indent}    世界位置: {child.worldPosition}");
			report.AppendLine($"  {indent}    组件数量: {child.componentCount}");
			report.AppendLine($"  {indent}    组件类型: {string.Join(", ", child.componentTypes.ToArray())}");
			
			// 递归显示子对象的子对象
			if (child.children != null && child.children.Count > 0)
			{
				report.AppendLine($"  {indent}    子对象数量: {child.children.Count}");
				foreach (ChildInfo grandChild in child.children)
				{
					AppendChildInfoRecursive(report, grandChild, indent + "    ");
				}
			}
			report.AppendLine();
		}
		
		/// <summary>
		/// 打印Player信息到控制台
		/// </summary>
		/// <param name="playerInfo">Player信息</param>
		public static void PrintPlayerInfo(PlayerInfo playerInfo)
		{
			if (playerInfo.name == "NOT_FOUND")
			{
				Debug.LogWarning("Player对象未找到");
				return;
			}
			
			Debug.Log("=== Player对象信息 ===");
			Debug.Log($"名称: {playerInfo.name}");
			Debug.Log($"路径: {playerInfo.fullPath}");
			Debug.Log($"位置: {playerInfo.position}");
			Debug.Log($"组件数量: {playerInfo.components.Count}");
			Debug.Log($"子对象数量: {playerInfo.children.Count}");
			Debug.Log($"有物理组件: {playerInfo.physics.hasRigidbody}");
			Debug.Log($"有碰撞体: {playerInfo.physics.hasCollider}");
		}
	}
}
