using System;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine;
using GoiRuntime.Core.Interfaces;

namespace GoiRuntime.PlayerControl
{
	/// <summary>
	/// Player 复制体管理器
	/// 创建多个 Player 实例，每个实例有独立输入，且不会互相碰撞
	/// 使用 Physics2D.IgnoreCollision 直接忽略碰撞体，保留与环境的交互
	/// </summary>
	public class PlayerDuplicateManager : IDuplicateManager
	{
		#region 字段
		
		private GameObject originalPlayer;
		private List<DuplicateInstance> duplicates = new List<DuplicateInstance>();
		private bool isInitialized = false;
		
		private const int MAX_DUPLICATES = 8;
		
		#endregion
		
		#region 数据结构
		
		/// <summary>
		/// 复制体实例数据
		/// </summary>
		public class DuplicateInstance
		{
			public int index;
			public GameObject gameObject;
			public Component playerControl;
			public FieldInfo mouseInputField;
			public FieldInfo inputEnabledField;
			public Rigidbody2D rigidbody;
			public List<Collider2D> colliders = new List<Collider2D>();
			
			public void SetMouseInput(Vector2 input)
			{
				if (mouseInputField != null && playerControl != null)
				{
					mouseInputField.SetValue(playerControl, input);
				}
			}
			
			public Vector2 GetMouseInput()
			{
				if (mouseInputField != null && playerControl != null)
				{
					object value = mouseInputField.GetValue(playerControl);
					if (value is Vector2 vec) return vec;
				}
				return Vector2.zero;
			}
		}
		
		#endregion
		
		#region IDuplicateManager 实现
		
		public bool Initialize()
		{
			if (isInitialized)
			{
				Debug.Log("PlayerDuplicateManager 已初始化");
				return true;
			}
			
			// 查找原始 Player
			originalPlayer = GameObject.Find("Player");
			if (originalPlayer == null)
			{
				Debug.LogError("未找到原始 Player 对象");
				return false;
			}
			
			Debug.Log($"找到原始 Player: {originalPlayer.name}");
			isInitialized = true;
			return true;
		}
		
		public int GetDuplicateCount()
		{
			return duplicates.Count;
		}
		
		public GameObject GetDuplicate(int index)
		{
			if (index >= 0 && index < duplicates.Count)
			{
				return duplicates[index].gameObject;
			}
			return null;
		}
		
		public GameObject[] GetAllDuplicates()
		{
			GameObject[] result = new GameObject[duplicates.Count];
			for (int i = 0; i < duplicates.Count; i++)
			{
				result[i] = duplicates[i].gameObject;
			}
			return result;
		}
		
		public void SetInputForAll(Vector2[] inputs)
		{
			int count = Math.Min(inputs.Length, duplicates.Count);
			for (int i = 0; i < count; i++)
			{
				SetInputForDuplicate(i, inputs[i]);
			}
		}
		
		public void SetInputForDuplicate(int index, Vector2 input)
		{
			if (index >= 0 && index < duplicates.Count)
			{
				// 设置目标输入（会在每帧 LateUpdate 中持续应用）
				SetTargetInput(index, input);
				
				// 立即设置一次
				duplicates[index].SetMouseInput(input);
			}
		}
		
		#endregion
		
		#region 创建复制体
		
		/// <summary>
		/// 创建指定数量的复制体
		/// </summary>
		/// <param name="count">复制体数量</param>
		/// <param name="offset">每个复制体的位置偏移（用于调试可视化）</param>
		public bool CreateDuplicates(int count, Vector3 offset = default)
		{
			if (!isInitialized)
			{
				Debug.LogError("请先初始化 PlayerDuplicateManager");
				return false;
			}
			
			if (count > MAX_DUPLICATES)
			{
				Debug.LogWarning($"最大复制体数量为 {MAX_DUPLICATES}，已限制");
				count = MAX_DUPLICATES;
			}
			
			// 获取原始 Player 的所有碰撞体
			List<Collider2D> originalColliders = new List<Collider2D>();
			CollectAllColliders(originalPlayer, originalColliders);
			Debug.Log($"原始 Player 碰撞体数量: {originalColliders.Count}");
			
			// 创建复制体
			for (int i = 0; i < count; i++)
			{
				DuplicateInstance instance = CreateSingleDuplicate(i, offset * (i + 1));
				if (instance != null)
				{
					duplicates.Add(instance);
					Debug.Log($"创建复制体 #{i}: {instance.gameObject.name}, 碰撞体数量={instance.colliders.Count}");
				}
			}
			
			// 设置碰撞忽略（复制体之间 + 复制体与原始Player）
			SetupCollisionIgnore(originalColliders);
			
			Debug.Log($"共创建 {duplicates.Count} 个复制体，已设置碰撞隔离");
			return duplicates.Count > 0;
		}
		
		/// <summary>
		/// 创建单个复制体
		/// </summary>
		private DuplicateInstance CreateSingleDuplicate(int index, Vector3 positionOffset)
		{
			try
			{
				// 克隆 Player
				GameObject clone = UnityEngine.Object.Instantiate(originalPlayer);
				clone.name = $"Player_Duplicate_{index}";
				
				// 设置位置偏移（可选，用于调试）
				if (positionOffset != Vector3.zero)
				{
					clone.transform.position += positionOffset;
				}
				
				// 获取 PlayerControl 组件
				Component playerControl = clone.GetComponent("PlayerControl");
				if (playerControl == null)
				{
					Debug.LogError($"复制体 #{index} 未找到 PlayerControl 组件");
					UnityEngine.Object.Destroy(clone);
					return null;
				}
				
				// 反射获取字段
				Type playerControlType = playerControl.GetType();
				FieldInfo mouseInputField = playerControlType.GetField("mouseInput",
					BindingFlags.NonPublic | BindingFlags.Instance);
				FieldInfo inputEnabledField = playerControlType.GetField("input_enabled",
					BindingFlags.NonPublic | BindingFlags.Instance);
				
				if (mouseInputField == null)
				{
					Debug.LogError($"复制体 #{index} 未找到 mouseInput 字段");
					UnityEngine.Object.Destroy(clone);
					return null;
				}
				
				// 保持 input_enabled = true，让 PlayerControl 正常运行
				// 我们会在每帧 LateUpdate 中覆盖 mouseInput
				if (inputEnabledField != null)
				{
					// 确保输入启用
					inputEnabledField.SetValue(playerControl, true);
					Debug.Log($"  复制体 #{index} input_enabled = true（保持启用，我们每帧覆盖 mouseInput）");
				}
				
				// 初始化 mouseInput 为零
				if (mouseInputField != null)
				{
					mouseInputField.SetValue(playerControl, Vector2.zero);
				}
				
				// 禁用可能干扰游戏的其他组件
				DisableInterferingComponents(clone, index);
				
				// 修复关节连接（关键！）
				FixJointConnections(clone, index);
				
				// 收集所有碰撞体
				List<Collider2D> colliders = new List<Collider2D>();
				CollectAllColliders(clone, colliders);
				
				// 创建实例数据
				DuplicateInstance instance = new DuplicateInstance
				{
					index = index,
					gameObject = clone,
					playerControl = playerControl,
					mouseInputField = mouseInputField,
					inputEnabledField = inputEnabledField,
					rigidbody = clone.GetComponent<Rigidbody2D>(),
					colliders = colliders
				};
				
				return instance;
			}
			catch (Exception e)
			{
				Debug.LogError($"创建复制体 #{index} 失败: {e.Message}");
				return null;
			}
		}
		
		#endregion
		
		#region 关节修复
		
		/// <summary>
		/// 修复复制体的关节连接
		/// Unity.Instantiate 会复制关节，但 connectedBody 仍然指向原始对象的 Rigidbody2D
		/// 需要重新映射到复制体内部的 Rigidbody2D
		/// </summary>
		private void FixJointConnections(GameObject clone, int index)
		{
			// 按 GetComponentsInChildren 深度优先遍历顺序建立映射：
			// Unity.Instantiate 保证克隆体子树结构与原始完全一致，因此同索引处的 RB 一一对应。
			// 用 InstanceID（int）作 key，彻底消除同名子节点的歧义。
			Rigidbody2D[] originalRBs = originalPlayer.GetComponentsInChildren<Rigidbody2D>(true);
			Rigidbody2D[] cloneRBs    = clone.GetComponentsInChildren<Rigidbody2D>(true);

			if (originalRBs.Length != cloneRBs.Length)
			{
				Debug.LogWarning($"  复制体 #{index}: 原始 RB({originalRBs.Length}) 与克隆 RB({cloneRBs.Length}) 数量不一致，跳过关节修复");
				return;
			}

			// InstanceID → 克隆 RB
			Dictionary<int, Rigidbody2D> idToCloneRB = new Dictionary<int, Rigidbody2D>(originalRBs.Length);
			for (int i = 0; i < originalRBs.Length; i++)
			{
				idToCloneRB[originalRBs[i].GetInstanceID()] = cloneRBs[i];
			}

			Debug.Log($"  复制体 #{index} 建立了 {idToCloneRB.Count} 个 RB 映射（基于 InstanceID + 子树顺序）");

			// 收集克隆体所有 RB 的 InstanceID，用于检测"已在克隆体内"的 connectedBody
			HashSet<int> cloneRBIds = new HashSet<int>();
			foreach (var rb in cloneRBs) cloneRBIds.Add(rb.GetInstanceID());

			// 修复所有 Joint2D 的 connectedBody
			Joint2D[] joints = clone.GetComponentsInChildren<Joint2D>(true);
			int fixedCount = 0;

			foreach (var joint in joints)
			{
				if (joint.connectedBody == null) continue;

				int origId = joint.connectedBody.GetInstanceID();
				if (idToCloneRB.TryGetValue(origId, out Rigidbody2D mapped))
				{
					joint.connectedBody = mapped;
					fixedCount++;
				}
				else if (!cloneRBIds.Contains(origId))
				{
					Debug.LogWarning($"  复制体 #{index} 关节 [{joint.name}] 的 connectedBody 未找到映射（InstanceID={origId}）");
				}
			}

			Debug.Log($"  复制体 #{index} 修复了 {fixedCount} 个关节连接");
		}
		
		
		#endregion
		
		#region 组件管理
		
		/// <summary>
		/// 禁用复制体中可能干扰游戏的组件
		/// </summary>
		private void DisableInterferingComponents(GameObject clone, int index)
		{
			int disabledCount = 0;
			
			// 1. 禁用所有 AudioSource（避免声音重叠）
			AudioSource[] audioSources = clone.GetComponentsInChildren<AudioSource>(true);
			foreach (var audio in audioSources)
			{
				audio.enabled = false;
				disabledCount++;
			}
			
			// 2. 禁用所有 Camera（避免多相机冲突）
			Camera[] cameras = clone.GetComponentsInChildren<Camera>(true);
			foreach (var cam in cameras)
			{
				cam.enabled = false;
				disabledCount++;
			}
			
			// 3. 禁用 AudioListener
			AudioListener[] listeners = clone.GetComponentsInChildren<AudioListener>(true);
			foreach (var listener in listeners)
			{
				listener.enabled = false;
				disabledCount++;
			}
			
			// 4. 只禁用特定的干扰脚本，保留物理相关脚本
			MonoBehaviour[] scripts = clone.GetComponentsInChildren<MonoBehaviour>(true);
			foreach (var script in scripts)
			{
				if (script == null) continue;
				
				string typeName = script.GetType().Name;
				
				// 禁用可能触发全局逻辑的脚本
				if (typeName.Contains("Save") ||
				    typeName.Contains("Load") ||
				    typeName.Contains("Menu") ||
				    typeName.Contains("UI") ||
				    typeName.Contains("Game") ||  // GameManager 等
				    typeName.Contains("Manager") ||
				    typeName.Contains("Controller") && typeName != "PlayerControl" ||
				    typeName.Contains("Input") && typeName != "PlayerControl" ||
				    typeName.Contains("Settings") ||
				    typeName.Contains("Analytics") ||
				    typeName.Contains("Achievement"))
				{
					script.enabled = false;
					disabledCount++;
					Debug.Log($"    禁用脚本: {typeName}");
				}
			}
			
			Debug.Log($"  复制体 #{index} 已禁用 {disabledCount} 个可能干扰的组件");
		}
		
		#endregion
		
		#region 碰撞隔离
		
		/// <summary>
		/// 递归收集 GameObject 及其所有子对象的碰撞体
		/// </summary>
		private void CollectAllColliders(GameObject obj, List<Collider2D> colliders)
		{
			// 收集当前对象的碰撞体
			Collider2D[] objColliders = obj.GetComponents<Collider2D>();
			colliders.AddRange(objColliders);
			
			// 递归收集子对象
			foreach (Transform child in obj.transform)
			{
				CollectAllColliders(child.gameObject, colliders);
			}
		}
		
		/// <summary>
		/// 设置碰撞忽略
		/// 复制体之间不碰撞，复制体与原始Player不碰撞
		/// 但都保留与环境（Mountain等）的碰撞
		/// </summary>
		private void SetupCollisionIgnore(List<Collider2D> originalColliders)
		{
			int ignoreCount = 0;
			
			// 1. 复制体与原始 Player 之间的碰撞忽略
			foreach (var dup in duplicates)
			{
				foreach (var dupCollider in dup.colliders)
				{
					foreach (var origCollider in originalColliders)
					{
						if (dupCollider != null && origCollider != null)
						{
							Physics2D.IgnoreCollision(dupCollider, origCollider, true);
							ignoreCount++;
						}
					}
				}
			}
			
			// 2. 复制体之间的碰撞忽略
			for (int i = 0; i < duplicates.Count; i++)
			{
				for (int j = i + 1; j < duplicates.Count; j++)
				{
					foreach (var colliderA in duplicates[i].colliders)
					{
						foreach (var colliderB in duplicates[j].colliders)
						{
							if (colliderA != null && colliderB != null)
							{
								Physics2D.IgnoreCollision(colliderA, colliderB, true);
								ignoreCount++;
							}
						}
					}
				}
			}
			
			Debug.Log($"碰撞隔离已设置：共 {ignoreCount} 对碰撞体互相忽略");
		}
		
		#endregion
		
		#region 每帧更新
		
	private Dictionary<int, Vector2> targetInputs = new Dictionary<int, Vector2>();

	public void UpdateDuplicates()
	{
		foreach (var dup in duplicates)
		{
			if (dup == null || dup.gameObject == null) continue;

			if (targetInputs.TryGetValue(dup.index, out Vector2 targetInput))
			{
				dup.SetMouseInput(targetInput);
			}
		}
	}

	public void SetTargetInput(int index, Vector2 input)
	{
		targetInputs[index] = input;
	}
		
		#endregion
		
		#region 状态查询
		
		/// <summary>
		/// 获取复制体实例数据
		/// </summary>
		public DuplicateInstance GetDuplicateInstance(int index)
		{
			if (index >= 0 && index < duplicates.Count)
			{
				return duplicates[index];
			}
			return null;
		}
		
	/// <summary>
	/// 打印所有复制体状态
	/// </summary>
		public void PrintStatus()
		{
			Debug.Log($"=== 复制体状态 ({duplicates.Count}个) ===");
			foreach (var dup in duplicates)
			{
				if (dup.gameObject != null)
				{
					Vector2 pos = dup.gameObject.transform.position;
					Vector2 input = dup.GetMouseInput();
					Debug.Log($"  #{dup.index}: pos=({pos.x:F2},{pos.y:F2}), input=({input.x:F1},{input.y:F1}), colliders={dup.colliders.Count}");
				}
			}
		}
		
		#endregion
		
		#region 清理
		
		/// <summary>
		/// 销毁所有复制体
		/// </summary>
		public void DestroyAll()
		{
			foreach (var dup in duplicates)
			{
				if (dup.gameObject != null)
				{
					UnityEngine.Object.Destroy(dup.gameObject);
				}
			}
			duplicates.Clear();
			Debug.Log("所有复制体已销毁");
		}
		
		#endregion
	}
}
