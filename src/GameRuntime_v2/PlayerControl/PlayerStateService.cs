using UnityEngine;
using System;
using System.Collections.Generic;
using GoiRuntime.Core.Interfaces;

namespace GoiRuntime.PlayerControl
{
	/// <summary>
	/// Player 状态采集服务
	/// 采集 Player 的位置、速度、各部件状态等
	/// </summary>
	public class PlayerStateService : IPlayerStateService
	{
		#region 组件引用
		
		// Player 主体
		private GameObject player;
		private Transform playerTransform;
		private Rigidbody2D playerRigidbody;
		private PolygonCollider2D playerBodyCollider;
		
		// Hub（连接器）
		private Transform hubTransform;
		private Rigidbody2D hubRigidbody;
		
		// Slider（滑动件）
		private Transform sliderTransform;
		private Rigidbody2D sliderRigidbody;
		
		// Handle（手柄）
		private Transform handleTransform;
		private Rigidbody2D handleRigidbody;
		
		// PoleMiddle（锤子中段）
		private Transform poleTransform;
		private Rigidbody2D poleRigidbody;
		
		// Tip（锤子尖端）
		private Transform tipTransform;
		private Rigidbody2D tipRigidbody;
		private PolygonCollider2D tipCollider;
		
		// PotCollider（锅底碰撞体）
		private Transform potColliderTransform;
		private PolygonCollider2D potCollider;
		
		private bool isInitialized = false;
		
		#endregion
		
		#region IPlayerStateService 实现
		
		/// <summary>
		/// 初始化服务
		/// </summary>
		public bool Initialize()
		{
			if (isInitialized)
			{
				Debug.Log("PlayerStateService 已初始化");
				return true;
			}
			
			return FindGameComponents();
		}
		
		/// <summary>
		/// 服务是否就绪
		/// </summary>
		public bool IsReady => isInitialized;
		
		/// <summary>
		/// 采集当前状态
		/// </summary>
		public PlayerState GetCurrentState()
		{
			if (!isInitialized)
			{
				Debug.LogWarning("PlayerStateService 未初始化");
				return default;
			}
			
			return SampleCurrentState();
		}
		
		/// <summary>
		/// 获取状态数组（用于 UDP 发送）
		/// </summary>
		public float[] GetStateArray()
		{
			PlayerState state = GetCurrentState();
			return StateToArray(state);
		}
		
		#endregion
		
		#region 状态采集
		
		/// <summary>
		/// 采样当前状态
		/// </summary>
		private PlayerState SampleCurrentState()
		{
			PlayerState state = new PlayerState();
			
			// Player 主体位置和速度
			if (playerTransform != null)
			{
				state.playerX = playerTransform.position.x;
				state.playerY = playerTransform.position.y;
			}
			
			if (playerRigidbody != null)
			{
				state.velocityX = playerRigidbody.velocity.x;
				state.velocityY = playerRigidbody.velocity.y;
				state.angularVelocity = playerRigidbody.angularVelocity;
			}
			
			// Hub（连接器）
			if (hubTransform != null)
			{
				state.hubX = hubTransform.position.x;
				state.hubY = hubTransform.position.y;
				state.hubAngle = hubTransform.eulerAngles.z;
			}
			if (hubRigidbody != null)
			{
				state.hubVelX = hubRigidbody.velocity.x;
				state.hubVelY = hubRigidbody.velocity.y;
			}
			
			// Slider（滑动件）
			if (sliderTransform != null)
			{
				state.sliderX = sliderTransform.position.x;
				state.sliderY = sliderTransform.position.y;
				state.sliderAngle = sliderTransform.eulerAngles.z;
			}
			if (sliderRigidbody != null)
			{
				state.sliderVelX = sliderRigidbody.velocity.x;
				state.sliderVelY = sliderRigidbody.velocity.y;
			}
			
			// Handle（手柄）
			if (handleTransform != null)
			{
				state.handleX = handleTransform.position.x;
				state.handleY = handleTransform.position.y;
			}
			if (handleRigidbody != null)
			{
				state.handleVelX = handleRigidbody.velocity.x;
				state.handleVelY = handleRigidbody.velocity.y;
			}
			
			// PoleMiddle（锤子中段）
			if (poleTransform != null)
			{
				state.poleX = poleTransform.position.x;
				state.poleY = poleTransform.position.y;
			}
			if (poleRigidbody != null)
			{
				state.poleVelX = poleRigidbody.velocity.x;
				state.poleVelY = poleRigidbody.velocity.y;
			}
			
			// Tip（锤子尖端）
			if (tipTransform != null)
			{
				state.tipX = tipTransform.position.x;
				state.tipY = tipTransform.position.y;
			}
			if (tipRigidbody != null)
			{
				state.tipVelX = tipRigidbody.velocity.x;
				state.tipVelY = tipRigidbody.velocity.y;
			}
			
			// 时间戳
			state.timestamp = Time.time;
			
			// 计算锤子角度
			if (hubTransform != null && tipTransform != null)
			{
				Vector2 dir = tipTransform.position - hubTransform.position;
				state.hammerAngle = Mathf.Atan2(dir.y, dir.x) * Mathf.Rad2Deg;
			}
			
			return state;
		}
		
		/// <summary>
		/// 将状态转换为数组（39 个浮点数）
		/// </summary>
		private float[] StateToArray(PlayerState state)
		{
			return new float[]
			{
				// Player 主体 (0-4)
				state.playerX, state.playerY,
				state.velocityX, state.velocityY,
				state.angularVelocity,
				
				// Hub (5-9)
				state.hubX, state.hubY,
				state.hubVelX, state.hubVelY,
				state.hubAngle,
				
				// Slider (10-14)
				state.sliderX, state.sliderY,
				state.sliderVelX, state.sliderVelY,
				state.sliderAngle,
				
				// Handle (15-18)
				state.handleX, state.handleY,
				state.handleVelX, state.handleVelY,
				
				// PoleMiddle (19-22)
				state.poleX, state.poleY,
				state.poleVelX, state.poleVelY,
				
				// Tip (23-26)
				state.tipX, state.tipY,
				state.tipVelX, state.tipVelY,
				
				// 额外状态 (27-38)
				state.hammerAngle,
				state.timestamp,
				0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f  // 预留位
			};
		}
		
		#endregion
		
		#region 组件查找
		
		/// <summary>
		/// 查找游戏组件
		/// </summary>
		private bool FindGameComponents()
		{
			Debug.Log("🔍 查找 Player 组件...");
			
			player = GameObject.Find("Player");
			if (player == null)
			{
				Debug.LogError("❌ 未找到 Player 对象");
				return false;
			}
			
			Debug.Log("✅ 找到 Player 对象");
			
			// Player 主体组件
			playerTransform = player.transform;
			playerRigidbody = player.GetComponent<Rigidbody2D>();
			playerBodyCollider = player.GetComponent<PolygonCollider2D>();
			
			// Hub（连接器）
			hubTransform = FindDeepChild(player.transform, "Hub");
			if (hubTransform != null)
			{
				hubRigidbody = hubTransform.GetComponent<Rigidbody2D>();
			}
			
			// Slider（滑动件）
			sliderTransform = FindDeepChild(player.transform, "Slider");
			if (sliderTransform != null)
			{
				sliderRigidbody = sliderTransform.GetComponent<Rigidbody2D>();
			}
			
			// Handle（手柄）
			handleTransform = FindDeepChild(player.transform, "Handle");
			if (handleTransform != null)
			{
				handleRigidbody = handleTransform.GetComponent<Rigidbody2D>();
			}
			
			// PoleMiddle（锤子中段）
			poleTransform = FindDeepChild(player.transform, "PoleMiddle");
			if (poleTransform != null)
			{
				poleRigidbody = poleTransform.GetComponent<Rigidbody2D>();
			}
			
			// Tip（锤子尖端）
			tipTransform = FindDeepChild(player.transform, "Tip");
			if (tipTransform != null)
			{
				tipRigidbody = tipTransform.GetComponent<Rigidbody2D>();
				tipCollider = tipTransform.GetComponent<PolygonCollider2D>();
			}
			
			// PotCollider（锅底碰撞体）
			potColliderTransform = FindDeepChild(player.transform, "PotCollider");
			if (potColliderTransform != null)
			{
				potCollider = potColliderTransform.GetComponent<PolygonCollider2D>();
			}
			
			Debug.Log($"✅ 组件查找完成 - Hub:{hubRigidbody != null}, Slider:{sliderRigidbody != null}, Tip:{tipRigidbody != null}");
			
			isInitialized = playerRigidbody != null;
			return isInitialized;
		}
		
		/// <summary>
		/// 深度查找子对象
		/// </summary>
		private Transform FindDeepChild(Transform parent, string name)
		{
			Queue<Transform> queue = new Queue<Transform>();
			queue.Enqueue(parent);
			
			while (queue.Count > 0)
			{
				Transform current = queue.Dequeue();
				
				if (current.name == name)
				{
					return current;
				}
				
				for (int i = 0; i < current.childCount; i++)
				{
					queue.Enqueue(current.GetChild(i));
				}
			}
			
			return null;
		}
		
		#endregion
		
		#region 公共方法
		
		/// <summary>
		/// 获取 Player 对象
		/// </summary>
		public GameObject GetPlayerObject()
		{
			return player;
		}
		
		/// <summary>
		/// 获取 Player 位置
		/// </summary>
		public Vector2 GetPlayerPosition()
		{
			if (playerTransform == null) return Vector2.zero;
			return playerTransform.position;
		}
		
		/// <summary>
		/// 获取 Player 速度
		/// </summary>
		public Vector2 GetPlayerVelocity()
		{
			if (playerRigidbody == null) return Vector2.zero;
			return playerRigidbody.velocity;
		}
		
		/// <summary>
		/// 获取锤子尖端位置
		/// </summary>
		public Vector2 GetTipPosition()
		{
			if (tipTransform == null) return Vector2.zero;
			return tipTransform.position;
		}
		
		/// <summary>
		/// 获取锤子角度
		/// </summary>
		public float GetHammerAngle()
		{
			if (hubTransform == null || tipTransform == null) return 0f;
			Vector2 dir = tipTransform.position - hubTransform.position;
			return Mathf.Atan2(dir.y, dir.x) * Mathf.Rad2Deg;
		}
		
		/// <summary>
		/// 获取 PotCollider（锅底碰撞体）
		/// </summary>
		public PolygonCollider2D GetPotCollider()
		{
			return potCollider;
		}
		
		/// <summary>
		/// 获取 TipCollider（锤子尖端碰撞体）
		/// </summary>
		public PolygonCollider2D GetTipCollider()
		{
			return tipCollider;
		}
		
		/// <summary>
		/// 获取 Player 主体碰撞体
		/// </summary>
		public PolygonCollider2D GetBodyCollider()
		{
			return playerBodyCollider;
		}
		
		#endregion
	}
}

