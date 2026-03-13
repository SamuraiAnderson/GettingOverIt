using UnityEngine;
using System.Collections.Generic;
using GoiRuntime.Core.Events;

namespace GoiRuntime.ColliderCollection
{
	/// <summary>
	/// Player 碰撞箱服务
	/// 负责实时采集 Player 的碰撞箱数据（Pot、Tip、Body等）
	/// </summary>
	public class PlayerColliderService : ColliderCollectorBase
	{
		private GameObject playerObject;
		private Dictionary<string, ColliderData> namedColliders;  // 按名称索引的碰撞箱

		// Player 关键碰撞箱名称
		private const string POT_COLLIDER_NAME = "Pot";
		private const string TIP_COLLIDER_NAME = "Tip";
		private const string BODY_COLLIDER_NAME = "Player";  // 或其他名称

		/// <summary>
		/// 初始化服务
		/// </summary>
		/// <param name="player">Player 对象</param>
		public bool Initialize(GameObject player)
		{
			if (player == null)
			{
				Debug.LogError("Player 对象为 null");
				return false;
			}

			playerObject = player;
			namedColliders = new Dictionary<string, ColliderData>();

			Debug.Log($"PlayerColliderService 初始化成功，Player: {player.name}");
			isInitialized = true;
			return true;
		}

		/// <summary>
		/// 采集 Player 碰撞箱
		/// </summary>
		public override ColliderData[] CollectPlayerColliders(GameObject player)
		{
			if (player != null)
			{
				playerObject = player;
			}

			if (playerObject == null)
			{
				Debug.LogWarning("Player 对象未设置");
				return new ColliderData[0];
			}

			// 采集所有碰撞箱
			cachedColliders = CollectFromGameObject(playerObject, includeChildren: true);

			// 建立名称索引
			namedColliders.Clear();
			foreach (var collider in cachedColliders)
			{
				namedColliders[collider.name] = collider;
			}

			// 发布事件
			EventBus.Publish(GameEvents.CollidersCollected, cachedColliders.ToArray());

			return cachedColliders.ToArray();
		}

		/// <summary>
		/// 获取指定名称的碰撞箱
		/// </summary>
		public ColliderData? GetColliderByName(string name)
		{
			if (namedColliders.ContainsKey(name))
			{
				return namedColliders[name];
			}
			return null;
		}

		/// <summary>
		/// 获取 Pot 碰撞箱
		/// </summary>
		public ColliderData? GetPotCollider()
		{
			return GetColliderByName(POT_COLLIDER_NAME);
		}

		/// <summary>
		/// 获取 Tip 碰撞箱
		/// </summary>
		public ColliderData? GetTipCollider()
		{
			return GetColliderByName(TIP_COLLIDER_NAME);
		}

		/// <summary>
		/// 获取 Body 碰撞箱
		/// </summary>
		public ColliderData? GetBodyCollider()
		{
			return GetColliderByName(BODY_COLLIDER_NAME);
		}

		/// <summary>
		/// 获取所有碰撞箱名称
		/// </summary>
		public string[] GetColliderNames()
		{
			string[] names = new string[namedColliders.Count];
			namedColliders.Keys.CopyTo(names, 0);
			return names;
		}

		/// <summary>
		/// 打印 Player 碰撞箱信息
		/// </summary>
		public void PrintColliderInfo()
		{
			if (cachedColliders.Count == 0)
			{
				Debug.Log("没有 Player 碰撞箱数据");
				return;
			}

			Debug.Log("=== Player 碰撞箱信息 ===");
			Debug.Log($"Player: {playerObject.name}");
			Debug.Log($"碰撞箱总数: {cachedColliders.Count}");

			foreach (var collider in cachedColliders)
			{
				Debug.Log($"  - {collider.name}: {collider.vertexCount} 个顶点");
			}

			// 特定碰撞箱信息
			var pot = GetPotCollider();
			var tip = GetTipCollider();
			var body = GetBodyCollider();

			if (pot.HasValue)
				Debug.Log($"Pot 顶点数: {pot.Value.vertexCount}");
			if (tip.HasValue)
				Debug.Log($"Tip 顶点数: {tip.Value.vertexCount}");
			if (body.HasValue)
				Debug.Log($"Body 顶点数: {body.Value.vertexCount}");
		}

		/// <summary>
		/// 实时更新（在 Update 中调用）
		/// </summary>
		public void UpdateRealtime()
		{
			if (!IsReady) return;

			// 实时更新碰撞箱数据（顶点坐标会随 Player 运动而变化）
			CollectPlayerColliders(playerObject);
		}
	}
}

