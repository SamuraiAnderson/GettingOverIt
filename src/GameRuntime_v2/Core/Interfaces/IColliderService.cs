using UnityEngine;

namespace GoiRuntime.Core.Interfaces
{
	/// <summary>
	/// 碰撞箱数据结构
	/// </summary>
	public struct ColliderData
	{
		public string name;              // 碰撞体名称
		public Vector2[] worldVertices;  // 世界坐标顶点
		public Bounds bounds;            // 边界框
		public int vertexCount;          // 顶点数量
		public bool isTrigger;           // 是否触发器
		public Vector3 position;         // 位置
	}

	/// <summary>
	/// 碰撞箱采集服务接口
	/// 用于采集环境和 Player 的碰撞箱数据
	/// </summary>
	public interface IColliderService
	{
		/// <summary>
		/// 初始化服务
		/// </summary>
		bool Initialize();

		/// <summary>
		/// 服务是否就绪
		/// </summary>
		bool IsReady { get; }

		/// <summary>
		/// 采集环境碰撞箱（Mountain 等静态环境）
		/// </summary>
		ColliderData[] CollectEnvironmentColliders();

		/// <summary>
		/// 采集 Player 的碰撞箱（Pot、Tip、Body 等）
		/// </summary>
		ColliderData[] CollectPlayerColliders(GameObject player);

		/// <summary>
		/// 获取碰撞箱数量
		/// </summary>
		int GetColliderCount();

		/// <summary>
		/// 获取组合边界框
		/// </summary>
		Bounds GetCombinedBounds();

		/// <summary>
		/// 导出碰撞箱数据到文件
		/// </summary>
		string ExportToFile(string filePath);
	}
}

