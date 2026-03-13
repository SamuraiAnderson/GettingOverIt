using UnityEngine;
using System;

namespace GoiRuntime.Core.Interfaces
{
	/// <summary>
	/// Player 状态数据（29个浮点数）
	/// </summary>
	[Serializable]
	public struct PlayerState
	{
		// 基础
		public float timestamp;

		// Player主体
		public float playerX;
		public float playerY;
		public float velocityX;
		public float velocityY;
		public float angularVelocity;
		public float hammerAngle;

		// Hub（连接器）
		public float hubX, hubY, hubVelX, hubVelY, hubAngle;

		// Slider（滑动件）
		public float sliderX, sliderY, sliderVelX, sliderVelY, sliderAngle;

		// Handle（手柄）
		public float handleX, handleY, handleVelX, handleVelY;

		// PoleMiddle（锤子中段）
		public float poleX, poleY, poleVelX, poleVelY;

		// Tip（锤子尖端）
		public float tipX, tipY, tipVelX, tipVelY;

	/// <summary>
	/// 转换为浮点数组（29个元素）
	/// </summary>
	public float[] ToFloatArray()
	{
		return new float[]
		{
			// Player 主体 (0-4)
			playerX, playerY,
			velocityX, velocityY,
			angularVelocity,

			// Hub (5-9)
			hubX, hubY,
			hubVelX, hubVelY,
			hubAngle,

			// Slider (10-14)
			sliderX, sliderY,
			sliderVelX, sliderVelY,
			sliderAngle,

			// Handle (15-18)
			handleX, handleY,
			handleVelX, handleVelY,

			// PoleMiddle (19-22)
			poleX, poleY,
			poleVelX, poleVelY,

			// Tip (23-26)
			tipX, tipY,
			tipVelX, tipVelY,

			// 派生量 (27-28)
			hammerAngle,
			timestamp,
		};
	}
	}

	/// <summary>
	/// Player 输入控制接口
	/// 通过反射直接注入输入到 PlayerControl 组件
	/// </summary>
	public interface IPlayerInputService
	{
		/// <summary>
		/// 初始化服务（自动查找 Player 对象并设置反射）
		/// </summary>
		bool Initialize();

		/// <summary>
		/// 服务是否就绪
		/// </summary>
		bool IsReady { get; }

		/// <summary>
		/// 设置鼠标输入（会自动验证范围）
		/// </summary>
		void SetMouseInput(Vector2 mouseInput);

		/// <summary>
		/// 获取当前鼠标输入
		/// </summary>
		Vector2 GetMouseInput();

		/// <summary>
		/// 设置输入启用状态
		/// </summary>
		void SetInputEnabled(bool enabled);

		/// <summary>
		/// 获取输入启用状态
		/// </summary>
		bool GetInputEnabled();
	}

	/// <summary>
	/// 复制体管理接口
	/// 管理多个 Player 实例（num_duplis）
	/// </summary>
	public interface IDuplicateManager
	{
		/// <summary>
		/// 初始化（查找所有复制体）
		/// </summary>
		bool Initialize();

		/// <summary>
		/// 获取复制体数量
		/// </summary>
		int GetDuplicateCount();

		/// <summary>
		/// 获取指定索引的复制体
		/// </summary>
		GameObject GetDuplicate(int index);

		/// <summary>
		/// 获取所有复制体
		/// </summary>
		GameObject[] GetAllDuplicates();

		/// <summary>
		/// 为所有复制体设置输入（批量操作）
		/// </summary>
		void SetInputForAll(Vector2[] inputs);

		/// <summary>
		/// 为指定复制体设置输入
		/// </summary>
		void SetInputForDuplicate(int index, Vector2 input);
	}

	/// <summary>
	/// Player 状态服务接口
	/// 采集 Player 的所有状态数据
	/// </summary>
	public interface IPlayerStateService
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
		/// 获取当前状态
		/// </summary>
		PlayerState GetCurrentState();

		/// <summary>
		/// 获取状态数组（29 维，用于 TCP 步进回包）
		/// </summary>
		float[] GetStateArray();
	}
}

