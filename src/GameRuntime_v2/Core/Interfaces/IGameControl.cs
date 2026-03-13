using UnityEngine;

namespace GoiRuntime.Core.Interfaces
{
	/// <summary>
	/// 游戏控制服务接口
	/// 提供游戏暂停、恢复、时间控制等基础功能
	/// </summary>
	public interface IGameControlService
	{
		/// <summary>
		/// 游戏是否暂停
		/// </summary>
		bool IsPaused { get; }
		
		/// <summary>
		/// 当前时间缩放
		/// </summary>
		float TimeScale { get; }
		
		/// <summary>
		/// 暂停游戏
		/// </summary>
		void Pause();
		
		/// <summary>
		/// 恢复游戏
		/// </summary>
		void Resume();
		
		/// <summary>
		/// 切换暂停状态
		/// </summary>
		void TogglePause();
		
		/// <summary>
		/// 设置时间缩放
		/// </summary>
		void SetTimeScale(float scale);
		
		/// <summary>
		/// 单帧步进（暂停状态下前进指定帧数）
		/// </summary>
		void StepFrames(int frameCount = 1);
		
		/// <summary>
		/// 是否正在步进中
		/// </summary>
		bool IsStepping { get; }
		
		/// <summary>
		/// 处理帧更新（需要在 LateUpdate 中调用）
		/// </summary>
		void OnLateUpdate();
		
		/// <summary>
		/// 手动推进物理（暂停状态下使用）
		/// </summary>
		/// <param name="steps">物理步数（默认1步）</param>
		void SimulatePhysics(int steps = 1);
		
		/// <summary>
		/// 暂停并执行一次物理步进
		/// </summary>
		void PauseAndStep();
	}
}

