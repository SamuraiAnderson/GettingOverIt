using UnityEngine;
using GoiRuntime.Core.Interfaces;

namespace GoiRuntime.Core.Services
{
	/// <summary>
	/// 游戏控制服务
	/// 提供游戏暂停、恢复、时间控制等基础功能
	/// </summary>
	public class GameControlService : IGameControlService
	{
		private bool isPaused = false;
		private bool isStepping = false;
		private int stepFramesRemaining = 0;
		private float savedTimeScale = 1f;
		private bool stepStartedThisFrame = false;  // 防止同一帧立即减少计数
		
		/// <summary>
		/// 游戏是否暂停
		/// </summary>
		public bool IsPaused => isPaused;
		
		/// <summary>
		/// 当前时间缩放
		/// </summary>
		public float TimeScale => Time.timeScale;
		
		/// <summary>
		/// 是否正在步进中
		/// </summary>
		public bool IsStepping => isStepping;
		
		/// <summary>
		/// 暂停游戏
		/// </summary>
		public void Pause()
		{
			if (isPaused) return;
			
			savedTimeScale = Time.timeScale;
			Time.timeScale = 0f;
			isPaused = true;
			
			Debug.Log("⏸️ 游戏已暂停");
		}
		
		/// <summary>
		/// 恢复游戏
		/// </summary>
		public void Resume()
		{
			if (!isPaused) return;
			
			Time.timeScale = savedTimeScale > 0f ? savedTimeScale : 1f;
			isPaused = false;
			isStepping = false;
			stepFramesRemaining = 0;
			
			Debug.Log("▶️ 游戏已恢复");
		}
		
		/// <summary>
		/// 切换暂停状态
		/// </summary>
		public void TogglePause()
		{
			if (isPaused)
				Resume();
			else
				Pause();
		}
		
		/// <summary>
		/// 设置时间缩放
		/// </summary>
		public void SetTimeScale(float scale)
		{
			Time.timeScale = scale;
			isPaused = (scale == 0f);
			savedTimeScale = scale > 0f ? scale : savedTimeScale;
			
			Debug.Log($"🎮 TimeScale: {scale}");
		}
		
		/// <summary>
		/// 单帧步进（暂停状态下前进指定帧数）
		/// </summary>
		public void StepFrames(int frameCount = 1)
		{
			if (!isPaused)
			{
				Debug.LogWarning("⚠️ 请先暂停游戏再使用单帧步进");
				return;
			}
			
			if (frameCount <= 0) return;
			
			// 启动步进
			isStepping = true;
			stepFramesRemaining = frameCount;
			stepStartedThisFrame = true;  // 标记这一帧刚开始步进
			Time.timeScale = savedTimeScale > 0f ? savedTimeScale : 1f;
			
			Debug.Log($"⏯️ 步进 {frameCount} 帧...");
		}
		
		/// <summary>
		/// 处理帧更新（需要在 LateUpdate 中调用）
		/// </summary>
		public void OnLateUpdate()
		{
			if (!isStepping) return;
			
			// 如果是刚开始步进的那一帧，跳过（等待下一帧才开始计数）
			if (stepStartedThisFrame)
			{
				stepStartedThisFrame = false;
				return;
			}
			
			stepFramesRemaining--;
			
			if (stepFramesRemaining <= 0)
			{
				// 步进完成，重新暂停
				Time.timeScale = 0f;
				isStepping = false;
				isPaused = true;  // 重要：标记为暂停状态
				Debug.Log("⏸️ 步进完成");
			}
		}
		
		/// <summary>
		/// 获取状态信息
		/// </summary>
		public string GetStatusInfo()
		{
			return $"暂停: {isPaused} | TimeScale: {Time.timeScale} | 步进中: {isStepping} | 剩余帧: {stepFramesRemaining}";
		}
		
		/// <summary>
		/// 手动推进物理（暂停状态下使用）
		/// 使用 Physics2D.Simulate() 在 timeScale=0 时也能更新物理
		/// </summary>
		/// <param name="steps">物理步数（默认1步，每步 fixedDeltaTime）</param>
		public void SimulatePhysics(int steps = 1)
		{
			if (steps <= 0) return;
			
			float dt = Time.fixedDeltaTime;
			
			// 保存原始模拟模式
			var originalMode = Physics2D.simulationMode;
			
			// 切换到脚本模式以允许手动模拟
			Physics2D.simulationMode = SimulationMode2D.Script;
			
			for (int i = 0; i < steps; i++)
			{
				// 手动推进 2D 物理
				Physics2D.Simulate(dt);
			}
			
			// 恢复原始模式
			Physics2D.simulationMode = originalMode;
			
			Debug.Log($"🔬 物理模拟: {steps} 步 (dt={dt:F4}s)");
		}
		
		/// <summary>
		/// 暂停并执行一次物理步进
		/// 用于同步训练模式
		/// </summary>
		public void PauseAndStep()
		{
			// 确保游戏暂停
			if (!isPaused)
			{
				Pause();
			}
			
			// 手动推进一步物理
			SimulatePhysics(1);
		}
	}
}

