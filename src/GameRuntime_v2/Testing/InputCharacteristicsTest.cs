using UnityEngine;
using GoiRuntime.Core.Interfaces;
using GoiRuntime.PlayerControl;

namespace GoiRuntime.Testing
{
	public class InputCharacteristicsTest
	{
		private IGameControlService gameControl;
		private PlayerInputService inputService;
		private PlayerStateService stateService;

		private bool isInitialized = false;
		private bool isTestRunning = false;

		private int frameSyncPhase = 0;
		private PlayerState frameSyncS0;
		private PlayerState frameSyncS1;
		private float frameSyncInputValue = 50f;

		public bool IsTestRunning => isTestRunning;

		public bool Initialize(
			IGameControlService control,
			PlayerInputService input,
			PlayerStateService state)
		{
			gameControl = control;
			inputService = input;
			stateService = state;

			isInitialized = gameControl != null &&
			                inputService != null && inputService.IsReady &&
			                stateService != null && stateService.IsReady;

			return isInitialized;
		}

		public void StartFrameSyncTest(float inputValue = 50f)
		{
			if (!isInitialized)
			{
				Debug.LogError("InputCharacteristicsTest 未初始化");
				return;
			}

			if (isTestRunning)
			{
				Debug.LogWarning("已有测试在运行");
				return;
			}

			Debug.Log("=== 开始帧同步测试 ===");
			Debug.Log($"测试输入值: {inputValue}");

			isTestRunning = true;
			frameSyncPhase = 0;
			frameSyncInputValue = inputValue;

			gameControl.Pause();
		}

		public void StopTest()
		{
			if (isTestRunning)
			{
				isTestRunning = false;
				inputService?.SetMouseInput(Vector2.zero);
				Debug.Log("测试已停止");
			}
		}

		public void Update()
		{
			if (!isTestRunning) return;

			switch (frameSyncPhase)
			{
				case 0:
					if (gameControl.IsPaused && !gameControl.IsStepping)
					{
						frameSyncS0 = stateService.GetCurrentState();
						Debug.Log($"Phase 1: S0 位置=({frameSyncS0.playerX:F3}, {frameSyncS0.playerY:F3}) Tip=({frameSyncS0.tipX:F3}, {frameSyncS0.tipY:F3}) 角={frameSyncS0.hammerAngle:F2}");

						inputService.SetMouseInput(new Vector2(frameSyncInputValue, 0f));
						Debug.Log($"Phase 2: 注入 mouseInput=({frameSyncInputValue}, 0)");

						gameControl.StepFrames(2);
						frameSyncPhase = 1;
					}
					break;

				case 1:
					if (gameControl.IsPaused && !gameControl.IsStepping)
					{
						frameSyncS1 = stateService.GetCurrentState();
						Debug.Log($"Phase 4: S1 位置=({frameSyncS1.playerX:F3}, {frameSyncS1.playerY:F3}) Tip=({frameSyncS1.tipX:F3}, {frameSyncS1.tipY:F3}) 角={frameSyncS1.hammerAngle:F2}");

						inputService.SetMouseInput(Vector2.zero);
						AnalyzeResult();

						isTestRunning = false;
					}
					break;
			}
		}

		private void AnalyzeResult()
		{
			float deltaTipX   = Mathf.Abs(frameSyncS1.tipX        - frameSyncS0.tipX);
			float deltaTipY   = Mathf.Abs(frameSyncS1.tipY        - frameSyncS0.tipY);
			float deltaAngle  = Mathf.Abs(frameSyncS1.hammerAngle - frameSyncS0.hammerAngle);
			float deltaPlayerX = Mathf.Abs(frameSyncS1.playerX    - frameSyncS0.playerX);
			float deltaPlayerY = Mathf.Abs(frameSyncS1.playerY    - frameSyncS0.playerY);

			float totalDelta = deltaTipX + deltaTipY + deltaAngle + deltaPlayerX + deltaPlayerY;

			Debug.Log($"帧同步结果: ΔTip=({deltaTipX:F6},{deltaTipY:F6}) ΔAngle={deltaAngle:F6} ΔPlayer=({deltaPlayerX:F6},{deltaPlayerY:F6}) Total={totalDelta:F6}");

			if (totalDelta > 0.001f)
				Debug.Log("帧同步验证通过：mouseInput 在单帧内生效");
			else
				Debug.LogWarning($"帧同步验证未通过：变化量 {totalDelta:F6} 低于阈值");
		}
	}
}
