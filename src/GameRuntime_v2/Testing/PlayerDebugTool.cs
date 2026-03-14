using UnityEngine;
using GoiRuntime.PlayerControl;
using GoiRuntime.Core.Interfaces;

namespace GoiRuntime.Testing
{
	/// <summary>
	/// Player 调试工具
	/// 提供热键调试功能，使用 PlayerControl 和 GameControl 服务
	/// </summary>
	public class PlayerDebugTool : IDebugTool
	{
		private PlayerInputService inputService;
		private PlayerStateService stateService;
		private IGameControlService gameControl;
		
		private PlayerDuplicateManager duplicateManager;
		private InputCharacteristicsTest inputTest;
		
		private bool stepTestActive = false;
		private int stepTestPhase = 0;
		private PlayerState stepTestS0;
		
		private bool isInitialized = false;

		public string ToolName => "PlayerDebugTool";
		public bool IsEnabled { get; private set; } = true;
		public void Enable() => IsEnabled = true;
		public void Disable() => IsEnabled = false;
		public void OnGUI() { }
		
		/// <summary>
		/// 初始化调试工具
		/// </summary>
		public bool Initialize(
			PlayerInputService input, 
			PlayerStateService state, 
			IGameControlService control)
		{
			inputService = input;
			stateService = state;
			gameControl = control;
			
			isInitialized = gameControl != null &&
			                ((inputService != null && inputService.IsReady) ||
			                 (stateService != null && stateService.IsReady));
			
			if (isInitialized)
			{
				// 初始化原始输入特性测试
				inputTest = new InputCharacteristicsTest();
				inputTest.Initialize(gameControl, inputService, stateService);
				
				Debug.Log("PlayerDebugTool 初始化成功");
				PrintHelp();
			}
			
			return isInitialized;
		}
		
		/// <summary>
		/// 更新（在 Update 中调用）
		/// </summary>
		public void Update()
		{
			if (!isInitialized) return;
			
			// 更新原始测试
			if (inputTest != null)
			{
				inputTest.Update();
			}
			
			// 处理步进测试状态机
			UpdateStepTest();
			
			HandleHotkeys();
		}
		
		/// <summary>
		/// LateUpdate（在 LateUpdate 中调用）
		/// 在游戏 Update 之后执行，覆盖复制体的 mouseInput
		/// </summary>
		public void LateUpdate()
		{
			if (!isInitialized || gameControl == null) return;
			
			gameControl.OnLateUpdate();
			
			// 每帧覆盖复制体的 mouseInput（在游戏 Update 读取真实鼠标之后）
			if (duplicateManager != null)
			{
				duplicateManager.UpdateDuplicates();
			}
		}
		
		/// <summary>
		/// 更新单步测试状态机
		/// </summary>
		private void UpdateStepTest()
		{
			if (!stepTestActive) return;
			
			switch (stepTestPhase)
			{
				case 1:
					// 等待步进完成
					if (gameControl.IsPaused && !gameControl.IsStepping)
					{
						// 记录新状态
						var s1 = stateService.GetCurrentState();
						
						// 清除输入
						inputService.SetMouseInput(Vector2.zero);
						
						// 打印变化
						float dx = s1.playerX - stepTestS0.playerX;
						float dy = s1.playerY - stepTestS0.playerY;
						float dAngle = s1.hammerAngle - stepTestS0.hammerAngle;
						float dTipX = s1.tipX - stepTestS0.tipX;
						float dTipY = s1.tipY - stepTestS0.tipY;
						
						Debug.Log("=== 新版单步测试结果 ===");
						Debug.Log($"Player 位置变化: Δx={dx:F4}, Δy={dy:F4}");
						Debug.Log($"锤子角度变化: Δ={dAngle:F4}°");
						Debug.Log($"Tip 位置变化: Δx={dTipX:F4}, Δy={dTipY:F4}");
						
						float totalDelta = Mathf.Abs(dx) + Mathf.Abs(dy) + Mathf.Abs(dAngle) + Mathf.Abs(dTipX) + Mathf.Abs(dTipY);
						Debug.Log($"总变化量: {totalDelta:F4} {(totalDelta > 0.01f ? "(有效)" : "(无变化)")}");
						
						stepTestActive = false;
						stepTestPhase = 0;
					}
					break;
			}
		}
		
		/// <summary>
		/// 处理热键
		/// </summary>
		private void HandleHotkeys()
		{
			// F3: 打印 Player 状态
			if (Input.GetKeyDown(KeyCode.F3))
			{
				PrintPlayerState();
			}
			
			// F4: 打印 Player 碰撞箱
			if (Input.GetKeyDown(KeyCode.F4))
			{
				PrintPlayerColliders();
			}
			
			// F5: 暂停/恢复游戏
			if (Input.GetKeyDown(KeyCode.F5))
			{
				gameControl.TogglePause();
			}
			
			// F6: 新版单步测试
			if (Input.GetKeyDown(KeyCode.F6))
			{
				TestSingleStep();
			}
			
			// F7: 打印暂停状态
			if (Input.GetKeyDown(KeyCode.F7))
			{
				PrintPauseStatus();
			}
			
			// F8: 原版帧同步测试（InputCharacteristicsTest）
			if (Input.GetKeyDown(KeyCode.F8))
			{
				if (inputTest != null && !inputTest.IsTestRunning)
				{
					inputTest.StartFrameSyncTest(50f);
				}
				else if (inputTest != null && inputTest.IsTestRunning)
				{
					inputTest.StopTest();
				}
			}
			
			// F9: 创建复制体
			if (Input.GetKeyDown(KeyCode.F9))
			{
				TestCreateDuplicates();
			}
			
			// F10: 为复制体设置输入
			if (Input.GetKeyDown(KeyCode.F10))
			{
				TestDuplicateInputs();
			}
			
			// F11: 打印复制体状态
			if (Input.GetKeyDown(KeyCode.F11))
			{
				PrintDuplicateStatus();
			}
			
			// F12: 销毁复制体
			if (Input.GetKeyDown(KeyCode.F12))
			{
				DestroyAllDuplicates();
			}
		}
		
		#region 调试功能
		
		/// <summary>
		/// 打印 Player 状态
		/// </summary>
		private void PrintPlayerState()
		{
			if (stateService == null || !stateService.IsReady)
			{
				Debug.LogWarning("PlayerStateService 未就绪");
				return;
			}
			
			var state = stateService.GetCurrentState();
			
			Debug.Log("=== Player 状态 ===");
			Debug.Log($"位置: ({state.playerX:F2}, {state.playerY:F2})");
			Debug.Log($"速度: ({state.velocityX:F2}, {state.velocityY:F2})");
			Debug.Log($"角速度: {state.angularVelocity:F2}");
			Debug.Log($"锤子角度: {state.hammerAngle:F1}°");
			Debug.Log($"Hub: ({state.hubX:F2}, {state.hubY:F2})");
			Debug.Log($"Tip: ({state.tipX:F2}, {state.tipY:F2})");
			Debug.Log($"时间戳: {state.timestamp:F3}");
		}
		
		/// <summary>
		/// 打印 Player 碰撞箱
		/// </summary>
		private void PrintPlayerColliders()
		{
			if (stateService == null || !stateService.IsReady)
			{
				Debug.LogWarning("PlayerStateService 未就绪");
				return;
			}
			
			GameObject player = stateService.GetPlayerObject();
			if (player == null)
			{
				Debug.LogWarning("Player 对象为 null");
				return;
			}

			var polys = player.GetComponentsInChildren<PolygonCollider2D>(true);
			Debug.Log($"=== Player 碰撞箱 ({polys.Length} 个 PolygonCollider2D) ===");
			foreach (var poly in polys)
			{
				int totalPts = poly.GetTotalPointCount();
				Debug.Log($"  {poly.gameObject.name}: {poly.pathCount} path(s), {totalPts} vertices, enabled={poly.enabled}");
			}
		}
		
		/// <summary>
		/// 新版单步测试（带输入注入）
		/// </summary>
		private void TestSingleStep()
		{
			if (!gameControl.IsPaused)
			{
				Debug.LogWarning("请先按 F5 暂停游戏");
				return;
			}
			
			if (stateService == null || !stateService.IsReady)
			{
				Debug.LogWarning("PlayerStateService 未就绪");
				return;
			}
			
			if (inputService == null || !inputService.IsReady)
			{
				Debug.LogWarning("PlayerInputService 未就绪");
				return;
			}
			
			if (stepTestActive)
			{
				Debug.LogWarning("上一次测试还未完成");
				return;
			}
			
			// 确保输入启用
			inputService.SetInputEnabled(true);
			
			// 记录初始状态
			stepTestS0 = stateService.GetCurrentState();
			Debug.Log("=== 开始新版单步测试 ===");
			Debug.Log($"初始位置: ({stepTestS0.playerX:F3}, {stepTestS0.playerY:F3})");
			Debug.Log($"初始 Tip: ({stepTestS0.tipX:F3}, {stepTestS0.tipY:F3})");
			Debug.Log($"初始锤子角度: {stepTestS0.hammerAngle:F2}°");
			
			// 注入输入
			Vector2 testInput = new Vector2(50f, 0f);
			inputService.SetMouseInput(testInput);
			
			// 验证输入是否成功
			Vector2 actualInput = inputService.GetMouseInput();
			Debug.Log($"注入输入: mouseInput = ({testInput.x}, {testInput.y})");
			Debug.Log($"验证输入: 实际值 = ({actualInput.x:F1}, {actualInput.y:F1})");
			
			// 步进多帧
			gameControl.StepFrames(5);
			Debug.Log("执行 StepFrames(5)...");
			
			// 设置状态机
			stepTestActive = true;
			stepTestPhase = 1;
		}
		
		/// <summary>
		/// 打印暂停状态
		/// </summary>
		private void PrintPauseStatus()
		{
			Debug.Log("=== 游戏控制状态 ===");
			Debug.Log($"游戏暂停: {gameControl.IsPaused}");
			Debug.Log($"Time.timeScale: {gameControl.TimeScale}");
			Debug.Log($"步进中: {gameControl.IsStepping}");
			Debug.Log($"Time.deltaTime: {Time.deltaTime}");
			Debug.Log($"Time.unscaledDeltaTime: {Time.unscaledDeltaTime}");
			
			if (duplicateManager != null)
			{
				Debug.Log($"复制体数量: {duplicateManager.GetDuplicateCount()}");
			}
			
			if (inputTest != null)
			{
				Debug.Log($"原版测试运行中: {inputTest.IsTestRunning}");
			}
		}
		
		#endregion
		
		#region 复制体测试
		
		/// <summary>
		/// 测试创建复制体
		/// </summary>
		private void TestCreateDuplicates()
		{
			if (duplicateManager == null)
			{
				duplicateManager = new PlayerDuplicateManager();
				if (!duplicateManager.Initialize())
				{
					Debug.LogError("复制体管理器初始化失败");
					return;
				}
			}
			
			// 如果已有复制体，先销毁
			if (duplicateManager.GetDuplicateCount() > 0)
			{
				Debug.Log("已存在复制体，先销毁后重新创建");
				duplicateManager.DestroyAll();
			}
			
			// 创建 3 个复制体
			bool success = duplicateManager.CreateDuplicates(3, Vector3.zero);
			
			if (success)
			{
				Debug.Log("复制体创建成功！按 F10 设置不同输入");
				duplicateManager.PrintStatus();
			}
		}
		
		/// <summary>
		/// 为复制体设置不同输入测试
		/// </summary>
		private void TestDuplicateInputs()
		{
			if (duplicateManager == null || duplicateManager.GetDuplicateCount() == 0)
			{
				Debug.LogWarning("请先按 F9 创建复制体");
				return;
			}
			
			// 为每个复制体设置不同的输入
			Vector2[] inputs = new Vector2[]
			{
				new Vector2(50f, 0f),    // 复制体0: 向右
				new Vector2(-50f, 0f),   // 复制体1: 向左
				new Vector2(0f, 50f)     // 复制体2: 向上
			};
			
			duplicateManager.SetInputForAll(inputs);
			
			Debug.Log("已为复制体设置不同输入:");
			Debug.Log("  #0: (50, 0) - 向右");
			Debug.Log("  #1: (-50, 0) - 向左");
			Debug.Log("  #2: (0, 50) - 向上");
		}
		
		/// <summary>
		/// 打印复制体状态
		/// </summary>
		private void PrintDuplicateStatus()
		{
			if (duplicateManager == null || duplicateManager.GetDuplicateCount() == 0)
			{
				Debug.LogWarning("没有复制体，请先按 F9 创建");
				return;
			}
			
			duplicateManager.PrintStatus();
		}
		
		/// <summary>
		/// 销毁所有复制体
		/// </summary>
		private void DestroyAllDuplicates()
		{
			if (duplicateManager != null)
			{
				duplicateManager.DestroyAll();
			}
			else
			{
				Debug.Log("没有复制体需要销毁");
			}
		}
		
		#endregion
		
		#region 帮助信息
		
		/// <summary>
		/// 打印帮助信息
		/// </summary>
		private void PrintHelp()
		{
			Debug.Log("=== Player 调试工具 ===");
			Debug.Log("快捷键:");
			Debug.Log("  F3: 打印 Player 状态");
			Debug.Log("  F4: 打印 Player 碰撞箱");
			Debug.Log("  F5: 暂停/恢复游戏");
			Debug.Log("  F6: 新版单步测试");
			Debug.Log("  F7: 打印暂停状态");
			Debug.Log("  F8: 【原版】帧同步测试（InputCharacteristicsTest）");
			Debug.Log("  --- 复制体测试 ---");
			Debug.Log("  F9: 创建 3 个复制体");
			Debug.Log("  F10: 为复制体设置不同输入");
			Debug.Log("  F11: 打印复制体状态");
			Debug.Log("  F12: 销毁所有复制体");
		}
		
		#endregion
	}
}
