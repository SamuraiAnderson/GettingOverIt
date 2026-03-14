using BepInEx;
using UnityEngine;
using System.IO;
using GoiRuntime.Core.Configuration;
using GoiRuntime.Core.Interfaces;
using GoiRuntime.Core.Services;
using GoiRuntime.Core.Utilities;
using GoiRuntime.ColliderCollection;
using GoiRuntime.PlayerControl;
using GoiRuntime.Communication;
using GoiRuntime.CameraControl;
using GoiRuntime.Testing;

namespace GoiRuntime.Core
{
	/// <summary>
	/// 游戏运行时管理器
	/// 统一入口，根据模式初始化不同的服务
	/// </summary>
	[BepInPlugin("com.symbol.goi.runtime_v2", "Game Runtime v2", "2.0.0")]
	public class GameRuntimeManager : BaseUnityPlugin
	{
		private RuntimeConfig config;
		private ModeManager modeManager;

		// 核心服务
		private GameControlService gameControlService;
		
		// Player 控制服务
		private PlayerInputService playerInputService;
		private PlayerStateService playerStateService;
		private PlayerDuplicateManager duplicateManager;

		// 训练核心（GameRuntime 模式）
		private StepController stepController;
		private TcpStepServer tcpStepServer;

		// 碰撞箱可视化
		private ColliderVisualizer colliderVisualizer;

		// 自由相机
		private FreeCameraController freeCameraController;

		// 测试工具
		private PlayerDebugTool playerDebugTool;

		private bool isInitialized = false;

		void Awake()
		{
			Logger.LogInfo("=== Game Runtime v2 启动 ===");


		// 注册 Rewired Player.GetAxis/GetAxisRaw Harmony patch（Plan B：Rewired 层拦截）
		GoiRuntime.PlayerControl.RewiredMouseOverride.ApplyPatches();

		// 允许后台运行（游戏最小化或失去焦点时 Unity 不降速）
		Application.runInBackground = true;
			QualitySettings.vSyncCount  = 0;   // 关闭垂直同步，避免帧率受后台限制
			Application.targetFrameRate = -1;   // 不限制渲染帧率上限
			Logger.LogInfo("后台运行模式已启用（runInBackground=true, vSync=0）");

			// 加载配置
			config = RuntimeConfig.Load();
			config.Save(); // 首次运行时创建默认配置

			// 初始化模式管理器
			modeManager = new ModeManager();
			modeManager.Initialize(config);
			modeManager.PrintModeInfo();

			// 订阅场景加载事件
			UnityEngine.SceneManagement.SceneManager.sceneLoaded += OnSceneLoaded;

			Logger.LogInfo("运行时管理器初始化完成");
		}

		void OnDestroy()
		{
			UnityEngine.SceneManagement.SceneManager.sceneLoaded -= OnSceneLoaded;
			ShutdownServices();
		}

	/// <summary>
	/// 场景加载时初始化服务
	/// </summary>
	private void OnSceneLoaded(UnityEngine.SceneManagement.Scene scene, UnityEngine.SceneManagement.LoadSceneMode mode)
	{
		Logger.LogInfo($"场景加载: {scene.name}");

		// 如果是 Loader 场景（主菜单），自动加载 Main 场景
		if (scene.name == "Loader")
		{
			StartCoroutine(AutoLoadMainScene());
			return;
		}

		// 只在主场景初始化
		if (scene.name != "Mian")
		{
			Logger.LogInfo($"当前场景不是 Mian 场景（当前: {scene.name}），跳过初始化");
			return;
		}

		if (isInitialized)
		{
			Logger.LogWarning("服务已初始化，跳过");
			return;
		}

		// 根据模式初始化服务
		if (modeManager.IsDataCollectionMode())
		{
			InitializeDataCollectionMode();
		}
		else if (modeManager.IsGameRuntimeMode())
		{
			InitializeGameRuntimeMode();
		}
		else if (modeManager.IsGameTestingMode())
		{
			InitializeGameTestingMode();
		}

		isInitialized = true;
	}

	/// <summary>
	/// 自动加载 Main 场景的协程
	/// </summary>
	private System.Collections.IEnumerator AutoLoadMainScene()
	{
		// 等待 1 秒，让 Loader 场景完全加载
		yield return new WaitForSeconds(1f);
		
		Logger.LogInfo("🔍 尝试点击新游戏按钮...");
		
		// 尝试点击新游戏按钮，最多重试 10 次
		int retryCount = 0;
		int maxRetries = 10;
		bool success = false;
		
		while (retryCount < maxRetries && !success)
		{
			success = TryClickNewGameButton();
			
			if (success)
			{
				Logger.LogInfo("已点击新游戏按钮，等待场景切换");
				break;
			}
			else
			{
				retryCount++;
				Logger.LogInfo($"未找到新游戏按钮，重试 {retryCount}/{maxRetries}");
				yield return new WaitForSeconds(0.5f);
			}
		}
		
		if (!success)
		{
			Logger.LogError("无法找到新游戏按钮，请手动点击开始");
		}
	}
	
	/// <summary>
	/// 尝试点击新游戏按钮
	/// </summary>
	private bool TryClickNewGameButton()
	{
		// 尝试已知路径
		GameObject newGameButton = GameObject.Find("Canvas/Column/NewGame");
		if (newGameButton != null)
		{
			return TryClickButton(newGameButton, "已知路径");
		}
		
		// 搜索所有按钮
		GameObject foundButton = SearchNewGameButton();
		if (foundButton != null)
		{
			return TryClickButton(foundButton, "搜索");
		}
		
		return false;
	}
	
	/// <summary>
	/// 搜索新游戏按钮
	/// </summary>
	private GameObject SearchNewGameButton()
	{
		GameObject[] allObjects = FindObjectsOfType<GameObject>();
		
		foreach (var obj in allObjects)
		{
			if (IsNewGameButton(obj))
			{
				Logger.LogInfo($"找到新游戏按钮: {obj.name}");
				return obj;
			}
		}
		
		return null;
	}
	
	/// <summary>
	/// 判断是否是新游戏按钮
	/// </summary>
	private bool IsNewGameButton(GameObject obj)
	{
		string name = obj.name.ToLower();
		
		// 根据对象名判断
		if (name.Contains("newgame") || name == "newgame" || 
		    (name.Contains("new") && name.Contains("game")))
		{
			return true;
		}
		
		// 根据按钮文本判断
		string buttonText = GetButtonText(obj);
		if (!string.IsNullOrEmpty(buttonText))
		{
			string text = buttonText.ToLower();
			if (text.Contains("new") && text.Contains("game") ||
			    text.Contains("开始") || text.Contains("新游戏"))
			{
				return true;
			}
		}
		
		return false;
	}
	
	/// <summary>
	/// 获取按钮的文本内容
	/// </summary>
	private string GetButtonText(GameObject button)
	{
		var allComponents = button.GetComponentsInChildren<Component>();
		
		foreach (var comp in allComponents)
		{
			string compName = comp.GetType().Name;
			if (compName.Contains("Text"))
			{
				// 使用反射获取text属性
				var textProp = comp.GetType().GetProperty("text");
				if (textProp != null)
				{
					var textValue = textProp.GetValue(comp, null);
					if (textValue != null) return textValue.ToString();
				}
			}
		}
		
		return "";
	}
	
	/// <summary>
	/// 尝试点击按钮
	/// </summary>
	private bool TryClickButton(GameObject button, string method)
	{
		try
		{
			Logger.LogInfo($"尝试点击按钮 [{method}]: {button.name}");
			
			// 方式1: 使用反射调用Button的onClick事件
			var buttonComponents = button.GetComponents<Component>();
			foreach (var comp in buttonComponents)
			{
				string typeName = comp.GetType().Name;
				if (typeName.Contains("Button"))
				{
					// 获取 onClick 字段
					var onClickField = comp.GetType().GetProperty("onClick");
					if (onClickField != null)
					{
						var onClick = onClickField.GetValue(comp, null);
						if (onClick != null)
						{
							// 调用 Invoke 方法
							var invokeMethod = onClick.GetType().GetMethod("Invoke");
							if (invokeMethod != null)
							{
								invokeMethod.Invoke(onClick, null);
								Logger.LogInfo($"成功点击按钮: {button.name}");
								return true;
							}
						}
					}
				}
			}
			
			Logger.LogWarning($"无法点击按钮: {button.name}");
			return false;
		}
		catch (System.Exception e)
		{
			Logger.LogError($"点击按钮失败: {e.Message}");
			return false;
		}
	}

		#region 模式初始化

		/// <summary>
		/// 初始化数据采集模式
		/// 采集环境碰撞体并导出后自动退出
		/// </summary>
		private void InitializeDataCollectionMode()
		{
			Logger.LogInfo("初始化数据采集模式");
			StartCoroutine(DataCollectionCoroutine());
		}

		private System.Collections.IEnumerator DataCollectionCoroutine()
		{
			yield return null;

			Logger.LogInfo("开始导出碰撞体数据...");
			string collidersDir = PathManager.CollidersPath;
			ColliderExporter.ExportEnvironment(Path.Combine(collidersDir, "environment.json"));

			yield return new WaitForSeconds(2f);
			Logger.LogInfo("数据采集完成，退出游戏");
			Application.Quit();

#if UNITY_EDITOR
			UnityEditor.EditorApplication.isPlaying = false;
#endif
		}

		/// <summary>
		/// 初始化游戏运行模式（帧级别 RL 训练）
		/// 启动 TCP 步进服务器、StepController、PlayerDuplicateManager
		/// </summary>
		private void InitializeGameRuntimeMode()
		{
			Logger.LogInfo("=== 初始化游戏运行模式（帧级别 RL 训练）===");

			// --- 原始 Player 服务 ---
			playerStateService = new PlayerStateService();
			if (!playerStateService.Initialize())
			{
				Logger.LogError("PlayerStateService 初始化失败，中止");
				return;
			}
			Logger.LogInfo("PlayerStateService 初始化成功");

			playerInputService = new PlayerInputService();
			if (!playerInputService.Initialize())
			{
				Logger.LogError("PlayerInputService 初始化失败，中止");
				return;
			}
			Logger.LogInfo("PlayerInputService 初始化成功");

			// --- 复制体管理器（numDuplicates > 1 时创建复制体）---
			int numAgents = Mathf.Max(1, config.numDuplicates);
			if (numAgents > 1)
			{
				duplicateManager = new PlayerDuplicateManager();
				if (duplicateManager.Initialize())
				{
					// 创建 numAgents-1 个复制体（原始 Player 算 agent 0）
					duplicateManager.CreateDuplicates(numAgents - 1, Vector3.zero);
					Logger.LogInfo($"创建了 {numAgents - 1} 个复制体（共 {numAgents} 个 agent）");
				}
				else
				{
					Logger.LogWarning("PlayerDuplicateManager 初始化失败，退化为单 agent");
					numAgents = 1;
					duplicateManager = null;
				}
			}

			// --- StepController ---
			stepController = new StepController(numAgents, config.stepFrames, config.stateDimension, config.actionDimension);
			if (!stepController.Initialize(playerStateService, playerInputService, duplicateManager))
			{
				Logger.LogError("StepController 初始化失败，中止");
				return;
			}
			Logger.LogInfo($"StepController 初始化成功（{numAgents} agent，每 step {config.stepFrames} 物理帧）");

			// --- TCP 步进服务器 ---
			tcpStepServer = new TcpStepServer(config.stateDimension);
			tcpStepServer.StartListening(config.tcpPort);
			Logger.LogInfo($"TcpStepServer 已启动，监听端口 {config.tcpPort}");

		// 激活 RL 模式：阻止 PlayerControl.Update() 读取真实鼠标
		GoiRuntime.PlayerControl.PlayerControlUpdatePatch.RlModeActive = true;
		// action 名称已确认为 "mouseX"/"mouseY"，直接激活拦截
		GoiRuntime.PlayerControl.RewiredMouseOverride.Active = true;
		Logger.LogInfo("RL 模式已激活（Rewired 拦截 mouseX/mouseY，注入值将替换真实鼠标）");

		// --- 碰撞箱可视化（默认开启，黑底白环境 + 彩色 Player）---
		var mainCam = Camera.main;
		if (mainCam != null)
		{
			colliderVisualizer = mainCam.gameObject.AddComponent<ColliderVisualizer>();

			var roots = new System.Collections.Generic.List<GameObject>();
			roots.Add(playerStateService.GetPlayerObject());
			if (duplicateManager != null)
			{
				foreach (var dup in duplicateManager.GetAllDuplicates())
					roots.Add(dup);
			}
			colliderVisualizer.playerRoots = roots.ToArray();
			Logger.LogInfo($"ColliderVisualizer 已附加到主摄像机（{roots.Count} 个 Player，默认开启）");
			// 自由相机控制器（默认关闭，由 Python 'F' 命令开启）
			freeCameraController = mainCam.gameObject.AddComponent<FreeCameraController>();
			Logger.LogInfo("FreeCameraController 已附加到主摄像机（默认关闭）");
		}
		else
		{
			Logger.LogWarning("未找到主摄像机，ColliderVisualizer / FreeCameraController 未创建");
		}

		// --- 启动训练主循环协程 ---
		StartCoroutine(TrainingLoop());
		Logger.LogInfo("游戏运行模式初始化完成，等待 Python 连接...");
		}

		/// <summary>
		/// 训练主循环协程
		/// Python 通过 TCP 发送 RESET/STEP/CLOSE 命令；C# 执行物理步进后回包
		/// </summary>
		private System.Collections.IEnumerator TrainingLoop()
		{
			while (tcpStepServer != null && tcpStepServer.IsRunning)
			{
				// 非阻塞轮询（每帧检查一次）
				if (tcpStepServer.TryGetCommand(out StepCommand cmd))
				{
					StepResponse resp;

				try
				{
					switch (cmd.Type)
					{
						case CommandType.Reset:
							stepController.Reset();
							resp = new StepResponse
							{
								States = stepController.CollectAllStates(),
								Dones  = new bool[stepController.NumAgents],
							};
							tcpStepServer.SendResponse(resp);
							break;

						case CommandType.Step:
							float[] states = stepController.ExecuteStep(cmd.Actions ?? new float[0]);
							resp = new StepResponse
							{
								States = states,
								Dones  = new bool[stepController.NumAgents],
							};
							tcpStepServer.SendResponse(resp);
							break;

						case CommandType.NewSnapshot:
							stepController.TakeNewSnapshot();
							resp = new StepResponse
							{
								States = stepController.CollectAllStates(),
								Dones  = new bool[stepController.NumAgents],
							};
							tcpStepServer.SendResponse(resp);
							break;

						case CommandType.Config:
							// L3 可重复性测试：启用 RewiredMouseOverride，屏蔽真实鼠标
							GoiRuntime.PlayerControl.RewiredMouseOverride.Active = cmd.ConfigRewiredMouseActive;
							if (cmd.ConfigMouseXActionId >= 0)
								GoiRuntime.PlayerControl.RewiredMouseOverride.MouseXActionId = cmd.ConfigMouseXActionId;
							if (cmd.ConfigMouseYActionId >= 0)
								GoiRuntime.PlayerControl.RewiredMouseOverride.MouseYActionId = cmd.ConfigMouseYActionId;
							Logger.LogInfo($"[TrainingLoop] RewiredMouseOverride: Active={cmd.ConfigRewiredMouseActive} MouseX={cmd.ConfigMouseXActionId} MouseY={cmd.ConfigMouseYActionId}");
							resp = new StepResponse { States = new float[0], Dones = new bool[0] };
							tcpStepServer.SendResponse(resp);
							break;

					case CommandType.ExportColliders:
						ColliderExporter.ExportAll(playerStateService.GetPlayerObject(), PathManager.CollidersPath);
						resp = new StepResponse { States = new float[0], Dones = new bool[0] };
						tcpStepServer.SendResponse(resp);
						break;

					case CommandType.Teleport:
						float[] teleStates = stepController.Teleport(
							cmd.TeleportAgentIndex,
							new Vector2(cmd.TeleportX, cmd.TeleportY));
						resp = new StepResponse
						{
							States = teleStates,
							Dones  = new bool[stepController.NumAgents],
						};
						tcpStepServer.SendResponse(resp);
						break;

						case CommandType.Visualize:
							if (colliderVisualizer != null)
							{
								colliderVisualizer.enabled = cmd.VisualizeEnabled;
								Logger.LogInfo($"[TrainingLoop] ColliderVisualizer {(cmd.VisualizeEnabled ? "开启" : "关闭")}");
							}
							resp = new StepResponse { States = new float[0], Dones = new bool[0] };
							tcpStepServer.SendResponse(resp);
							break;

					case CommandType.CameraFree:
						if (freeCameraController != null)
						{
							freeCameraController.SetFreeMode(cmd.CameraFreeEnabled);
							Logger.LogInfo($"[TrainingLoop] FreeCameraController {(cmd.CameraFreeEnabled ? "启用" : "禁用")}");
						}
						resp = new StepResponse { States = new float[0], Dones = new bool[0] };
						tcpStepServer.SendResponse(resp);
						break;

						case CommandType.Close:
							Logger.LogInfo("[TrainingLoop] 收到 CLOSE，退出训练循环");
							tcpStepServer.Stop();
							yield break;
					}
				}
				catch (System.Exception ex)
				{
					Logger.LogError($"[TrainingLoop] cmd={cmd.Type} 处理异常: {ex.GetType().Name}: {ex.Message}\n{ex.StackTrace}");
					// 发送空回包，避免 Python 永久阻塞
					try
					{
						tcpStepServer.SendResponse(new StepResponse
						{
							States = new float[stepController.NumAgents * 29],
							Dones  = new bool[stepController.NumAgents],
						});
					}
					catch { }
				}
				}

				yield return null;
			}
		}

		/// <summary>
		/// 初始化游戏测试模式
		/// 启动测试工具
		/// </summary>
		private void InitializeGameTestingMode()
		{
			Logger.LogInfo("=== 初始化游戏测试模式 ===");

			// 初始化游戏控制服务（基础服务）
			gameControlService = new GameControlService();
			Logger.LogInfo("GameControlService 初始化成功");

			// 初始化 Player 状态采集服务
			playerStateService = new PlayerStateService();
			if (playerStateService.Initialize())
			{
				Logger.LogInfo("PlayerStateService 初始化成功");
			}
			else
			{
				Logger.LogError("PlayerStateService 初始化失败");
			}

			// 初始化 Player 输入控制服务
			playerInputService = new PlayerInputService();
			if (playerInputService.Initialize())
			{
				Logger.LogInfo("PlayerInputService 初始化成功");
			}
			else
			{
				Logger.LogError("PlayerInputService 初始化失败");
			}

			// 初始化 Player 调试工具（使用所有服务）
			playerDebugTool = new PlayerDebugTool();
			if (playerDebugTool.Initialize(playerInputService, playerStateService, gameControlService))
			{
				Logger.LogInfo("PlayerDebugTool 初始化成功");
			}

			Logger.LogInfo("游戏测试模式初始化完成");

			// 自动运行物理能力探测（结果输出到 BepInEx 日志）
			StartCoroutine(PhysicsProbeCoroutine());
		}

		/// <summary>
		/// 物理能力探测协程
		/// 验证 Physics2D.Simulate 在此游戏中是否可靠，结果写入日志供后续分析
		/// </summary>
		private System.Collections.IEnumerator PhysicsProbeCoroutine()
		{
			// 等待两帧，确保所有服务就绪
			yield return null;
			yield return null;

			Logger.LogInfo("=== [PhysicsProbe] 开始物理能力探测 ===");

			// --- 1. 基础物理参数 ---
			Logger.LogInfo($"[PhysicsProbe] Time.fixedDeltaTime    = {Time.fixedDeltaTime:F6} s ({1f / Time.fixedDeltaTime:F1} Hz)");
			Logger.LogInfo($"[PhysicsProbe] Time.timeScale         = {Time.timeScale}");
			Logger.LogInfo($"[PhysicsProbe] Physics2D.simulationMode = {Physics2D.simulationMode}");
			Logger.LogInfo($"[PhysicsProbe] Physics2D.gravity      = {Physics2D.gravity}");

			// --- 2. 采样原始 Player 位置 ---
			Vector2 pos0 = playerStateService != null ? playerStateService.GetPlayerPosition() : Vector2.zero;
			Logger.LogInfo($"[PhysicsProbe] 探测前 Player 位置: ({pos0.x:F4}, {pos0.y:F4})");

			// --- 3. 切换到 Script 模式，调用一次 Simulate ---
			var originalMode = Physics2D.simulationMode;
			Physics2D.simulationMode = SimulationMode2D.Script;
			Logger.LogInfo($"[PhysicsProbe] 切换 simulationMode → Script，当前值: {Physics2D.simulationMode}");

			Physics2D.Simulate(Time.fixedDeltaTime);

			Vector2 pos1 = playerStateService != null ? playerStateService.GetPlayerPosition() : Vector2.zero;
			Logger.LogInfo($"[PhysicsProbe] Simulate(fixedDeltaTime) 后 Player 位置: ({pos1.x:F4}, {pos1.y:F4})");
			Logger.LogInfo($"[PhysicsProbe] 位置变化量: Δ=({pos1.x - pos0.x:F4}, {pos1.y - pos0.y:F4})");

			// --- 4. 验证持久性：等一帧，检查位置是否自动又变了（说明自动物理没有被真正停止） ---
			yield return null;

			Vector2 pos2 = playerStateService != null ? playerStateService.GetPlayerPosition() : Vector2.zero;
			float autoAdvanceDelta = Vector2.Distance(pos1, pos2);
			Logger.LogInfo($"[PhysicsProbe] Script 模式下自然帧后位置: ({pos2.x:F4}, {pos2.y:F4})");
			Logger.LogInfo($"[PhysicsProbe] 自然帧位置变化量: {autoAdvanceDelta:F4}");
			if (autoAdvanceDelta < 0.0001f)
				Logger.LogInfo("[PhysicsProbe] 结论: Script 模式下游戏物理已停止自动推进 [可用于帧级别控制]");
			else
				Logger.LogWarning($"[PhysicsProbe] 结论: Script 模式下仍有自动物理推进 (delta={autoAdvanceDelta:F4}) [需要额外处理]");

			// --- 5. 恢复原始模式 ---
			Physics2D.simulationMode = originalMode;
		Logger.LogInfo($"[PhysicsProbe] 已恢复 simulationMode → {originalMode}");
		Logger.LogInfo("=== [PhysicsProbe] 探测完成 ===");
		}

		#endregion

		#region 服务关闭

		/// <summary>
		/// 关闭所有服务
		/// </summary>
		private void ShutdownServices()
		{
			Logger.LogInfo("关闭所有服务...");

			tcpStepServer?.Stop();
			tcpStepServer = null;
			stepController = null;
			duplicateManager = null;
			if (colliderVisualizer != null)
			{
				Destroy(colliderVisualizer);
				colliderVisualizer = null;
			}
			if (freeCameraController != null)
			{
				Destroy(freeCameraController);
				freeCameraController = null;
			}
			playerInputService = null;
			playerStateService = null;
			gameControlService = null;
			playerDebugTool = null;

			Logger.LogInfo("所有服务已关闭");
		}

		#endregion

		#region 调试命令

		void Update()
		{
			// 游戏测试模式：调用调试工具更新
			if (modeManager.IsGameTestingMode() && playerDebugTool != null)
			{
				playerDebugTool.Update();
			}
		}
		
		void LateUpdate()
		{
			// 游戏测试模式：调用调试工具的 LateUpdate（单帧步进支持）
			if (modeManager.IsGameTestingMode() && playerDebugTool != null)
			{
				playerDebugTool.LateUpdate();
			}
		}

		#endregion
	}
}

