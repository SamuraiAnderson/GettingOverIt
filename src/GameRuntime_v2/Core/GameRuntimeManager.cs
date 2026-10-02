using BepInEx;
using UnityEngine;
using System.IO;
using GoiRuntime.Core.Configuration;
using GoiRuntime.Core.Interfaces;
using GoiRuntime.Core.Protocol;
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


		// 训练核心（GameRuntime 模式）
		private IMultiPlayerRuntime players;
		private TcpStepServer tcpStepServer;

		// 碰撞箱可视化
		private ColliderVisualizer colliderVisualizer;

		// 自由相机
		private FreeCameraController freeCameraController;

		// 测试工具
		private PlayerDebugTool playerDebugTool;

		private bool isInitialized = false;
        private bool debugWorlds, runtimeBusy;


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
            DynamicsProbe.Configure(config);
			config.Save(); // 首次运行时创建默认配置

			// 初始化模式管理器
			modeManager = new ModeManager();
			modeManager.Initialize(config);
            if (config.suppressHubDepthWrite && modeManager.IsGameRuntimeMode())
            {
                HubDepthWritePatch.Apply();
                Logger.LogInfo("Hub visual-depth write suppressed; native Awake initialization preserved");
            }
            PlayerControlFixedUpdatePatch.Enabled = modeManager.IsGameRuntimeMode() && !DynamicsProbe.IsNative;
            RewiredMouseOverride.NativeInputSmoothing = PlayerControlFixedUpdatePatch.Enabled;
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
            if (DynamicsProbe.IsNative)
            {
                DynamicsProbe.BindScene();
                isInitialized = true;
                return;
            }
            isInitialized = true;
            players = new IsolatedWorldRuntime(config);
            StartCoroutine(StartRuntime());
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

        // Flatten nested coroutines so initialization/replay failures always close
        // the TCP connection rather than leaving a client blocked on a response.
        private System.Collections.IEnumerator RunGuarded(System.Collections.IEnumerator routine)
        {
            var stack = new System.Collections.Generic.Stack<System.Collections.IEnumerator>();
            stack.Push(routine);
            runtimeBusy = true;
            try
            {
                while (stack.Count > 0)
                {
                    bool more = false;
                    object current = null;
                    try
                    {
                        more = stack.Peek().MoveNext();
                        if (more) current = stack.Peek().Current;
                    }
                    catch (System.Exception ex)
                    {
                        Logger.LogError("Multi-player runtime failed: " + ex);
                        tcpStepServer?.Stop();
                        players = null;
                        yield break;
                    }
                    if (!more) { stack.Pop(); continue; }
                    var nested = current as System.Collections.IEnumerator;
                    if (nested != null) stack.Push(nested);
                    else yield return current;
                }
            }
            finally
            {
                runtimeBusy = false;
                foreach (var item in stack)
                {
                    var disposable = item as System.IDisposable;
                    if (disposable != null) disposable.Dispose();
                }
            }
        }

        private System.Collections.IEnumerator StartRuntime()
        {
            yield return RunGuarded(players.Initialize());
            if (players == null) yield break;
            if (debugWorlds) yield break;
            var calibration = AnchorCalibration.Load();
            var hello = new HelloSnapshot { NumAgents = players.NumAgents,
                FixedDeltaTime = Time.fixedDeltaTime, StepFrames = config.stepFrames,
                CalibrationAvailable = calibration.Available, AnchorId = calibration.AnchorId ?? "",
                CalibrationSha256 = calibration.Sha256 ?? "", PhysicsBackend = "isolated_worlds",
                PhysicsWorldCount = players.NumAgents };
            tcpStepServer = new TcpStepServer(StepController.STATE_DIM, StepController.LEGACY_WIRE_DIM, hello);
            tcpStepServer.StartListening(config.tcpPort);
            Logger.LogInfo("Multi-player runtime ready: " + players.NumAgents + " independent physics worlds");
            StartCoroutine(TrainingLoop());
        }

        private void StartDebugWorlds(int count)
        {
            playerDebugTool = null; // Native single-player hotkeys must not mutate controlled worlds.
            debugWorlds = true;
            config.numAgents = count;
            config.experimentalResetDiagnostics = true;
            HubDepthWritePatch.Apply();
            players = new IsolatedWorldRuntime(config);
            StartCoroutine(StartRuntime());
        }

		private StepResponse StatePacket(float[] states = null, string diagnosticJson = null)
		{
			bool anchorsOk = diagnosticJson != null || players.LastAnchorCollectOk;
			return new StepResponse
			{
				States = states,
				Dones = new bool[players.NumAgents],
				Sequence = players.LastCollectSequence,
				ErrorCode = anchorsOk ? 0 : 1,
				ErrorMessage = anchorsOk ? null : "reference_anchor_unavailable",
				DiagnosticJson = diagnosticJson,
			};
		}

		private StepResponse EmptyPacket()
		{
			return new StepResponse
			{
				States = new float[0],
				Dones = new bool[0],
				Sequence = players.LastCollectSequence,
			};
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
                    if (cmd.Type == CommandType.Reset)
                    {
                        yield return ResetSceneAndRespond();
                        continue;
                    }
					StepResponse resp;

				try
				{
						switch (cmd.Type)
						{
                        case CommandType.DynamicsProbe:
                            DynamicsProbe.BeginCandidate();
                            tcpStepServer.SendResponse(StatePacket(players.CollectAllStates()));
                            break;
                        case CommandType.ResetDiagnostics:
                            tcpStepServer.SendResponse(StatePacket(
                                diagnosticJson: players.CollectResetDiagnostics()));
                            break;
                        case CommandType.ObservationDiagnostics:
                            tcpStepServer.SendResponse(StatePacket(
                                diagnosticJson: players.CollectObservationDiagnostics()));
                            break;
						case CommandType.CalibrationCapture:
							tcpStepServer.SendResponse(StatePacket(
								diagnosticJson: players.CollectCalibrationCapture()));
							break;

						case CommandType.Step:
							float[] states = players.ExecuteStep(cmd.Actions ?? new float[0]);
							resp = StatePacket(states);
							tcpStepServer.SendResponse(resp);
							break;

						case CommandType.NewSnapshot:
							players.TakeNewSnapshot();
							resp = StatePacket(players.CollectAllStates());
							tcpStepServer.SendResponse(resp);
							break;

                        case CommandType.Config:
                            players.ConfigureInput(cmd.ConfigRewiredMouseActive, cmd.ConfigMouseXActionId, cmd.ConfigMouseYActionId);
							resp = EmptyPacket();
							tcpStepServer.SendResponse(resp);
							break;

					case CommandType.ExportColliders:
                        players.ExportColliders(PathManager.CollidersPath);
						resp = EmptyPacket();
						tcpStepServer.SendResponse(resp);
						break;

					case CommandType.Teleport:
						float[] teleStates = players.Teleport(
							cmd.TeleportAgentIndex,
							new Vector2(cmd.TeleportX, cmd.TeleportY));
						resp = StatePacket(teleStates);
						tcpStepServer.SendResponse(resp);
						break;

                        case CommandType.Visualize:
                            players.SetVisualization(cmd.VisualizeEnabled);
                            tcpStepServer.SendResponse(EmptyPacket());
                            break;
                        case CommandType.CameraFree:
                            players.SetCameraFree(cmd.CameraFreeEnabled);
                            tcpStepServer.SendResponse(EmptyPacket());
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
                    // Legacy framing cannot express command errors; never fabricate observations.
                    tcpStepServer.Stop();
				}
				}

				yield return null;
			}
		}

        private System.Collections.IEnumerator ResetSceneAndRespond()
        {
            yield return RunGuarded(players.Reset());
            if (players != null && tcpStepServer != null && tcpStepServer.IsRunning)
                tcpStepServer.SendResponse(StatePacket(players.CollectAllStates()));
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
            playerDebugTool.StartMultiPlayer = StartDebugWorlds;
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

		#region 训练环境安全

		/// <summary>
		/// 销毁场景中所有 RestartOnContact 组件，防止落水触发重置链路。
		/// 
		/// 必须使用 Destroy 而非 enabled=false，因为 Unity 的碰撞回调
		/// (OnCollisionEnter2D) 不受 MonoBehaviour.enabled 控制，
		/// 只要组件存在就会被物理引擎调用。
		/// </summary>
		private void DisableWaterReset()
		{
			int destroyedCount = 0;
			MonoBehaviour[] allScripts = FindObjectsOfType<MonoBehaviour>();
			foreach (var script in allScripts)
			{
				if (script != null && script.GetType().Name == "RestartOnContact")
				{
					string objName = script.gameObject.name;
					Destroy(script);
					destroyedCount++;
					Logger.LogInfo($"已销毁 RestartOnContact: {objName}");
				}
			}

			if (destroyedCount > 0)
				Logger.LogInfo($"共销毁 {destroyedCount} 个水体重置组件 — 落水不再触发视角重置和物理模式切换");
			else
				Logger.LogWarning("未找到 RestartOnContact 组件（水体重置可能使用了不同机制）");
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
			players = null;
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
            if (debugWorlds)
            {
                if (runtimeBusy || players == null) return;
                if (Input.GetKeyDown(KeyCode.F9)) StartCoroutine(RunGuarded(players.Reset()));
                if (Input.GetKeyDown(KeyCode.F10))
                {
                    var actions = new float[players.NumAgents * 2];
                    for (int i = 0; i < players.NumAgents; i++) actions[2*i] = i % 2 == 0 ? 50 : -50;
                    players.ExecuteStep(actions);
                }
                if (Input.GetKeyDown(KeyCode.F11)) Logger.LogInfo(players.CollectResetDiagnostics());
                if (Input.GetKeyDown(KeyCode.F12)) StartDebugWorlds(players.NumAgents == 1 ? 4 : 1);
                return;
            }

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
