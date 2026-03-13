using BepInEx;
using UnityEngine;
using GoiRuntime.Core.Configuration;
using GoiRuntime.Core.Events;
using GoiRuntime.Core.Interfaces;
using GoiRuntime.Core.Services;
using GoiRuntime.ColliderCollection;
using GoiRuntime.PlayerControl;
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
		
		// 碰撞箱服务
		private EnvironmentColliderService environmentColliderService;
		private PlayerColliderService playerColliderService;
		
		// Player 控制服务
		private PlayerInputService playerInputService;
		private PlayerStateService playerStateService;
		
		// 测试工具
		private PlayerDebugTool playerDebugTool;

		private bool isInitialized = false;

		void Awake()
		{
			Logger.LogInfo("=== Game Runtime v2 启动 ===");

			// 加载配置
			config = RuntimeConfig.Load();
			config.Save(); // 首次运行时创建默认配置

			// 初始化模式管理器
			modeManager = new ModeManager();
			modeManager.Initialize(config);
			modeManager.PrintModeInfo();

			// 订阅场景加载事件
			UnityEngine.SceneManagement.SceneManager.sceneLoaded += OnSceneLoaded;

			Logger.LogInfo("✅ 运行时管理器初始化完成");
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
				Logger.LogInfo("✅ 已点击新游戏按钮，等待场景切换");
				break;
			}
			else
			{
				retryCount++;
				Logger.LogInfo($"⚠️ 未找到新游戏按钮，重试 {retryCount}/{maxRetries}");
				yield return new WaitForSeconds(0.5f);
			}
		}
		
		if (!success)
		{
			Logger.LogError("❌ 无法找到新游戏按钮，请手动点击开始");
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
								Logger.LogInfo($"✅ 成功点击按钮: {button.name}");
								return true;
							}
						}
					}
				}
			}
			
			Logger.LogWarning($"⚠️ 无法点击按钮: {button.name}");
			return false;
		}
		catch (System.Exception e)
		{
			Logger.LogError($"❌ 点击按钮失败: {e.Message}");
			return false;
		}
	}

		#region 模式初始化

		/// <summary>
		/// 初始化数据采集模式
		/// 仅启动碰撞箱采集服务
		/// </summary>
		private void InitializeDataCollectionMode()
		{
			Logger.LogInfo("📦 初始化数据采集模式");

			// 创建环境碰撞箱采集服务
			environmentColliderService = new EnvironmentColliderService();
			if (environmentColliderService.Initialize())
			{
				Logger.LogInfo("✅ 环境碰撞箱服务初始化成功");

				// 自动采集并导出 Mountain 数据
				StartCoroutine(CollectMountainDataCoroutine());
			}
			else
			{
				Logger.LogError("❌ 环境碰撞箱服务初始化失败");
			}

			Logger.LogInfo("💡 数据采集模式已启动，等待采集完成后游戏将自动退出");
		}

		/// <summary>
		/// 采集 Mountain 数据的协程
		/// </summary>
		private System.Collections.IEnumerator CollectMountainDataCoroutine()
		{
			// 等待一帧，确保场景完全加载
			yield return null;

			Logger.LogInfo("开始采集 Mountain 碰撞箱数据...");

			var colliders = environmentColliderService.CollectEnvironmentColliders();
			
			if (colliders != null && colliders.Length > 0)
			{
				Logger.LogInfo($"✅ 采集到 {colliders.Length} 个碰撞箱");

				// 导出数据
				string exportPath = environmentColliderService.ExportEnvironmentColliders();
				Logger.LogInfo($"📁 数据已导出到: {exportPath}");

				// 发布采集完成事件
				EventBus.Publish(GameEvents.ColliderDataExported, exportPath);
			}
			else
			{
				Logger.LogError("❌ 未能采集到 Mountain 碰撞箱");
			}

			// 等待 2 秒后退出游戏
			yield return new WaitForSeconds(2f);
			Logger.LogInfo("🚪 数据采集完成，退出游戏");
			Application.Quit();

#if UNITY_EDITOR
			UnityEditor.EditorApplication.isPlaying = false;
#endif
		}

		/// <summary>
		/// 初始化游戏运行模式
		/// 启动 UDP 通信、Player 控制等
		/// </summary>
		private void InitializeGameRuntimeMode()
		{
			Logger.LogInfo("🎮 初始化游戏运行模式");

			// 初始化游戏控制服务（基础服务）
			gameControlService = new GameControlService();
			Logger.LogInfo("✅ GameControlService 初始化成功");

			// 初始化 Player 状态采集服务
			playerStateService = new PlayerStateService();
			if (playerStateService.Initialize())
			{
				Logger.LogInfo("✅ PlayerStateService 初始化成功");
			}
			else
			{
				Logger.LogError("❌ PlayerStateService 初始化失败");
			}

			// 初始化 Player 输入控制服务
			playerInputService = new PlayerInputService();
			if (playerInputService.Initialize())
			{
				Logger.LogInfo("✅ PlayerInputService 初始化成功");
			}
			else
			{
				Logger.LogError("❌ PlayerInputService 初始化失败");
			}

			// 初始化 Player 碰撞箱采集服务
			GameObject player = playerStateService.GetPlayerObject();
			if (player != null)
			{
				playerColliderService = new PlayerColliderService();
				if (playerColliderService.Initialize(player))
				{
					Logger.LogInfo("✅ PlayerColliderService 初始化成功");
					playerColliderService.CollectPlayerColliders(player);
					playerColliderService.PrintColliderInfo();
				}
			}

			Logger.LogInfo("🎮 游戏运行模式初始化完成");
		}

		/// <summary>
		/// 初始化游戏测试模式
		/// 启动测试工具
		/// </summary>
		private void InitializeGameTestingMode()
		{
			Logger.LogInfo("🧪 初始化游戏测试模式");

			// 初始化游戏控制服务（基础服务）
			gameControlService = new GameControlService();
			Logger.LogInfo("✅ GameControlService 初始化成功");

			// 初始化 Player 状态采集服务
			playerStateService = new PlayerStateService();
			if (playerStateService.Initialize())
			{
				Logger.LogInfo("✅ PlayerStateService 初始化成功");
			}
			else
			{
				Logger.LogError("❌ PlayerStateService 初始化失败");
			}

			// 初始化 Player 输入控制服务
			playerInputService = new PlayerInputService();
			if (playerInputService.Initialize())
			{
				Logger.LogInfo("✅ PlayerInputService 初始化成功");
			}
			else
			{
				Logger.LogError("❌ PlayerInputService 初始化失败");
			}

			// 初始化 Player 碰撞箱采集服务
			GameObject player = playerStateService.GetPlayerObject();
			if (player != null)
			{
				playerColliderService = new PlayerColliderService();
				if (playerColliderService.Initialize(player))
				{
					Logger.LogInfo("✅ PlayerColliderService 初始化成功");
				}
			}

			// 初始化 Player 调试工具（使用所有服务）
			playerDebugTool = new PlayerDebugTool();
			if (playerDebugTool.Initialize(playerInputService, playerStateService, playerColliderService, gameControlService))
			{
				Logger.LogInfo("✅ PlayerDebugTool 初始化成功");
			}

			Logger.LogInfo("🧪 游戏测试模式初始化完成");
		}

		#endregion

		#region 服务关闭

		/// <summary>
		/// 关闭所有服务
		/// </summary>
		private void ShutdownServices()
		{
			Logger.LogInfo("关闭所有服务...");

			// TODO: 关闭所有服务
			// if (udpReceiveService != null) udpReceiveService.Shutdown();
			// if (udpSendService != null) udpSendService.Shutdown();

			environmentColliderService = null;
			playerColliderService = null;

			Logger.LogInfo("✅ 所有服务已关闭");
		}

		#endregion

		#region 调试命令

		void Update()
		{
			// F1: 打印模式信息（所有模式）
			if (Input.GetKeyDown(KeyCode.F1))
			{
				modeManager.PrintModeInfo();
			}

			// F2: 手动采集环境碰撞箱（仅数据采集模式）
			if (Input.GetKeyDown(KeyCode.F2) && modeManager.IsDataCollectionMode())
			{
				if (environmentColliderService != null)
				{
					var colliders = environmentColliderService.CollectEnvironmentColliders();
					Logger.LogInfo($"手动采集: {colliders?.Length ?? 0} 个碰撞箱");
				}
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

