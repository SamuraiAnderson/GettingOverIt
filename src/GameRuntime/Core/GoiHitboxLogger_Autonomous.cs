
using BepInEx;
using UnityEngine;
using HarmonyLib;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using UnityEngine.SceneManagement;
using System.Runtime.InteropServices;

namespace GoiHitboxLogger
{
	/// <summary>
	/// GOI自主数据采集器 - 完全自包含，无需外部脚本
	/// 自动完成：场景检测 → UI操作 → 数据采集 → 结果导出
	/// </summary>
	[BepInPlugin("com.symbol.goi.autonomous_collector", "GOI Autonomous Data Collector", "2.0.0")]
	public class GoiHitboxLoggerAutonomous : BaseUnityPlugin
	{
		#region Windows API 声明
		
		[StructLayout(LayoutKind.Sequential)]
		public struct POINT
		{
			public int X;
			public int Y;
		}
		
		// Windows API - 鼠标相关
		[DllImport("user32.dll", SetLastError = true)]
		public static extern bool GetCursorPos(out POINT lpPoint);
		
		// 相对鼠标输入API
		[DllImport("user32.dll")]
		private static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint cButtons, UIntPtr dwExtraInfo);
		
		// mouse_event常量
		private const uint MOUSEEVENTF_MOVE = 0x0001;
		
		// Windows API 包装器方法 - 用于Harmony Hook
		public static bool GetCursorPosWrapper(out POINT lpPoint)
		{
			return GetCursorPos(out lpPoint);
		}
		
		#endregion
		#region 配置参数
		
		// 固定采集参数 - 用户可在此修改
		private const float COLLECTION_DURATION = 5.0f;    // 采集时长（秒）
		private const int COLLECTION_FREQUENCY = 30;       // 采样频率（Hz）
		private const float MIAN_SCENE_WAIT_TIME = 3.0f;   // Mian场景等待时间（秒）
		private const float NEW_GAME_RETRY_INTERVAL = 2.0f; // 新游戏按钮重试间隔（秒）
		private const float SCENE_SWITCH_TIMEOUT = 60.0f;  // 场景切换超时（秒）
		
		// 🔍 诊断模式 - 检查mouse_event传递链
		public static bool diagnosticMode = false; // ✅ 问题已解决，恢复正常模式
		
		#endregion
		
		#region 状态管理
		
		/// <summary>
		/// 自主采集状态
		/// </summary>
		private enum AutonomousState
		{
			Waiting,        // 等待游戏场景
			InLoader,       // 在Loader场景，处理新游戏
			WaitingMian,    // 等待Mian场景加载
			InMian,         // 在Mian场景，准备采集
			WaitingAnimation, // 等待Player开始动画完成
			Collecting,     // 正在采集数据
			Processing,     // 数据处理中
			Completed,      // 采集完成
			Error           // 出现错误
		}
		
		private AutonomousState currentState = AutonomousState.Waiting;
		private float stateStartTime;
		private float lastRetryTime;
		private string statusMessage = "";
		private float lastMousePromptTime = 0f;
		private Vector2 lastWindowsSetPos = Vector2.zero;
		
		// Harmony实例
		private static Harmony harmony;
		
		#endregion
		
		#region Unity生命周期
		
		void Awake()
		{
			// 初始化Harmony
			harmony = new Harmony("com.symbol.goi.autonomous_collector.harmony");
			harmony.PatchAll();
			Logger.LogInfo("🔧 Harmony patches applied - Input.mousePosition hook enabled");
		}
		
		void Start()
		{
			Logger.LogInfo("🤖 GOI自主数据采集器启动 - 完全自动化模式");
			Logger.LogInfo($"📊 配置参数: {COLLECTION_DURATION}秒@{COLLECTION_FREQUENCY}Hz");
			Logger.LogInfo("🎯 鼠标控制: 相对输入API (mouse_event) ✅");
			
			ChangeState(AutonomousState.Waiting);
		}
		
		/// <summary>
		/// 调查Unity输入管理器设置
		/// </summary>
		private void InvestigateInputSettings()
		{
			try
			{
				Logger.LogInfo("🔍 === Unity 输入管理器调查 ===");
				
				// 检查基本输入设置
				Logger.LogInfo($"🖱️ 鼠标存在: {Input.mousePresent}");
				// Logger.LogInfo($"🖱️ 鼠标按钮数量: {Input.mouseCount}"); // 此属性不存在于当前Unity版本
				
				// 尝试获取输入管理器的灵敏度设置
				// 注意：Unity的InputManager设置在运行时通常不可直接访问
				Logger.LogInfo("🔍 检查已知的鼠标轴设置...");
				
				// 检查鼠标轴的当前状态
				float mouseX = Input.GetAxis("Mouse X");
				float mouseY = Input.GetAxis("Mouse Y");
				Logger.LogInfo($"🎯 当前鼠标轴状态: X={mouseX:F6}, Y={mouseY:F6}");
				
				// 检查原始输入状态
				float mouseXRaw = Input.GetAxisRaw("Mouse X");
				float mouseYRaw = Input.GetAxisRaw("Mouse Y");
				Logger.LogInfo($"🎯 原始鼠标轴状态: X={mouseXRaw:F6}, Y={mouseYRaw:F6}");
				
				// 检查其他可能影响的设置
				Logger.LogInfo($"🔍 当前时间缩放: {Time.timeScale}");
				Logger.LogInfo($"🔍 固定更新时间: {Time.fixedDeltaTime:F6}");
				Logger.LogInfo($"🔍 上一帧时间: {Time.deltaTime:F6}");
				
				Logger.LogInfo("🔍 === 调查完成 ===");
			}
			catch (System.Exception ex)
			{
				Logger.LogWarning($"🔍 输入设置调查失败: {ex.Message}");
			}
		}
		
		void Update()
		{
			// 模拟鼠标移动增量（Getting Over It需要速度，不是位置！）
			SimulateMouseMovementDelta();
			
			// 更新连续跟踪器
			ContinuousTrackerAutonomous.Update();
			
			// 处理状态机
			HandleCurrentState();
		}
		
		void OnGUI()
		{
			// 显示当前状态
			GUI.Box(new Rect(10, 10, 300, 100), "");
			GUI.Label(new Rect(20, 30, 280, 20), $"🤖 自主数据采集器");
			GUI.Label(new Rect(20, 50, 280, 20), $"状态: {GetStateDisplayName()}");
			GUI.Label(new Rect(20, 70, 280, 20), statusMessage);
		}
		
		#endregion
		
		#region Windows API 鼠标控制
		
		/// <summary>
		/// 模拟鼠标移动增量（Getting Over It需要速度，不是位置！）
		/// </summary>
		private void SimulateMouseMovementDelta()
		{
			// 只在测试模式且在Collecting状态时模拟鼠标移动
			if (currentState == AutonomousState.Collecting && 
				ContinuousTrackerAutonomous.TryGetTestMousePosition(out Vector2 testPos))
			{
				// 计算移动增量
				Vector2 delta = testPos - lastWindowsSetPos;
				float distance = delta.magnitude;
				
				if (distance > 1f) // 更敏感的阈值
				{					
					// 使用相对鼠标输入API
					SendRelativeMouseInput(delta);
					
					lastWindowsSetPos = testPos;
				}
			} // 只在Collecting状态时模拟鼠标移动
		}
		
		/// <summary>
		/// 发送相对鼠标输入
		/// </summary>
		private void SendRelativeMouseInput(Vector2 delta)
		{
			// 🔧 修复坐标系：Windows → Unity Y轴翻转  
			int deltaX = (int)(delta.x);
			int deltaY = (int)(-delta.y); // Y轴预翻转，抵消Windows→Unity的翻转
			
			// 🔧 限制数值范围，避免被Unity过滤（经验值：±100以内比较安全）
			deltaX = Mathf.Clamp(deltaX, -100, 100);
			deltaY = Mathf.Clamp(deltaY, -100, 100);
			
			// 发送相对移动事件
			mouse_event(MOUSEEVENTF_MOVE, (uint)deltaX, (uint)deltaY, 0, UIntPtr.Zero);

		}
		
		#endregion
		
		#region 状态机核心逻辑
		
		/// <summary>
		/// 处理当前状态
		/// </summary>
		private void HandleCurrentState()
		{
			switch (currentState)
			{
				case AutonomousState.Waiting:
					HandleWaitingState();
					break;
					
				case AutonomousState.InLoader:
					HandleLoaderState();
					break;
					
				case AutonomousState.WaitingMian:
					HandleWaitingMianState();
					break;
					
				case AutonomousState.InMian:
					HandleMianState();
					break;
					
				case AutonomousState.WaitingAnimation:
					HandleWaitingAnimationState();
					break;
					
				case AutonomousState.Collecting:
					HandleCollectingState();
					break;
					
				case AutonomousState.Processing:
					HandleProcessingState();
					break;
					
				case AutonomousState.Completed:
					// 采集完成，什么都不做
					break;
					
				case AutonomousState.Error:
					// 错误状态，等待手动重启
					break;
			}
		}
		
		/// <summary>
		/// 切换状态
		/// </summary>
		private void ChangeState(AutonomousState newState)
		{
			if (currentState != newState)
			{
				Logger.LogInfo($"🔄 状态切换: {currentState} → {newState}");
				currentState = newState;
				stateStartTime = Time.time;
				lastRetryTime = Time.time;
				
				// 进入数据采集状态时重置鼠标提示时间
				if (newState == AutonomousState.Collecting)
				{
					lastMousePromptTime = 0f;
				}
			}
		}
		
		#endregion
		
		#region 各状态处理方法
		
		/// <summary>
		/// 等待状态 - 检测当前场景
		/// </summary>
		private void HandleWaitingState()
		{
			string sceneName = SceneManager.GetActiveScene().name;
			statusMessage = $"检测场景: {sceneName}";
			
			if (sceneName.ToLower().Contains("loader"))
			{
				ChangeState(AutonomousState.InLoader);
			}
			else if (sceneName.ToLower().Contains("mian"))
			{
				ChangeState(AutonomousState.InMian);
			}
		}
		
		/// <summary>
		/// Loader场景处理 - 自动点击新游戏
		/// </summary>
		private void HandleLoaderState()
		{
			statusMessage = "正在Loader场景，尝试开始新游戏...";
			
			// 每隔一段时间重试点击新游戏按钮
			if (Time.time - lastRetryTime >= NEW_GAME_RETRY_INTERVAL)
			{
				if (TryClickNewGameButton())
				{
					Logger.LogInfo("✅ 已点击新游戏按钮，等待场景切换");
					// 给游戏一点时间处理点击事件
					lastRetryTime = Time.time;
					ChangeState(AutonomousState.WaitingMian);
				}
				else
				{
					Logger.LogInfo("⚠️ 未找到新游戏按钮，继续重试");
					lastRetryTime = Time.time;
				}
			}
		}
		
		/// <summary>
		/// 等待Mian场景加载
		/// </summary>
		private void HandleWaitingMianState()
		{
			string sceneName = SceneManager.GetActiveScene().name;
			statusMessage = $"等待Mian场景... 当前:{sceneName}";
			
			if (sceneName.ToLower().Contains("mian"))
			{
				Logger.LogInfo("✅ 已进入Mian场景");
				ChangeState(AutonomousState.InMian);
			}
			else if (Time.time - stateStartTime > SCENE_SWITCH_TIMEOUT)
			{
				Logger.LogError($"❌ 等待Mian场景超时 ({SCENE_SWITCH_TIMEOUT}秒)，当前场景: {sceneName}");
				statusMessage = "错误: Mian场景加载超时";
				ChangeState(AutonomousState.Error);
			}
		}
		
		/// <summary>
		/// Mian场景处理 - 等待游戏完全加载
		/// </summary>
		private void HandleMianState()
		{
			statusMessage = "在Mian场景，检查游戏状态...";
			
			// 等待一段时间让游戏完全加载
			if (Time.time - stateStartTime < MIAN_SCENE_WAIT_TIME)
			{
				return;
			}
			
			// 检查游戏是否完全加载
			if (IsGameFullyLoaded())
			{
				Logger.LogInfo("✅ 游戏完全加载，等待Player动画完成");
				ChangeState(AutonomousState.WaitingAnimation);
			}
			else if (Time.time - stateStartTime > 30f)
			{
				Logger.LogError("❌ 游戏加载检查超时");
				statusMessage = "错误: 游戏加载超时";
				ChangeState(AutonomousState.Error);
			}
		}
		
		/// <summary>
		/// 等待Player开始动画完成
		/// </summary>
		private void HandleWaitingAnimationState()
		{
			statusMessage = "等待Player开始动画完成...";
			
			// 等待动画时间 (Getting Over It的开始动画通常3-5秒)
			float animationWaitTime = 5.0f;
			
			if (Time.time - stateStartTime < animationWaitTime)
			{
				return;
			}
			
			// 检查Player是否处于稳定状态（动画结束）或超时
			if (IsPlayerAnimationComplete() || Time.time - stateStartTime > 15f)
			{
				if (Time.time - stateStartTime > 15f)
				{
					Logger.LogWarning("⚠️ Player动画等待超时，强制开始数据采集");
				}
				else
				{
					Logger.LogInfo("🚀 Player动画完成，开始数据采集");
				}
				
				// 初始化跟踪器并开始采集
				if (ContinuousTrackerAutonomous.Initialize())
				{
					ContinuousTrackerAutonomous.SetSamplingRate(COLLECTION_FREQUENCY);
					StartDataCollection();
					ChangeState(AutonomousState.Collecting);
				}
				else
				{
					statusMessage = "错误: 跟踪器初始化失败";
					ChangeState(AutonomousState.Error);
				}
			}
		}
		
		
		/// <summary>
		/// 数据采集状态
		/// </summary>
		private void HandleCollectingState()
		{
			float elapsed = Time.time - stateStartTime;
			float remaining = COLLECTION_DURATION - elapsed;
			
			statusMessage = $"数据采集中... 剩余 {remaining:F1}秒";
			
			// 定期更新采集进度（每5秒更新一次）
			if (Time.time - lastMousePromptTime >= 5.0f)
			{
				Vector2 mousePos = ContinuousTrackerAutonomous.GetCurrentMousePosition();
				int dataPointCount = ContinuousTrackerAutonomous.GetDataPointCount();
				
				if (elapsed < 1f)
				{
					Logger.LogInfo("🤖 自动化数据采集开始");
				}
				else
				{
					Logger.LogInfo($"📊 采集进行中 ({elapsed:F0}s/{COLLECTION_DURATION}s) - 数据点: {dataPointCount}, 当前位置: ({mousePos.x:F0},{mousePos.y:F0})");
				}
				lastMousePromptTime = Time.time;
			}
			
			// 检查是否采集完成
			if (elapsed >= COLLECTION_DURATION)
			{
				Logger.LogInfo("✅ 数据采集完成，开始处理数据");
				StopDataCollection();
				ChangeState(AutonomousState.Processing);
			}
		}
		
		/// <summary>
		/// 数据处理状态
		/// </summary>
		private void HandleProcessingState()
		{
			statusMessage = "正在处理和导出数据...";
			
			// 导出跟踪数据
			string csvPath = ContinuousTrackerAutonomous.ExportTrackingData();
			
			if (!string.IsNullOrEmpty(csvPath))
			{
				Logger.LogInfo($"🎉 数据采集完成！数据已保存到: {csvPath}");
				Logger.LogInfo($"📊 采集统计: {ContinuousTrackerAutonomous.GetDataPointCount()} 数据点");
				Logger.LogInfo($"📈 数据摘要: {ContinuousTrackerAutonomous.GetDataSummary()}");
				
				statusMessage = $"✅ 采集完成! 数据点: {ContinuousTrackerAutonomous.GetDataPointCount()}";
				ChangeState(AutonomousState.Completed);
			}
			else
			{
				Logger.LogError("❌ 数据导出失败");
				statusMessage = "错误: 数据导出失败";
				ChangeState(AutonomousState.Error);
			}
		}
		
		#endregion
		
		#region 辅助方法
		
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
			
			// 搜索其他可能的按钮
			GameObject foundButton = SearchAllButtons();
			if (foundButton != null)
			{
				return TryClickButton(foundButton, "搜索");
			}
			
			return false;
		}
		
		
		/// <summary>
		/// 搜索所有可能的按钮对象
		/// </summary>
		private GameObject SearchAllButtons()
		{
			GameObject[] allObjects = FindObjectsOfType<GameObject>();
			
			// 首先查找明确的新游戏按钮
			foreach (var obj in allObjects)
			{
				if (IsLikelyButton(obj) && IsNewGameButton(obj.name, GetButtonText(obj)))
				{
					Logger.LogInfo($"找到新游戏按钮: {obj.name}");
					return obj;
				}
			}
			
			// 如果没找到，返回第一个可能的按钮
			foreach (var obj in allObjects)
			{
				if (HasButtonComponent(obj) || IsLikelyButton(obj))
				{
					Logger.LogInfo($"使用候选按钮: {obj.name}");
					return obj;
				}
			}
			
			return null;
		}
		
		/// <summary>
		/// 获取按钮的文本内容
		/// </summary>
		private string GetButtonText(GameObject button)
		{
			// 尝试获取各种可能的Text组件
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
		/// 判断是否是新游戏按钮
		/// </summary>
		private bool IsNewGameButton(string objectName, string buttonText)
		{
			string name = objectName.ToLower();
			string text = buttonText.ToLower();
			
			return name.Contains("new") && name.Contains("game") ||
				   text.Contains("new") && text.Contains("game") ||
				   text.Contains("开始") || text.Contains("新游戏") ||
				   name == "newgame" || name == "start";
		}
		
		/// <summary>
		/// 尝试点击按钮
		/// </summary>
		private bool TryClickButton(GameObject button, string method)
		{
			try
			{
				// 方式1: 使用反射调用Button的onClick事件
				var buttonComponents = button.GetComponents<Component>();
				foreach (var comp in buttonComponents)
				{
					string compTypeName = comp.GetType().Name;
					if (compTypeName.Contains("Button"))
					{
						string[] onClickNames = { "onClick", "onPress", "OnClick", "clickEvent" };
						
						foreach (var fieldName in onClickNames)
						{
							var field = comp.GetType().GetField(fieldName);
							if (field != null)
							{
								var onClick = field.GetValue(comp);
								if (onClick != null && TryInvokeEvent(onClick, fieldName))
								{
									return true;
								}
							}
							
							var prop = comp.GetType().GetProperty(fieldName);
							if (prop != null)
							{
								var onClick = prop.GetValue(comp, null);
								if (onClick != null && TryInvokeEvent(onClick, fieldName))
								{
									return true;
								}
							}
						}
					}
				}
				
				// 方式2: 发送多种点击消息
				string[] messages = { "OnClick", "OnPress", "OnMouseDown", "OnMouseUp", "OnTouch" };
				foreach (var msg in messages)
				{
					button.SendMessage(msg, SendMessageOptions.DontRequireReceiver);
				}
				
				// 方式3: 备用场景切换
				if (button.name.ToLower().Contains("new") && button.name.ToLower().Contains("game"))
				{
					string[] possibleScenes = { "Mian", "Main", "Game", "Level1" };
					foreach (var sceneName in possibleScenes)
					{
						try 
						{
							SceneManager.LoadScene(sceneName);
							break;
						}
						catch 
						{
							// 继续尝试下一个场景名称
						}
					}
				}
				
				return true;
			}
			catch (System.Exception e)
			{
				Logger.LogError($"❌ 点击按钮失败: {e.Message}");
				return false;
			}
		}
		
		/// <summary>
		/// 尝试调用事件对象
		/// </summary>
		private bool TryInvokeEvent(object eventObj, string eventName)
		{
			try
			{
				// 尝试Invoke方法
				var invokeMethod = eventObj.GetType().GetMethod("Invoke", new System.Type[0]);
				if (invokeMethod != null)
				{
					invokeMethod.Invoke(eventObj, null);
					return true;
				}
				
				// 尝试调用无参数的Invoke
				var methods = eventObj.GetType().GetMethods();
				foreach (var method in methods)
				{
					if (method.Name == "Invoke" && method.GetParameters().Length == 0)
					{
						method.Invoke(eventObj, null);
						return true;
					}
				}
				
				return false;
			}
			catch
			{
				return false;
			}
		}
		
		/// <summary>
		/// 检查GameObject是否有Button相关组件
		/// </summary>
		private bool HasButtonComponent(GameObject obj)
		{
			var components = obj.GetComponents<Component>();
			foreach (var comp in components)
			{
				if (comp.GetType().Name.Contains("Button"))
				{
					return true;
				}
			}
			return false;
		}
		
		/// <summary>
		/// 判断GameObject是否看起来像按钮
		/// </summary>
		private bool IsLikelyButton(GameObject obj)
		{
			string name = obj.name.ToLower();
			
			// 基于名称的判断
			if (name.Contains("button") || name.Contains("btn") || 
				name.Contains("new") || name.Contains("start") || 
				name.Contains("play") || name.Contains("game"))
			{
				return true;
			}
			
			// 基于文本内容的判断
			string text = GetButtonText(obj);
			if (!string.IsNullOrEmpty(text))
			{
				text = text.ToLower();
				if (text.Contains("new") || text.Contains("start") || 
					text.Contains("game") || text.Contains("开始") || 
					text.Contains("新游戏"))
				{
					return true;
				}
			}
			
			return false;
		}
		
		/// <summary>
		/// 获取GameObject的完整路径
		/// </summary>
		private string GetGameObjectFullPath(GameObject obj)
		{
			string path = obj.name;
			Transform parent = obj.transform.parent;
			
			while (parent != null)
			{
				path = parent.name + "/" + path;
				parent = parent.parent;
			}
			
			return path;
		}
		
		/// <summary>
		/// 检查游戏是否完全加载
		/// </summary>
		private bool IsGameFullyLoaded()
		{
			GameObject player = GameObject.Find("Player");
			if (player == null) return false;
			
			GameObject mountain = GameObject.Find("Mountain");
			if (mountain == null) return false;
			
			// 检查Player的Rigidbody2D
			var playerRb = player.GetComponent<Rigidbody2D>();
			if (playerRb == null) return false;
			
			// 检查位置是否合理
			if (player.transform.position.y < -10f) return false;
			
			return true;
		}
		
		
		/// <summary>
		/// 检查Player开始动画是否完成
		/// </summary>
		private bool IsPlayerAnimationComplete()
		{
			GameObject player = GameObject.Find("Player");
			if (player == null) return false;
			
			var playerRb = player.GetComponent<Rigidbody2D>();
			if (playerRb == null) return false;
			
			// 检查Player是否处于相对稳定的状态
			// Getting Over It中，开始动画结束后Player通常速度会变小
			float velocityThreshold = 0.5f; // 速度阈值
			
			if (playerRb.velocity.magnitude < velocityThreshold)
			{
				// 进一步检查是否有动画组件在播放
				var animators = player.GetComponentsInChildren<Animator>();
				foreach (var animator in animators)
				{
					if (animator != null && animator.enabled)
					{
						// 如果有动画组件在播放特定的开始动画，等待其完成
						// 这里可以检查特定的动画状态，但为简单起见，我们主要依赖速度检查
					}
				}
				
				return true;
			}
			
			return false;
		}
		
		/// <summary>
		/// 开始数据采集
		/// </summary>
		private void StartDataCollection()
		{
			ContinuousTrackerAutonomous.ClearData();
			ContinuousTrackerAutonomous.StartTracking();
		}
		
		/// <summary>
		/// 停止数据采集
		/// </summary>
		private void StopDataCollection()
		{
			ContinuousTrackerAutonomous.StopTracking();
		}
		
		/// <summary>
		/// 获取状态显示名称
		/// </summary>
		private string GetStateDisplayName()
		{
			switch (currentState)
			{
				case AutonomousState.Waiting: return "等待中";
				case AutonomousState.InLoader: return "Loader场景";
				case AutonomousState.WaitingMian: return "等待Mian场景";
				case AutonomousState.InMian: return "Mian场景准备中";
				case AutonomousState.WaitingAnimation: return "等待Player动画";
				case AutonomousState.Collecting: return "数据采集中";
				case AutonomousState.Processing: return "数据处理中";
				case AutonomousState.Completed: return "采集完成";
				case AutonomousState.Error: return "出现错误";
				default: return "未知状态";
			}
		}
		
		#endregion
		
	}
	
	/// <summary>
	/// Harmony补丁 - Hook Unity Input系统
	/// </summary>
	[HarmonyPatch(typeof(Input), "mousePosition", MethodType.Getter)]
	public static class InputMousePositionPatch
	{
		private static Vector3 lastReportedPosition = Vector3.zero;
		private static bool firstCall = true;
		
		public static bool Prefix(ref Vector3 __result)
		{
			// 生产默认静默：仅在诊断模式下启用该探针
			if (!GoiHitboxLoggerAutonomous.diagnosticMode)
			{
				return true; // 放行原方法
			}
			// 检查是否在测试模式并获取测试位置
			if (ContinuousTrackerAutonomous.TryGetTestMousePosition(out Vector2 testPos))
			{
				// 在测试模式，返回测试位置，阻止原方法执行
				__result = new Vector3(testPos.x, testPos.y, 0);
				
				// 减少日志噪音 - 只记录较大的位置变化
				Vector3 delta = __result - lastReportedPosition;
				if (!firstCall && delta.magnitude > 5.0f) // 只记录明显的移动
				{
					Debug.Log($"🎮 Input.mousePosition被hook: 位置({testPos.x:F1}, {testPos.y:F1}) 增量({delta.x:F2}, {delta.y:F2}) 幅度:{delta.magnitude:F2}");
				}
				else if (firstCall)
				{
					Debug.Log($"🎮 Input.mousePosition被hook: 初始位置({testPos.x:F1}, {testPos.y:F1})");
					firstCall = false;
				}
				
				lastReportedPosition = __result;
				return false;
			}
			
			// 不在测试模式，允许原方法正常执行
			return true;
		}
	}
	
	/// <summary>
	/// 🔍 Hook Unity的内部鼠标增量计算 - 查看mouse_event是否1:1传递到这里
	/// </summary>
	[HarmonyPatch(typeof(Input), "GetAxis")]
	public static class InputAxisDiagnosticPatch
	{
		private static Dictionary<string, float> lastValues = new Dictionary<string, float>();
		private static Dictionary<string, float> callCounts = new Dictionary<string, float>();
		
		public static void Postfix(string axisName, float __result)
		{
			// 只监控鼠标轴
			if (axisName.Contains("Mouse"))
			{
				// 计数调用
				if (!callCounts.ContainsKey(axisName))
					callCounts[axisName] = 0;
				callCounts[axisName]++;
				
				// 检查值变化
				bool hasChanged = false;
				if (lastValues.ContainsKey(axisName))
				{
					hasChanged = Math.Abs(__result - lastValues[axisName]) > 0.001f;
				}
				else
				{
					hasChanged = Math.Abs(__result) > 0.001f;
				}
				
				if (hasChanged)
				{
					Debug.Log($"🔍 Unity原生 {axisName}: {__result:F6} (第{callCounts[axisName]}次调用) ← mouse_event传递到这里？");
					lastValues[axisName] = __result;
				}
			}
		}
	}
	
	[HarmonyPatch(typeof(Input), "GetAxis")]
	public static class InputGetAxisPatch
	{
		private static Vector2 lastMousePos = Vector2.zero;
		private static bool hasLastPos = false;
		private static Dictionary<string, float> lastLogTime = new Dictionary<string, float>();
		
		public static bool Prefix(string axisName, ref float __result)
		{
			// 🔍 诊断模式：让mouse_event的真实数据通过，不拦截
			if (GoiHitboxLoggerAutonomous.diagnosticMode)
			{
				return true; // 允许Unity原生处理
			}
			
			// 如果游戏请求鼠标轴，我们计算增量并返回
			if ((axisName == "Mouse X" || axisName == "Mouse Y") && 
				ContinuousTrackerAutonomous.TryGetTestMousePosition(out Vector2 testPos))
			{
				if (hasLastPos)
				{
					Vector2 delta = testPos - lastMousePos;
					float sensitivity = 100.0f; // 极大增加灵敏度（从20.0f改为100.0f）
					
					if (axisName == "Mouse X")
					{
						__result = delta.x / Screen.width * sensitivity;
					}
					else if (axisName == "Mouse Y")
					{
						__result = delta.y / Screen.height * sensitivity;
					}
					
					// 降噪：提高阈值并按轴名节流日志频率
					float absVal = Mathf.Abs(__result);
					if (absVal > 0.20f)
					{
						float now = Time.unscaledTime;
						if (!lastLogTime.ContainsKey(axisName) || now - lastLogTime[axisName] >= 0.5f)
						{
							Debug.Log($"🎮 GetAxis('{axisName}')={__result:F3} Δ={delta} {Screen.width}x{Screen.height}");
							lastLogTime[axisName] = now;
						}
					}
					
					lastMousePos = testPos;
					
					return false; // 阻止原方法执行
				}
				else
				{
					lastMousePos = testPos;
					hasLastPos = true;
					__result = 0f;
					return false;
				}
			}
			
			return true; // 允许其他轴正常工作
		}
		
		public static void Postfix(string axisName, float __result)
		{
			// 生产默认静默，仅在诊断模式下用于辅助观测
			if (!GoiHitboxLoggerAutonomous.diagnosticMode) return;
		}
	}

	/// <summary>
	/// Windows API Hook - 监控和拦截GetCursorPos调用
	/// </summary>
	[HarmonyPatch(typeof(GoiHitboxLoggerAutonomous), "GetCursorPosWrapper")]
	public static class GetCursorPosHook
	{
		public static bool Prefix(ref bool __result, out GoiHitboxLoggerAutonomous.POINT lpPoint)
		{
			// 生产默认静默：仅在诊断模式下启用该探针
			if (!GoiHitboxLoggerAutonomous.diagnosticMode)
			{
				lpPoint = new GoiHitboxLoggerAutonomous.POINT();
				return true; // 放行原方法
			}
			// 检查是否在测试模式
			if (ContinuousTrackerAutonomous.TryGetTestMousePosition(out Vector2 testPos))
			{
				// 将Unity坐标转换为Windows POINT
				lpPoint = new GoiHitboxLoggerAutonomous.POINT
				{
					X = (int)testPos.x,
					Y = Screen.height - (int)testPos.y // Y轴翻转
				};
				
				__result = true;
				return false; // 阻止原方法执行
			}
			
			// 不在测试模式，调用原始API
			lpPoint = new GoiHitboxLoggerAutonomous.POINT();
			bool realResult = GoiHitboxLoggerAutonomous.GetCursorPos(out lpPoint);
			
			__result = realResult;
			return false; // 我们已经处理了，不需要再调用
		}
	}

}
