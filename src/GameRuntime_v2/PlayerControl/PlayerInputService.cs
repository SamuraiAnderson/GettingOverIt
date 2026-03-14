using UnityEngine;
using System;
using System.Reflection;
using GoiRuntime.Core.Interfaces;
using GoiRuntime.Core.Utilities;

namespace GoiRuntime.PlayerControl
{
	/// <summary>
	/// Player 输入控制服务
	/// 通过反射注入 mouseInput 字段来控制 Player
	/// </summary>
	public class PlayerInputService : IPlayerInputService
	{
		#region 组件引用
		
	private GameObject playerObject;
	private Component playerControlComponent;
	private FieldInfo mouseInputField;
	private FieldInfo inputEnabledField;
	private MethodInfo fixedUpdateMethod;
	
	private bool isInitialized = false;

	/// <summary>此服务绑定的 agent 索引，由 StepController 在 Initialize 时设置。</summary>
	public int AgentIndex = 0;
		
		#endregion
		
		#region 配置
		
		/// <summary>
		/// 输入值范围限制
		/// </summary>
		public const float MIN_INPUT = 10f;
		public const float MAX_INPUT = 100f;
		
		#endregion
		
		#region IPlayerInputService 实现
		
		/// <summary>
		/// 初始化服务（自动查找场景中名为 "Player" 的对象）
		/// </summary>
		public bool Initialize()
		{
			if (isInitialized)
			{
				Debug.Log("PlayerInputService 已初始化");
				return true;
			}
			
			return InitializePlayerControl(null);
		}

		/// <summary>
		/// 初始化服务并绑定到指定 GameObject（用于复制体）
		/// </summary>
		public bool InitializeFor(GameObject targetPlayer)
		{
			isInitialized = false;
			return InitializePlayerControl(targetPlayer);
		}
		
		/// <summary>
		/// 服务是否就绪
		/// </summary>
		public bool IsReady => isInitialized;
		
		/// <summary>
		/// 设置鼠标输入。
		/// 双路注入：① 反射写入 mouseInput 字段（兼容旧路径）；
		///           ② 更新 MouseInputOverride 全局覆盖层（拦截 Input.GetAxis）。
		/// </summary>
		public void SetMouseInput(Vector2 input)
		{
			if (!isInitialized)
			{
				Debug.LogWarning("PlayerInputService 未初始化");
				return;
			}
			
			try
			{
			Vector2 clampedInput = ClampInput(input);
			RewiredMouseOverride.SetForAgent(AgentIndex, clampedInput.x, clampedInput.y);
			mouseInputField.SetValue(playerControlComponent, clampedInput);
			}
			catch (Exception e)
			{
				Debug.LogError($"设置 mouseInput 失败: {e.Message}");
			}
		}

		/// <summary>
		/// 手动调用 PlayerControl.FixedUpdate()，让游戏逻辑读取 mouseInput 并对刚体施力。
		/// 须在 SetMouseInput() 之后、Physics2D.Simulate() 之前调用。
		/// </summary>
		public void InvokeFixedUpdate()
		{
			if (!isInitialized || fixedUpdateMethod == null) return;
			try
			{
				RewiredMouseOverride.CurrentAgentIndex = AgentIndex;
				RewiredMouseOverride.InsideInvokeFixedUpdate = true;
				fixedUpdateMethod.Invoke(playerControlComponent, null);
			}
			catch (Exception e)
			{
				Debug.LogError($"InvokeFixedUpdate 失败: {e.Message}");
			}
			finally
			{
				RewiredMouseOverride.CurrentAgentIndex = -1;
				RewiredMouseOverride.InsideInvokeFixedUpdate = false;
			}
		}
		
		/// <summary>
		/// 获取当前鼠标输入
		/// </summary>
		public Vector2 GetMouseInput()
		{
			if (!isInitialized || mouseInputField == null)
			{
				return Vector2.zero;
			}
			
			try
			{
				object value = mouseInputField.GetValue(playerControlComponent);
				if (value is Vector2 vec)
				{
					return vec;
				}
			}
			catch (Exception e)
			{
				Debug.LogError($"获取 mouseInput 失败: {e.Message}");
			}
			
			return Vector2.zero;
		}
		
		/// <summary>
		/// 设置输入启用状态
		/// </summary>
		public void SetInputEnabled(bool enabled)
		{
			if (!isInitialized || inputEnabledField == null)
			{
				Debug.LogWarning("无法设置 input_enabled");
				return;
			}
			
			try
			{
			inputEnabledField.SetValue(playerControlComponent, enabled);
			}
			catch (Exception e)
			{
				Debug.LogError($"设置 input_enabled 失败: {e.Message}");
			}
		}
		
		/// <summary>
		/// 获取输入启用状态
		/// </summary>
		public bool GetInputEnabled()
		{
			if (!isInitialized || inputEnabledField == null)
			{
				return false;
			}
			
			try
			{
				object value = inputEnabledField.GetValue(playerControlComponent);
				if (value is bool b)
				{
					return b;
				}
			}
			catch (Exception e)
			{
				Debug.LogError($"获取 input_enabled 失败: {e.Message}");
			}
			
			return false;
		}
		
		#endregion
		
		#region 初始化
		
		/// <summary>
		/// 初始化 Player 控制；targetPlayer 为 null 时自动 Find("Player")
		/// </summary>
		private bool InitializePlayerControl(GameObject targetPlayer)
		{
			try
			{
				playerObject = targetPlayer != null ? targetPlayer : GameObject.Find("Player");
				if (playerObject == null)
				{
					Debug.LogWarning("PlayerInputService: 未找到 Player 对象");
					return false;
				}

				Debug.Log($"PlayerInputService: 绑定到 {playerObject.name}");
				
				// 获取 PlayerControl 组件
				playerControlComponent = playerObject.GetComponent("PlayerControl");
				if (playerControlComponent == null)
				{
			Debug.LogError("未找到 PlayerControl 组件");
				return false;
			}

			Debug.Log($"找到 PlayerControl 组件");
				
				// 反射获取字段
				Type playerControlType = playerControlComponent.GetType();
				
				mouseInputField = playerControlType.GetField("mouseInput", 
					BindingFlags.NonPublic | BindingFlags.Instance);
				
		inputEnabledField = playerControlType.GetField("input_enabled", 
				BindingFlags.NonPublic | BindingFlags.Instance);
			
			if (mouseInputField == null)
				{
			Debug.LogError("未找到 mouseInput 字段");
				return false;
			}

		Debug.Log("成功获取 mouseInput 字段");

	if (inputEnabledField == null)
		Debug.LogWarning("未找到 input_enabled 字段（可能不需要）");

		// 缓存 PlayerControl.FixedUpdate()，用于手动驱动游戏逻辑
		fixedUpdateMethod = playerControlType.GetMethod("FixedUpdate",
			BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
		if (fixedUpdateMethod != null)
			Debug.Log("成功缓存 PlayerControl.FixedUpdate()");
		else
			Debug.LogWarning("未找到 PlayerControl.FixedUpdate()（输入将无法施力）");

		var harmony = new HarmonyLib.Harmony("com.symbol.goi.playercontrol.lifecycle");

		// Patch Update()：RL 模式下跳过真实鼠标读取
		var updateMethod = playerControlType.GetMethod("Update",
			BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
		if (updateMethod != null)
		{
			try
			{
				var prefix = new HarmonyLib.HarmonyMethod(
					typeof(PlayerControlUpdatePatch),
					nameof(PlayerControlUpdatePatch.Prefix));
				harmony.Patch(updateMethod, prefix: prefix);
				Debug.Log("PlayerControl.Update() 已被 Harmony patch（RL 模式下跳过）");
			}
			catch (Exception e)
			{
				Debug.LogWarning($"Harmony patch PlayerControl.Update 失败: {e.Message}");
			}
		}
		else
			Debug.LogWarning("未找到 PlayerControl.Update()");

		if (inputEnabledField != null)
		{
			inputEnabledField.SetValue(playerControlComponent, true);
			Debug.Log("PlayerInputService: input_enabled = true");
		}

		isInitialized = true;
			return true;
		}
		catch (Exception e)
		{
			Debug.LogError($"PlayerInputService 初始化失败: {e.Message}");
				Debug.LogError($"堆栈跟踪: {e.StackTrace}");
				return false;
			}
		}
		
		#endregion
		
		#region 工具方法
		
		/// <summary>
		/// 限制输入范围
		/// </summary>
		private Vector2 ClampInput(Vector2 input)
		{
			return new Vector2(
				Mathf.Clamp(input.x, -MAX_INPUT, MAX_INPUT),
				Mathf.Clamp(input.y, -MAX_INPUT, MAX_INPUT)
			);
		}
		
		/// <summary>
		/// 验证输入是否有效
		/// </summary>
		public bool IsValidInput(Vector2 input)
		{
			float magnitude = input.magnitude;
			return magnitude >= MIN_INPUT && magnitude <= MAX_INPUT;
		}
		
		/// <summary>
		/// 获取 Player 对象
		/// </summary>
		public GameObject GetPlayerObject()
		{
			return playerObject;
		}
		
		/// <summary>
		/// 通过反射获取 PlayerControl 内部的 fakeCursorRB (Rigidbody2D)。
		/// 用于 StepController 的快照/重置。
		/// </summary>
		public Rigidbody2D GetFakeCursorRB()
		{
			if (!isInitialized || playerControlComponent == null) return null;
			try
			{
				FieldInfo field = playerControlComponent.GetType().GetField("fakeCursorRB",
					BindingFlags.NonPublic | BindingFlags.Instance);
				return field?.GetValue(playerControlComponent) as Rigidbody2D;
			}
			catch { return null; }
		}
		
		#endregion
	}

	/// <summary>
	/// Harmony prefix patch：RL 模式激活时跳过 PlayerControl.Update()，
	/// 防止游戏从真实鼠标读取输入并覆盖我们注入的 mouseInput 值。
	/// </summary>
	public static class PlayerControlUpdatePatch
	{
		/// <summary>RL 模式激活后设为 true，阻止 Update() 读取真实鼠标。</summary>
		public static volatile bool RlModeActive = false;

		public static bool Prefix()
		{
			return !RlModeActive;
		}
	}

}

