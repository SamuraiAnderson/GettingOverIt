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
		
		private bool isInitialized = false;
		
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
		/// 设置鼠标输入
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
				// 限制输入范围
				Vector2 clampedInput = ClampInput(input);
				mouseInputField.SetValue(playerControlComponent, clampedInput);
			}
			catch (Exception e)
			{
				Debug.LogError($"设置 mouseInput 失败: {e.Message}");
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
				Debug.Log($"输入已{(enabled ? "启用" : "禁用")}");
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
		
		#endregion
	}
}

