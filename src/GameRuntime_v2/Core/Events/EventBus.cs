using System;
using System.Collections.Generic;
using UnityEngine;

namespace GoiRuntime.Core.Events
{
	/// <summary>
	/// 轻量级事件总线
	/// 用于模块间解耦通信
	/// </summary>
	public static class EventBus
	{
		private static Dictionary<string, List<Action<object>>> eventHandlers = 
			new Dictionary<string, List<Action<object>>>();

		/// <summary>
		/// 订阅事件
		/// </summary>
		public static void Subscribe(string eventName, Action<object> handler)
		{
			if (!eventHandlers.ContainsKey(eventName))
			{
				eventHandlers[eventName] = new List<Action<object>>();
			}
			eventHandlers[eventName].Add(handler);
		}

		/// <summary>
		/// 取消订阅
		/// </summary>
		public static void Unsubscribe(string eventName, Action<object> handler)
		{
			if (eventHandlers.ContainsKey(eventName))
			{
				eventHandlers[eventName].Remove(handler);
			}
		}

		/// <summary>
		/// 发布事件
		/// </summary>
		public static void Publish(string eventName, object data = null)
		{
			if (eventHandlers.ContainsKey(eventName))
			{
				foreach (var handler in eventHandlers[eventName])
				{
					try
					{
						handler?.Invoke(data);
					}
					catch (Exception e)
					{
						Debug.LogError($"事件处理器异常 [{eventName}]: {e.Message}");
					}
				}
			}
		}

		/// <summary>
		/// 清除所有订阅
		/// </summary>
		public static void Clear()
		{
			eventHandlers.Clear();
		}

		/// <summary>
		/// 清除指定事件的所有订阅
		/// </summary>
		public static void Clear(string eventName)
		{
			if (eventHandlers.ContainsKey(eventName))
			{
				eventHandlers[eventName].Clear();
			}
		}

		/// <summary>
		/// 获取订阅数量
		/// </summary>
		public static int GetSubscriberCount(string eventName)
		{
			if (eventHandlers.ContainsKey(eventName))
			{
				return eventHandlers[eventName].Count;
			}
			return 0;
		}
	}
}

