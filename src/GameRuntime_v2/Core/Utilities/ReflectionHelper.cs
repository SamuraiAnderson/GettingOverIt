using System;
using System.Reflection;
using UnityEngine;

namespace GoiRuntime.Core.Utilities
{
	/// <summary>
	/// 反射辅助工具
	/// 提供常用的反射操作方法
	/// </summary>
	public static class ReflectionHelper
	{
		/// <summary>
		/// 获取私有字段
		/// </summary>
		public static FieldInfo GetPrivateField(Type type, string fieldName)
		{
			return type.GetField(fieldName, BindingFlags.NonPublic | BindingFlags.Instance);
		}

		/// <summary>
		/// 获取公共字段
		/// </summary>
		public static FieldInfo GetPublicField(Type type, string fieldName)
		{
			return type.GetField(fieldName, BindingFlags.Public | BindingFlags.Instance);
		}

		/// <summary>
		/// 获取任意字段（私有或公共）
		/// </summary>
		public static FieldInfo GetField(Type type, string fieldName)
		{
			return type.GetField(fieldName, 
				BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
		}

		/// <summary>
		/// 获取公共方法
		/// </summary>
		public static MethodInfo GetPublicMethod(Type type, string methodName)
		{
			return type.GetMethod(methodName, BindingFlags.Public | BindingFlags.Instance);
		}

		/// <summary>
		/// 获取私有方法
		/// </summary>
		public static MethodInfo GetPrivateMethod(Type type, string methodName)
		{
			return type.GetMethod(methodName, BindingFlags.NonPublic | BindingFlags.Instance);
		}

		/// <summary>
		/// 获取字段值
		/// </summary>
		public static T GetFieldValue<T>(object obj, string fieldName)
		{
			if (obj == null)
			{
				Debug.LogError("反射目标对象为 null");
				return default(T);
			}

			FieldInfo field = GetField(obj.GetType(), fieldName);
			if (field == null)
			{
				Debug.LogError($"未找到字段: {fieldName} in {obj.GetType().Name}");
				return default(T);
			}

			try
			{
				return (T)field.GetValue(obj);
			}
			catch (Exception e)
			{
				Debug.LogError($"获取字段值失败 [{fieldName}]: {e.Message}");
				return default(T);
			}
		}

		/// <summary>
		/// 设置字段值
		/// </summary>
		public static void SetFieldValue(object obj, string fieldName, object value)
		{
			if (obj == null)
			{
				Debug.LogError("反射目标对象为 null");
				return;
			}

			FieldInfo field = GetField(obj.GetType(), fieldName);
			if (field == null)
			{
				Debug.LogError($"未找到字段: {fieldName} in {obj.GetType().Name}");
				return;
			}

			try
			{
				field.SetValue(obj, value);
			}
			catch (Exception e)
			{
				Debug.LogError($"设置字段值失败 [{fieldName}]: {e.Message}");
			}
		}

		/// <summary>
		/// 调用方法
		/// </summary>
		public static object InvokeMethod(object obj, string methodName, params object[] parameters)
		{
			if (obj == null)
			{
				Debug.LogError("反射目标对象为 null");
				return null;
			}

			MethodInfo method = GetPublicMethod(obj.GetType(), methodName);
			if (method == null)
			{
				Debug.LogError($"未找到方法: {methodName} in {obj.GetType().Name}");
				return null;
			}

			try
			{
				return method.Invoke(obj, parameters);
			}
			catch (Exception e)
			{
				Debug.LogError($"调用方法失败 [{methodName}]: {e.Message}");
				return null;
			}
		}

		/// <summary>
		/// 检查字段是否存在
		/// </summary>
		public static bool HasField(Type type, string fieldName)
		{
			return GetField(type, fieldName) != null;
		}

		/// <summary>
		/// 检查方法是否存在
		/// </summary>
		public static bool HasMethod(Type type, string methodName)
		{
			return GetPublicMethod(type, methodName) != null;
		}

		/// <summary>
		/// 获取组件的字段
		/// </summary>
		public static FieldInfo GetComponentField(Component component, string fieldName)
		{
			if (component == null)
			{
				Debug.LogError("组件为 null");
				return null;
			}

			return GetField(component.GetType(), fieldName);
		}

		/// <summary>
		/// 设置组件的字段值
		/// </summary>
		public static void SetComponentFieldValue(Component component, string fieldName, object value)
		{
			if (component == null)
			{
				Debug.LogError("组件为 null");
				return;
			}

			SetFieldValue(component, fieldName, value);
		}

		/// <summary>
		/// 获取组件的字段值
		/// </summary>
		public static T GetComponentFieldValue<T>(Component component, string fieldName)
		{
			if (component == null)
			{
				Debug.LogError("组件为 null");
				return default(T);
			}

			return GetFieldValue<T>(component, fieldName);
		}
	}
}

