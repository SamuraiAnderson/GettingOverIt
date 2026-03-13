using UnityEngine;

namespace GoiRuntime.Core.Utilities
{
	/// <summary>
	/// 输入验证工具
	/// 基于输入特性验证，有效范围 [10, 100]
	/// </summary>
	public static class InputValidator
	{
		/// <summary>
		/// 最小有效输入值
		/// </summary>
		public const float MIN_INPUT = 10f;

		/// <summary>
		/// 最大有效输入值
		/// </summary>
		public const float MAX_INPUT = 100f;

		/// <summary>
		/// 验证输入是否在有效范围内
		/// </summary>
		public static bool IsValid(Vector2 input)
		{
			return IsValid(input.x) && IsValid(input.y);
		}

		/// <summary>
		/// 验证单个值是否在有效范围内
		/// </summary>
		public static bool IsValid(float value)
		{
			// 允许 0 值（表示无输入）
			if (Mathf.Approximately(value, 0f))
			{
				return true;
			}

			// 检查绝对值是否在 [MIN_INPUT, MAX_INPUT] 范围内
			float absValue = Mathf.Abs(value);
			return absValue >= MIN_INPUT && absValue <= MAX_INPUT;
		}

		/// <summary>
		/// 限制输入到有效范围
		/// </summary>
		public static Vector2 Clamp(Vector2 input)
		{
			return new Vector2(Clamp(input.x), Clamp(input.y));
		}

		/// <summary>
		/// 限制单个值到有效范围
		/// </summary>
		public static float Clamp(float value)
		{
			// 保持符号
			if (value == 0f)
			{
				return 0f;
			}

			float sign = Mathf.Sign(value);
			float absValue = Mathf.Abs(value);

			// 限制到 [MIN_INPUT, MAX_INPUT]
			absValue = Mathf.Clamp(absValue, MIN_INPUT, MAX_INPUT);

			return sign * absValue;
		}

		/// <summary>
		/// 规范化输入（确保在有效范围内，如果无效则返回 0）
		/// </summary>
		public static Vector2 Normalize(Vector2 input)
		{
			return new Vector2(Normalize(input.x), Normalize(input.y));
		}

		/// <summary>
		/// 规范化单个值
		/// </summary>
		public static float Normalize(float value)
		{
			if (IsValid(value))
			{
				return value;
			}

			// 如果无效，返回 0
			return 0f;
		}

		/// <summary>
		/// 验证并获取有效输入
		/// </summary>
		/// <param name="input">原始输入</param>
		/// <param name="validatedInput">验证后的输入</param>
		/// <returns>是否需要修正</returns>
		public static bool ValidateAndClamp(Vector2 input, out Vector2 validatedInput)
		{
			validatedInput = Clamp(input);
			return validatedInput.x != input.x || validatedInput.y != input.y;
		}

		/// <summary>
		/// 获取输入的说明（用于调试）
		/// </summary>
		public static string GetInputDescription(Vector2 input)
		{
			bool xValid = IsValid(input.x);
			bool yValid = IsValid(input.y);

			if (xValid && yValid)
			{
				return "有效输入";
			}

			string desc = "无效输入: ";
			if (!xValid) desc += $"X={input.x:F1} ";
			if (!yValid) desc += $"Y={input.y:F1}";

			return desc;
		}
	}
}

