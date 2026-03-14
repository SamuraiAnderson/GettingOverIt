using HarmonyLib;
using UnityEngine;
using System.Reflection;
using System.Collections.Generic;

namespace GoiRuntime.PlayerControl
{
	/// <summary>
	/// 在 Rewired Player 层拦截鼠标输入（Plan B）：
	///   - 诊断：RL 模式激活前，记录所有 actionId 的非零 GetAxis 值，用于识别鼠标 actionId
	///   - RL 模式：拦截目标 actionId，返回注入值，屏蔽真实鼠标
	/// 使用运行时反射 patch，无需编译期 Rewired 引用。
	/// </summary>
	public static class RewiredMouseOverride
	{
		public const int MAX_AGENTS = 8;

		public static readonly float[] InjectedX = new float[MAX_AGENTS];
		public static readonly float[] InjectedY = new float[MAX_AGENTS];

		/// <summary>当前正在执行 FixedUpdate 的 agent 索引，-1 表示非 FixedUpdate 上下文。</summary>
		public static volatile int CurrentAgentIndex = -1;

		public static volatile bool  Active    = false;

		/// <summary>手动调用 InvokeFixedUpdate() 期间设为 true，供 Postfix 标注日志。</summary>
		public static volatile bool InsideInvokeFixedUpdate = false;

		/// <summary>鼠标 X/Y 对应的 Rewired actionId（int 版诊断后设置）。</summary>
		public static int MouseXActionId = -1;
		public static int MouseYActionId = -1;
		/// <summary>鼠标 X/Y 对应的 Rewired action 名称（string 版诊断后设置）。</summary>
		public static string MouseXActionName = null;
		public static string MouseYActionName = null;

		public static void SetForAgent(int idx, float x, float y)
		{
			if (idx >= 0 && idx < MAX_AGENTS)
			{
				InjectedX[idx] = x;
				InjectedY[idx] = y;
			}
		}

		public static void Set(float x, float y) { SetForAgent(0, x, y); }

		public static void Reset()
		{
			for (int i = 0; i < MAX_AGENTS; i++)
			{
				InjectedX[i] = 0f;
				InjectedY[i] = 0f;
			}
			CurrentAgentIndex = -1;
		}

		/// <summary>在 GameRuntimeManager.Awake() 中调用一次，运行时注册 Rewired 相关补丁。</summary>
		public static void ApplyPatches()
		{
			// 诊断已完成：直接硬编码已知的 action 名称
			MouseXActionName = "mouseX";
			MouseYActionName = "mouseY";
			Debug.Log("[RewiredMouseOverride] 已设置 actionName: mouseX / mouseY");

			var harmony = new Harmony("com.symbol.goi.rewired.player");

			string[] candidateTypes = new[]
			{
				"Rewired.Components.PlayerController",
				"Rewired.Player",
			};

			foreach (string typeName in candidateTypes)
			{
				System.Type t = null;
				foreach (var asm in System.AppDomain.CurrentDomain.GetAssemblies())
				{
					t = asm.GetType(typeName);
					if (t != null) break;
				}
				if (t == null) { Debug.LogWarning("[RewiredMouseOverride] 未找到类型: " + typeName); continue; }

				PatchGetAxis(harmony, t, typeName);
			}
		}

		private static void PatchGetAxis(Harmony harmony, System.Type t, string typeName)
		{
			var preS  = new HarmonyMethod(typeof(RewiredGetAxisStrPatch), nameof(RewiredGetAxisStrPatch.Prefix));
			var poS   = new HarmonyMethod(typeof(RewiredGetAxisStrPatch), nameof(RewiredGetAxisStrPatch.Postfix));
			var preRS = new HarmonyMethod(typeof(RewiredGetAxisRawStrPatch), nameof(RewiredGetAxisRawStrPatch.Prefix));

			// 只 patch 字符串版（游戏使用 GetAxis("mouseX") / GetAxis("mouseY")）
			// int 版不 patch（避免 SRE=False 环境下 IL 异常）
			TryPatch(harmony, t, typeName, "GetAxis",    new[] { typeof(string) }, preS,  poS);
			TryPatch(harmony, t, typeName, "GetAxisRaw", new[] { typeof(string) }, preRS, null);
		}

		private static void TryPatch(Harmony harmony, System.Type t, string typeName,
			string methodName, System.Type[] argTypes, HarmonyMethod prefix, HarmonyMethod postfix)
		{
			try
			{
				var mi = t.GetMethod(methodName, BindingFlags.Public | BindingFlags.Instance, null, argTypes, null);
				if (mi == null)
				{
					Debug.Log("[RewiredMouseOverride] 方法不存在: " + typeName + "." + methodName + "(" + argTypes[0].Name + ")");
					return;
				}
				harmony.Patch(mi, prefix: prefix, postfix: postfix);
				Debug.Log("[RewiredMouseOverride] " + typeName + "." + methodName + "(" + argTypes[0].Name + ") 已 patch");
			}
			catch (System.Exception ex)
			{
				Debug.LogWarning("[RewiredMouseOverride] patch 失败 " + typeName + "." + methodName + ": " + ex.Message);
			}
		}
	}

	// ── Patch 实现 ─────────────────────────────────────────────

	/// <summary>
	/// GetAxis(string) patch：RL 模式下拦截 "mouseX"/"mouseY" 并返回注入值。
	/// </summary>
	public static class RewiredGetAxisStrPatch
	{
		private static readonly Dictionary<string, float> _last = new Dictionary<string, float>();
		private static int _logCount = 0;

		public static bool Prefix(string __0, ref float __result)
		{
			if (!RewiredMouseOverride.Active) return true;
			int idx = RewiredMouseOverride.CurrentAgentIndex;
			if (idx < 0)
			{
				// Unity 自动调用 FixedUpdate 时 CurrentAgentIndex=-1，
				// 返回 0 使其成为无操作，只有手动 InvokeFixedUpdate 才提供真实输入。
				if (__0 == RewiredMouseOverride.MouseXActionName || __0 == RewiredMouseOverride.MouseYActionName)
				{
					__result = 0f;
					return false;
				}
				return true;
			}
			if (__0 == RewiredMouseOverride.MouseXActionName) { __result = RewiredMouseOverride.InjectedX[idx]; return false; }
			if (__0 == RewiredMouseOverride.MouseYActionName) { __result = RewiredMouseOverride.InjectedY[idx]; return false; }
			return true;
		}

		public static void Postfix(string __0, float __result)
		{
			if (_logCount >= 300) return;
			float prev = 0f;
			_last.TryGetValue(__0, out prev);
			if (__result != 0f || (prev != 0f && __result == 0f))
			{
				_logCount++;
				_last[__0] = __result;
				int idx = RewiredMouseOverride.CurrentAgentIndex;
				Debug.Log(string.Format("[Rewired.GetAxis(str)] name={0}  value={1:F4}  agentIdx={2}  inFixedUpdate={3}",
					__0, __result, idx, RewiredMouseOverride.InsideInvokeFixedUpdate));
			}
		}
	}

	/// <summary>GetAxisRaw(string) patch。</summary>
	public static class RewiredGetAxisRawStrPatch
	{
		public static bool Prefix(string __0, ref float __result)
		{
			if (!RewiredMouseOverride.Active) return true;
			int idx = RewiredMouseOverride.CurrentAgentIndex;
			if (idx < 0)
			{
				if (__0 == RewiredMouseOverride.MouseXActionName || __0 == RewiredMouseOverride.MouseYActionName)
				{
					__result = 0f;
					return false;
				}
				return true;
			}
			if (__0 == RewiredMouseOverride.MouseXActionName) { __result = RewiredMouseOverride.InjectedX[idx]; return false; }
			if (__0 == RewiredMouseOverride.MouseYActionName) { __result = RewiredMouseOverride.InjectedY[idx]; return false; }
			return true;
		}
	}
}
