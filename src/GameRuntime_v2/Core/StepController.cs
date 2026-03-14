using System.Collections.Generic;
using UnityEngine;
using GoiRuntime.PlayerControl;
using GoiRuntime.Core.Interfaces;

namespace GoiRuntime.Core
{
	/// <summary>
	/// RL 步进控制器（帧级别）
	///
	/// 职责：
	///   1. 启动时将 Physics2D.simulationMode 切换为 Script 模式（持久，不再还原），
	///      使游戏物理引擎完全停止自动推进
	///   2. ExecuteStep(actions) → 向各 agent 注入动作 → Physics2D.Simulate() × stepFrames
	///      → 采集各 agent 状态 → 返回
	///   3. Reset() → 将所有 agent 恢复到初始位置、速度、角度
	///
	/// 并行策略（Option A：单进程多复制体）：
	///   - agent 0 = 原始 Player（由外部传入的 playerStateService / playerInputService 管理）
	///   - agent 1..N-1 = PlayerDuplicateManager 管理的复制体，各有独立的 Input/State 服务
	/// </summary>
	public class StepController
	{
		private readonly int numAgents;
		private readonly int stepFrames;
		private readonly int stateDim;
		private readonly int actionDim;

		// 各 agent 的 Input 和 State 服务（索引 0 = 原始 Player）
		private readonly List<PlayerInputService> inputServices = new List<PlayerInputService>();
		private readonly List<PlayerStateService> stateServices = new List<PlayerStateService>();

		// 原始 Player 各刚体的快照（用于 Reset）
		private struct RigidbodySnapshot
		{
			public Rigidbody2D rb;
			public Vector2 position;
			public float rotation;
			public Vector2 velocity;
			public float angularVelocity;
		}
		private readonly List<List<RigidbodySnapshot>> initialSnapshots = new List<List<RigidbodySnapshot>>();

	private bool isInitialized;

	public bool IsReady => isInitialized;
	public int NumAgents => numAgents;

		public StepController(int numAgents, int stepFrames, int stateDim = 29, int actionDim = 2)
		{
			this.numAgents  = numAgents;
			this.stepFrames = stepFrames > 0 ? stepFrames : 1;
			this.stateDim   = stateDim > 0 ? stateDim : 29;
			this.actionDim  = actionDim > 0 ? actionDim : 2;
		}

		// ── 初始化 ──────────────────────────────────────────────

		/// <summary>
		/// 初始化：绑定 agent 0（原始 Player），并为每个复制体创建独立服务
		/// </summary>
		public bool Initialize(
			PlayerStateService originStateService,
			PlayerInputService originInputService,
			PlayerDuplicateManager duplicateManager)
		{
			inputServices.Clear();
			stateServices.Clear();
			initialSnapshots.Clear();

			// --- Agent 0：原始 Player ---
			inputServices.Add(originInputService);
			stateServices.Add(originStateService);

			// --- Agent 1..N-1：复制体 ---
			for (int i = 1; i < numAgents; i++)
			{
				GameObject dup = duplicateManager != null ? duplicateManager.GetDuplicate(i - 1) : null;
				if (dup == null)
				{
					Debug.LogWarning($"[StepController] 复制体 #{i - 1} 不存在，agent {i} 将使用 null 服务");
					inputServices.Add(null);
					stateServices.Add(null);
					continue;
				}

				var inputSvc = new PlayerInputService();
				if (!inputSvc.InitializeFor(dup))
					Debug.LogWarning($"[StepController] 复制体 #{i - 1} 输入服务初始化失败");

				var stateSvc = new PlayerStateService();
				if (!stateSvc.InitializeFor(dup))
					Debug.LogWarning($"[StepController] 复制体 #{i - 1} 状态服务初始化失败");

				inputServices.Add(inputSvc);
				stateServices.Add(stateSvc);
			}

			// --- 切换物理引擎到 Script 模式（持久）---
			Physics2D.simulationMode = SimulationMode2D.Script;
			Debug.Log($"[StepController] Physics2D.simulationMode = Script（持久）");
			Debug.Log($"[StepController] fixedDeltaTime = {Time.fixedDeltaTime:F4}s，每 step 推进 {stepFrames} 帧");

			// --- 保存各 agent 初始快照 ---
			CaptureAllSnapshots();

			isInitialized = true;
			Debug.Log($"[StepController] 初始化完成，管理 {numAgents} 个 agent");
			return true;
		}

		// ── 核心 API ──────────────────────────────────────────────

		/// <summary>
		/// 执行一个 RL step：
		///   1. 将 actions 分发给各 agent（SetMouseInput）
		///   2. 调用 Physics2D.Simulate(dt) × stepFrames
		///   3. 采集各 agent 的状态，打包为 flat float[]
		/// actions 布局：[a0_x, a0_y, a1_x, a1_y, ...]，长度 = numAgents × 2
		/// 返回 states 布局：[s0_0..s0_28, s1_0..s1_28, ...]，长度 = numAgents × 29
		/// </summary>
		public float[] ExecuteStep(float[] actions)
		{
			// 1. 分发动作 + 触发 FixedUpdate（必须在 Simulate 前调用，使关节电机力生效）
			for (int i = 0; i < numAgents; i++)
			{
				if (inputServices[i] == null || !inputServices[i].IsReady) continue;
				float ax = (i * actionDim + 0 < actions.Length) ? actions[i * actionDim + 0] : 0f;
				float ay = (i * actionDim + 1 < actions.Length) ? actions[i * actionDim + 1] : 0f;
				inputServices[i].SetMouseInput(new Vector2(ax, ay));
				inputServices[i].InvokeFixedUpdate();
			}

			// 2. 推进物理
			float dt = Time.fixedDeltaTime;
			for (int f = 0; f < stepFrames; f++)
			{
				Physics2D.Simulate(dt);
			}

			// 3. 采集状态
			return CollectAllStates();
		}

		/// <summary>
		/// 重置所有 agent 到初始快照状态
		/// </summary>
		public void Reset()
		{
			foreach (var snapList in initialSnapshots)
			{
				foreach (var snap in snapList)
				{
					if (snap.rb == null) continue;
					snap.rb.position        = snap.position;
					snap.rb.rotation        = snap.rotation;
					snap.rb.velocity        = snap.velocity;
					snap.rb.angularVelocity = snap.angularVelocity;
				}
			}
	// 清零注入值，避免上一 episode 的残余影响下一 episode
	GoiRuntime.PlayerControl.RewiredMouseOverride.Reset();

	// 关键修复：Reset 后的单步 Simulate 必须先调用 InvokeFixedUpdate，
	// 以确保各 agent 的关节电机力在 Simulate 前被施加。
	// 若跳过此步，关节约束求解器会因缺少电机对抗力而产生巨大修正冲量，
	// 导致克隆体刚体被弹射到 (0,0) 附近，物理状态不可恢复。
	for (int i = 0; i < numAgents; i++)
	{
		if (inputServices[i] == null || !inputServices[i].IsReady) continue;
		inputServices[i].SetMouseInput(Vector2.zero);  // 零输入，只激活电机阻尼/稳定力
		inputServices[i].InvokeFixedUpdate();
	}

	// 推进一帧让物理状态稳定
	Physics2D.Simulate(Time.fixedDeltaTime);
	Debug.Log("[StepController] 所有 agent 已重置");
		}

		/// <summary>
		/// 采集各 agent 当前状态，返回 flat float[]
		/// </summary>
		public float[] CollectAllStates()
		{
			float[] result = new float[numAgents * stateDim];
			for (int i = 0; i < numAgents; i++)
			{
				float[] s = (stateServices[i] != null && stateServices[i].IsReady)
					? stateServices[i].GetStateArray()
					: new float[stateDim];
				if (s == null || s.Length < stateDim)
				{
					Debug.LogError($"[StepController] agent {i} GetStateArray 返回 {(s == null ? "null" : s.Length.ToString())} 元素，期望 {stateDim}，用零补全");
					s = new float[stateDim];
				}
				System.Array.Copy(s, 0, result, i * stateDim, stateDim);
			}
			return result;
		}

	// ── 快照管理 ──────────────────────────────────────────────

	/// <summary>
	/// 以当前物理状态为基准重新拍快照（供 warmup 后调用）
	/// </summary>
	public void TakeNewSnapshot()
	{
		CaptureAllSnapshots();
		Debug.Log("[StepController] 已重新拍摄初始快照（warmup 完成后的稳定状态）");
	}

	private void CaptureAllSnapshots()
		{
			initialSnapshots.Clear();
			for (int agentIdx = 0; agentIdx < stateServices.Count; agentIdx++)
			{
				var stateSvc = stateServices[agentIdx];
				var list = new List<RigidbodySnapshot>();
				if (stateSvc != null && stateSvc.IsReady)
				{
					GameObject go = stateSvc.GetPlayerObject();
					if (go != null)
					{
						var rbs = go.GetComponentsInChildren<Rigidbody2D>(true);
						foreach (var rb in rbs)
						{
							list.Add(new RigidbodySnapshot
							{
								rb              = rb,
								position        = rb.position,
								rotation        = rb.rotation,
								velocity        = rb.velocity,
								angularVelocity = rb.angularVelocity,
							});
						}
						// 打印 rb[0] 坐标，诊断快照是否捕获了正确位置
						if (rbs.Length > 0)
							Debug.Log($"[StepController] agent{agentIdx} 快照 rb[0].pos={rbs[0].position:F2} rb[0].vel={rbs[0].velocity:F1}");
					}
				}
				else
				{
					Debug.LogWarning($"[StepController] agent{agentIdx} 状态服务未就绪，快照为空");
				}
				initialSnapshots.Add(list);
			}
			Debug.Log($"[StepController] 已保存 {initialSnapshots.Count} 个 agent 的初始快照");
		}
	}
}
