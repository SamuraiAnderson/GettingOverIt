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
		/// <summary>PlayerState.ToFloatArray() 的基础维度（不含 fakeCursor）。</summary>
		public const int BASE_STATE_DIM = 29;
		/// <summary>追加的 fakeCursor 维度：绝对坐标 x,y + 速度 vx,vy。</summary>
		public const int CURSOR_DIM = 4;
		/// <summary>每个 agent 的完整状态维度（基础 + fakeCursor）。</summary>
		public const int STATE_DIM = BASE_STATE_DIM + CURSOR_DIM;  // 33

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
		private readonly List<RigidbodySnapshot> fakeCursorSnapshots = new List<RigidbodySnapshot>();

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
			PlayerDuplicateManager duplicateManager,
			bool normalizePose = false,
			int settleFrames = 120,
			float upScale = 1.0f,
			float offsetX = 0f,
			float gain = 15f)
		{
			inputServices.Clear();
			stateServices.Clear();
			initialSnapshots.Clear();

			// --- Agent 0：原始 Player ---
			originInputService.AgentIndex = 0;
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
				inputSvc.AgentIndex = i;
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

			// --- 初始姿态归一化（须在拍快照前）---
			// 修复初始锤子扭曲：RL 激活前真实鼠标已把 cursor 拖到随机位置，
			// 这里用闭环把 cursor 驱到 hub 上方的确定位置，得到一致且非扭曲的起始姿态。
			if (normalizePose)
				NormalizeInitialPose(settleFrames, upScale, offsetX, gain);

			// --- 保存各 agent 初始快照 ---
			CaptureAllSnapshots();

			isInitialized = true;

			// 诊断：打印各 agent 的 Rigidbody2D / fakeCursorRB InstanceID 确认物理对象独立性
			for (int a = 0; a < stateServices.Count; a++)
			{
				var ss = stateServices[a];
				if (ss != null && ss.IsReady)
				{
					var go = ss.GetPlayerObject();
					if (go != null)
					{
						var rb = go.GetComponent<Rigidbody2D>();
						var fcRB = inputServices[a]?.GetFakeCursorRB();
						Debug.Log($"[StepController] agent{a} PlayerObject={go.name} rb.ID={rb?.GetInstanceID()} fcRB.ID={fcRB?.GetInstanceID()} inputSvc.AgentIndex={inputServices[a]?.AgentIndex}");
					}
				}
			}

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
	private int _diagStepCount = 0;

		public float[] ExecuteStep(float[] actions)
		{
			_diagStepCount++;
			bool diag = _diagStepCount <= 5;

			// 1. 分发动作 + 触发 FixedUpdate（必须在 Simulate 前调用，使关节电机力生效）
			for (int i = 0; i < numAgents; i++)
			{
				if (inputServices[i] == null || !inputServices[i].IsReady) continue;
				float ax = (i * actionDim + 0 < actions.Length) ? actions[i * actionDim + 0] : 0f;
				float ay = (i * actionDim + 1 < actions.Length) ? actions[i * actionDim + 1] : 0f;
				inputServices[i].SetMouseInput(new Vector2(ax, ay));
				inputServices[i].InvokeFixedUpdate();

				if (diag)
				{
					var ss = stateServices[i];
					var rb = (ss != null && ss.IsReady) ? ss.GetPlayerObject()?.GetComponent<Rigidbody2D>() : null;
					Debug.Log(string.Format("[StepDiag] step={0} agent={1} action=({2:F1},{3:F1}) agentIdx={4} rb.vel=({5:F4},{6:F4}) rb.ID={7}",
						_diagStepCount, i, ax, ay, inputServices[i].AgentIndex,
						rb != null ? rb.velocity.x : -999f,
						rb != null ? rb.velocity.y : -999f,
						rb != null ? rb.GetInstanceID() : 0));
					Rigidbody2D fcRB = inputServices[i]?.GetFakeCursorRB();
					if (fcRB != null)
					{
						Debug.Log(string.Format("[StepDiag] step={0} agent={1} fcRB.pos=({2:F4},{3:F4}) fcRB.vel=({4:F4},{5:F4}) fcRB.ID={6}",
							_diagStepCount, i,
							fcRB.position.x, fcRB.position.y,
							fcRB.velocity.x, fcRB.velocity.y,
							fcRB.GetInstanceID()));
					}
				}
			}

			// 2. 推进物理
			float dt = Time.fixedDeltaTime;
			for (int f = 0; f < stepFrames; f++)
			{
				Physics2D.Simulate(dt);
			}

			// 3. 采集状态（诊断：打印 Simulate 后的速度）
			float[] result = CollectAllStates();

			if (diag)
			{
				for (int i = 0; i < numAgents; i++)
				{
					var ss = stateServices[i];
					var rb = (ss != null && ss.IsReady) ? ss.GetPlayerObject()?.GetComponent<Rigidbody2D>() : null;
					Debug.Log(string.Format("[StepDiag] step={0} agent={1} AFTER_SIM rb.vel=({2:F4},{3:F4})",
						_diagStepCount, i,
						rb != null ? rb.velocity.x : -999f,
						rb != null ? rb.velocity.y : -999f));
					Rigidbody2D fcRB = inputServices[i]?.GetFakeCursorRB();
					if (fcRB != null)
					{
						Debug.Log(string.Format("[StepDiag] step={0} agent={1} AFTER_SIM fcRB.pos=({2:F4},{3:F4})",
							_diagStepCount, i, fcRB.position.x, fcRB.position.y));
					}
				}
			}

			return result;
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
	// 恢复各 agent 的 fakeCursorRB 状态
	foreach (var snap in fakeCursorSnapshots)
	{
		if (snap.rb == null) continue;
		snap.rb.position        = snap.position;
		snap.rb.rotation        = snap.rotation;
		snap.rb.velocity        = snap.velocity;
		snap.rb.angularVelocity = snap.angularVelocity;
	}
	// 重置诊断计数器，使 Reset 后的前几步能被记录
	_diagStepCount = 0;
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
		/// 将指定 agent 整体传送到目标世界坐标。
		/// Player 是铰接体（body/hub/slider/handle/pole/tip 各有独立 Rigidbody2D），
		/// 传送时以 Player 根刚体当前位置为基准，将 delta 平移应用到所有子刚体，
		/// 然后清零速度并推进一帧物理使关节稳定。
		/// </summary>
		public float[] Teleport(int agentIndex, Vector2 target)
		{
			if (agentIndex < 0 || agentIndex >= stateServices.Count)
			{
				Debug.LogError($"[StepController] Teleport: agentIndex={agentIndex} 越界 (0..{stateServices.Count - 1})");
				return CollectAllStates();
			}

			var stateSvc = stateServices[agentIndex];
			if (stateSvc == null || !stateSvc.IsReady)
			{
				Debug.LogError($"[StepController] Teleport: agent{agentIndex} 状态服务未就绪");
				return CollectAllStates();
			}

			GameObject playerObj = stateSvc.GetPlayerObject();
			if (playerObj == null)
			{
				Debug.LogError($"[StepController] Teleport: agent{agentIndex} PlayerObject 为 null");
				return CollectAllStates();
			}

			var rootRb = playerObj.GetComponent<Rigidbody2D>();
			if (rootRb == null)
			{
				Debug.LogError($"[StepController] Teleport: agent{agentIndex} 无根 Rigidbody2D");
				return CollectAllStates();
			}

			Vector2 delta = target - rootRb.position;
			var allRbs = playerObj.GetComponentsInChildren<Rigidbody2D>(true);

			foreach (var rb in allRbs)
			{
				rb.position += delta;
				rb.velocity = Vector2.zero;
				rb.angularVelocity = 0f;
			}

			// fakeCursor 也需要平移
			if (agentIndex < inputServices.Count && inputServices[agentIndex] != null)
			{
				Rigidbody2D fcRB = inputServices[agentIndex].GetFakeCursorRB();
				if (fcRB != null)
				{
					fcRB.position += delta;
					fcRB.velocity = Vector2.zero;
					fcRB.angularVelocity = 0f;
				}
			}

			// 零输入 + InvokeFixedUpdate + Simulate 稳定关节
			if (agentIndex < inputServices.Count && inputServices[agentIndex] != null && inputServices[agentIndex].IsReady)
			{
				inputServices[agentIndex].SetMouseInput(Vector2.zero);
				inputServices[agentIndex].InvokeFixedUpdate();
			}
			Physics2D.Simulate(Time.fixedDeltaTime);

			Debug.Log($"[StepController] Teleport agent{agentIndex} → ({target.x:F2}, {target.y:F2}), delta=({delta.x:F2}, {delta.y:F2}), {allRbs.Length} rigidbodies moved");

			return CollectAllStates();
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
					: new float[BASE_STATE_DIM];
				if (s == null || s.Length < BASE_STATE_DIM)
				{
					Debug.LogError($"[StepController] agent {i} GetStateArray 返回 {(s == null ? "null" : s.Length.ToString())} 元素，期望 {BASE_STATE_DIM}，用零补全");
					s = new float[BASE_STATE_DIM];
				}

				int baseOffset = i * stateDim;
				System.Array.Copy(s, 0, result, baseOffset, BASE_STATE_DIM);

				// 追加 fakeCursor：绝对坐标 + 速度（Python 侧转为相对 player 坐标以保持平移等变）。
				// 回退：fakeCursorRB 尚未创建时用 player 坐标（相对=0）+ 零速度，避免出现巨大的相对位移。
				float cx = s[0], cy = s[1], cvx = 0f, cvy = 0f;
				Rigidbody2D fcRB = (i < inputServices.Count) ? inputServices[i]?.GetFakeCursorRB() : null;
				if (fcRB != null)
				{
					cx = fcRB.position.x;
					cy = fcRB.position.y;
					cvx = fcRB.velocity.x;
					cvy = fcRB.velocity.y;
				}
				result[baseOffset + BASE_STATE_DIM + 0] = cx;
				result[baseOffset + BASE_STATE_DIM + 1] = cy;
				result[baseOffset + BASE_STATE_DIM + 2] = cvx;
				result[baseOffset + BASE_STATE_DIM + 3] = cvy;
			}
			return result;
		}

	// ── 初始姿态归一化 ──────────────────────────────────────────

	// ── 游戏原生静止姿态（反编译 Saviour.ResetPlayerButNotDialogue 提取）──────
	// 各刚体相对 Player 根(body)的位置偏移。顺序 = GetComponentsInChildren<Rigidbody2D>：
	//   0 body / 1 Hub / 2 Slider / 3 Handle / 4 PoleMiddle / 5 Tip。
	// 原始绝对坐标（游戏 -r 复位硬编码）：
	//   body(-44.2926,-2.4218) hub(-44.2926,-1.8218) slider(-44.2926,-1.8218)
	//   handle(-44.3906,-1.7535) pole(-43.4309,-2.4262) tip(-42.5466,-3.0462)
	private static readonly Vector2[] NATIVE_REL_POS = new Vector2[]
	{
		new Vector2( 0.000000f,  0.000000f),  // body
		new Vector2( 0.000000f,  0.600000f),  // Hub
		new Vector2( 0.000000f,  0.600000f),  // Slider
		new Vector2(-0.098025f,  0.668360f),  // Handle
		new Vector2( 0.861649f, -0.004411f),  // PoleMiddle
		new Vector2( 1.745987f, -0.624371f),  // Tip
	};
	// 各刚体角度（度），同样取自 -r 复位硬编码（rbAngles 静态数组）。
	private static readonly float[] NATIVE_ANGLE =
		{ -0.089379f, 0f, -34.89184f, 145.1082f, 144.9679f, 144.9679f };

	/// <summary>
	/// 把每个 agent 的锤子归一化到游戏原生的自然静止姿态。
	///
	/// 直接照搬游戏 `Saviour.ResetPlayerButNotDialogue`（按 -r 复位）硬编码的整套刚体姿态：
	/// 逐刚体写入相对 body 的固定位置 + 固定角度 + 零速度，再把 fakeCursor 吸附到锤头
	/// （复刻 `Saviour.Load` 结尾的 `cursor.position = hammer.position`，零牵引力），
	/// 最后零输入静置若干帧让关节收敛到精确平衡。
	///
	/// 相比旧实现（仅吸附光标到当前锤头再靠重力回位）：本法确定性地重建整条锤臂，
	/// 无论 RL 激活前光标被真实鼠标拖到哪里、锤臂是否交叉扭曲，都能得到唯一正确姿态。
	/// 须在物理已切 Script 模式、且 RewiredMouseOverride.Active=true 之后调用。
	/// （参数 upScale/offsetX/gain 已废弃，保留仅为兼容调用方。）
	/// </summary>
	private void NormalizeInitialPose(int settleFrames, float upScale, float offsetX, float gain)
	{
		// 1. 逐 agent 硬写原生姿态
		for (int i = 0; i < numAgents; i++)
		{
			var isvc = inputServices[i];
			var ssvc = stateServices[i];
			if (ssvc == null || !ssvc.IsReady) continue;

			GameObject go = ssvc.GetPlayerObject();
			if (go == null) continue;
			var rootRb = go.GetComponent<Rigidbody2D>();
			if (rootRb == null) continue;
			Vector2 body = rootRb.position;

			foreach (var rb in go.GetComponentsInChildren<Rigidbody2D>(true))
			{
				int idx = MapRigidbodyToNativeIndex(rb, go.transform);
				if (idx < 0) continue;
				rb.position        = body + NATIVE_REL_POS[idx];
				rb.rotation        = NATIVE_ANGLE[idx];
				rb.velocity        = Vector2.zero;
				rb.angularVelocity = 0f;
			}

			// fakeCursor 吸附到锤头(Tip 目标位)：光标=锤子 → 零牵引
			Rigidbody2D fcRB = isvc?.GetFakeCursorRB();
			if (fcRB != null)
			{
				fcRB.position        = body + NATIVE_REL_POS[5];
				fcRB.velocity        = Vector2.zero;
				fcRB.angularVelocity = 0f;
			}
		}

		// 2. 零输入静置：让关节从精确姿态收敛到物理平衡（每帧把光标跟到当前锤头，避免牵引）
		int frames = settleFrames > 0 ? settleFrames : 1;
		for (int f = 0; f < frames; f++)
		{
			for (int i = 0; i < numAgents; i++)
			{
				var isvc = inputServices[i];
				var ssvc = stateServices[i];
				if (isvc == null || !isvc.IsReady) continue;

				Rigidbody2D fcRB = isvc.GetFakeCursorRB();
				if (fcRB != null && ssvc != null && ssvc.IsReady)
				{
					float[] s = ssvc.GetStateArray();
					if (s != null && s.Length >= BASE_STATE_DIM)
					{
						fcRB.position        = new Vector2(s[23], s[24]);
						fcRB.velocity        = Vector2.zero;
						fcRB.angularVelocity = 0f;
					}
				}
				isvc.SetMouseInput(Vector2.zero);
				isvc.InvokeFixedUpdate();
			}
			Physics2D.Simulate(Time.fixedDeltaTime);
		}

		// 3. 收尾：清零注入残留 + 光标再次吸附锤头
		GoiRuntime.PlayerControl.RewiredMouseOverride.Reset();
		for (int i = 0; i < numAgents; i++)
		{
			var isvc = inputServices[i];
			var ssvc = stateServices[i];
			if (isvc == null || !isvc.IsReady) continue;
			Rigidbody2D fcRB = isvc.GetFakeCursorRB();
			if (fcRB != null && ssvc != null && ssvc.IsReady)
			{
				float[] s = ssvc.GetStateArray();
				if (s != null && s.Length >= BASE_STATE_DIM)
				{
					fcRB.position = new Vector2(s[23], s[24]);
					fcRB.velocity = Vector2.zero;
				}
			}
			isvc.SetMouseInput(Vector2.zero);
		}

		// 4. 诊断：打印各 agent 锤臂相对 body 的关键偏移，核对是否与原生一致
		for (int i = 0; i < numAgents; i++)
		{
			var ssvc = stateServices[i];
			if (ssvc == null || !ssvc.IsReady) continue;
			float[] s = ssvc.GetStateArray();
			if (s == null || s.Length < BASE_STATE_DIM) continue;
			Vector2 bodyPos = new Vector2(s[0], s[1]);
			Debug.Log($"[StepController] agent{i} 原生姿态重建后 hub-body=({s[5]-s[0]:F2},{s[6]-s[1]:F2}) " +
				$"tip-body=({s[23]-s[0]:F2},{s[24]-s[1]:F2})（期望 hub≈(0.00,0.60) tip≈(1.75,-0.62)）");
		}

		Debug.Log($"[StepController] 初始姿态归一化(重建原生摆位)完成 (settle={frames})");
	}

	/// <summary>
	/// 把一个 Rigidbody2D 映射到 NATIVE_REL_POS / NATIVE_ANGLE 的索引。
	/// body 用「transform==根」判定（复制体根名可能带后缀），其余按部件名匹配。
	/// 返回 -1 表示非锤臂刚体（跳过）。
	/// </summary>
	private static int MapRigidbodyToNativeIndex(Rigidbody2D rb, Transform root)
	{
		if (rb.transform == root) return 0;
		switch (rb.name)
		{
			case "Hub":        return 1;
			case "Slider":     return 2;
			case "Handle":     return 3;
			case "PoleMiddle": return 4;
			case "Tip":        return 5;
			default:           return -1;
		}
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
			fakeCursorSnapshots.Clear();
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
						if (rbs.Length > 0)
							Debug.Log($"[StepController] agent{agentIdx} 快照 rb[0].pos={rbs[0].position:F2} rb[0].vel={rbs[0].velocity:F1}");
					}
				}
				else
				{
					Debug.LogWarning($"[StepController] agent{agentIdx} 状态服务未就绪，快照为空");
				}
				initialSnapshots.Add(list);

				Rigidbody2D fcRB = (agentIdx < inputServices.Count) ? inputServices[agentIdx]?.GetFakeCursorRB() : null;
				if (fcRB != null)
				{
					fakeCursorSnapshots.Add(new RigidbodySnapshot
					{
						rb              = fcRB,
						position        = fcRB.position,
						rotation        = fcRB.rotation,
						velocity        = fcRB.velocity,
						angularVelocity = fcRB.angularVelocity,
					});
					Debug.Log($"[StepController] agent{agentIdx} fakeCursor 快照 pos={fcRB.position:F2} ID={fcRB.GetInstanceID()}");
				}
				else
				{
					fakeCursorSnapshots.Add(default);
				}
			}
			Debug.Log($"[StepController] 已保存 {initialSnapshots.Count} 个 agent 的初始快照（含 fakeCursor）");
		}
	}
}
