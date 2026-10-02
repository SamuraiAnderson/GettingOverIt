using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.SceneManagement;
using GoiRuntime.Core.Utilities;
using GoiRuntime.PlayerControl;
using GoiRuntime.Core.Interfaces;

namespace GoiRuntime.Core
{
	/// <summary>Single native player controller; use IMultiPlayerRuntime for batches and reset.</summary>
	public class StepController
	{
		/// <summary>PlayerState.ToFloatArray() 的基础维度（不含 fakeCursor）。</summary>
		public const int BASE_STATE_DIM = 29;
		/// <summary>追加的 fakeCursor 维度：绝对坐标 x,y + 速度 vx,vy。</summary>
		public const int CURSOR_DIM = 4;
		/// <summary>经标定的参考锚点世界坐标 x/y。</summary>
		public const int ANCHOR_DIM = 2;
		/// <summary>旧客户端线路宽度（基础 + fakeCursor）。</summary>
		public const int LEGACY_WIRE_DIM = BASE_STATE_DIM + CURSOR_DIM;  // 33
		/// <summary>内部完整状态步幅（基础 + fakeCursor + 锚点）。</summary>
		public const int STATE_DIM = LEGACY_WIRE_DIM + ANCHOR_DIM;  // 35

		private readonly int numAgents;
		private readonly int stepFrames;
		private readonly int stateDim;
		private readonly int actionDim;
        private bool observationDiagnosticsEnabled;
        private int observationSequence;
        private GoiRuntime.Testing.ObservationDiagnosticPacket observationDiagnostics;
		private bool anchorCalibrationAvailable;
		private float anchorLocalX;
		private float anchorLocalY;
		public bool LastAnchorCollectOk { get; private set; }
		public int LastCollectSequence { get { return observationSequence; } }

		// 各 agent 的 Input 和 State 服务（索引 0 = 原始 Player）
		private readonly List<PlayerInputService> inputServices = new List<PlayerInputService>();
		private readonly List<PlayerStateService> stateServices = new List<PlayerStateService>();
        private readonly Dictionary<GameObject, Rigidbody2D[]> controlledBodies =
            new Dictionary<GameObject, Rigidbody2D[]>();
        private readonly Dictionary<GameObject, GameObject> visualizationRoots =
            new Dictionary<GameObject, GameObject>();

        public GameObject[] GetVisualizationRoots()
        {
            var roots = new List<GameObject>();
            foreach (var service in stateServices)
            {
                var player = service != null ? service.GetPlayerObject() : null;
                GameObject root;
                roots.Add(player != null && visualizationRoots.TryGetValue(player, out root) ? root : player);
            }
            return roots.ToArray();
        }

        private Rigidbody2D[] PlayerBodies(GameObject go)
        {
            Rigidbody2D[] bodies;
            return controlledBodies.TryGetValue(go, out bodies) ? bodies
                : go.GetComponentsInChildren<Rigidbody2D>(true);
        }

        private void SeparatePhysicsTransforms()
        {
            // Nested dynamic bodies round-trip world/local coordinates on render
            // synchronization. Keep references but give each body a static parent.
            foreach (var service in stateServices)
            {
                var go = service.GetPlayerObject();
                var bodies = go.GetComponentsInChildren<Rigidbody2D>(true);
                controlledBodies.Add(go, bodies);
                var container = new GameObject(go.name + "_PhysicsRoot");
                SceneManager.MoveGameObjectToScene(container, go.scene);
                foreach (var rb in bodies) rb.transform.SetParent(container.transform, true);
                visualizationRoots.Add(go, container);
            }
        }

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
		// 修复1：每个 agent 的 PlayerControl 动作相关内部累积态（reset 须一并恢复）
		private readonly List<PlayerControlInternalState> internalSnapshots = new List<PlayerControlInternalState>();


	private bool isInitialized;

		public bool IsReady => isInitialized;
		public int NumAgents => numAgents;
		public int StepFrames => stepFrames;
        public bool ResetDiagnosticsEnabled;
        private static long simulationCalls;
        private ResetFrame initializationFrame;

        [System.Serializable]
        public class ResetBody
        {
            public int agent;
            public string name;
            public Vector2 position, velocity;
            public float rotation, angularVelocity;
            public bool sleeping;
        }

        [System.Serializable]
        public class ResetFrame
        {
            public string phase;
            public long manualCalls, blockedCalls, simulationCalls;
            public int worldBodyCount;
            public float timeScale;
            public ResetBody[] bodies;
            public PlayerControlInternalState[] controls;
            public float[] states;
            public string[] joints;
        }

        internal static string ResetFrameJson(ResetFrame frame)
        {
            // Unity omits arrays of plugin-defined classes (see ObservationDiagnostics).
            var bodies = new List<string>();
            foreach (var body in frame.bodies) bodies.Add(JsonUtility.ToJson(body));
            var controls = new List<string>();
            foreach (var control in frame.controls) controls.Add(JsonUtility.ToJson(control));
            var header = JsonUtility.ToJson(frame);
            return header.Substring(0, header.Length - 1) + ",\"bodies\":["
                + string.Join(",", bodies.ToArray()) + "],\"controls\":["
                + string.Join(",", controls.ToArray()) + "]}";
        }

        internal ResetFrame CaptureResetFrame(string phase)
        {
            var bodies = new List<ResetBody>();
            var controls = new List<PlayerControlInternalState>();
            var joints = new List<string>();
            for (int a = 0; a < numAgents; a++)
            {
                var rbs = new List<Rigidbody2D>();
                foreach (var snap in initialSnapshots[a]) rbs.Add(snap.rb);
                var cursor = inputServices[a].GetFakeCursorRB();
                if (cursor != null && !rbs.Contains(cursor)) rbs.Add(cursor);
                foreach (var rb in rbs)
                {
                    if (rb == null) throw new System.InvalidOperationException("Missing reset body");
                    bodies.Add(new ResetBody { agent = a, name = rb.name,
                        position = rb.position, rotation = rb.rotation,
                        velocity = rb.velocity, angularVelocity = rb.angularVelocity,
                        sleeping = rb.IsSleeping() });
                }
                controls.Add(inputServices[a].GetInternalState());
                var agentJoints = new List<Joint2D>();
                foreach (var body in PlayerBodies(stateServices[a].GetPlayerObject()))
                    agentJoints.AddRange(body.GetComponents<Joint2D>());
                foreach (var joint in agentJoints)
                {
                    var anchored = joint as AnchoredJoint2D;
                    var hinge = joint as HingeJoint2D;
                    var slider = joint as SliderJoint2D;
                    joints.Add(JsonUtility.ToJson(new ResetJoint {
                        agent = a, name = joint.name, type = joint.GetType().Name,
                        connected = joint.connectedBody == null ? "world" : joint.connectedBody.name,
                        anchor = anchored == null ? Vector2.zero : anchored.anchor,
                        connectedAnchor = anchored == null ? Vector2.zero : anchored.connectedAnchor,
                        autoAnchor = anchored != null && anchored.autoConfigureConnectedAnchor,
                        motorSpeed = hinge != null ? hinge.motor.motorSpeed : slider != null ? slider.motor.motorSpeed : 0f,
                        maxMotorTorque = hinge != null ? hinge.motor.maxMotorTorque : slider != null ? slider.motor.maxMotorTorque : 0f,
                        enabled = joint.enabled, enableCollision = joint.enableCollision,
                        useMotor = hinge != null ? hinge.useMotor : slider != null && slider.useMotor
                    }));
                }
            }
            var full = CollectAllStates();
            var legacy = new float[numAgents * LEGACY_WIRE_DIM];
            for (int a = 0; a < numAgents; a++)
                System.Array.Copy(full, a * STATE_DIM, legacy, a * LEGACY_WIRE_DIM, LEGACY_WIRE_DIM);
            return new ResetFrame { phase = phase, bodies = bodies.ToArray(), controls = controls.ToArray(),
                states = legacy, joints = joints.ToArray(), manualCalls = PlayerControlFixedUpdatePatch.ManualCalls,
                timeScale = Time.timeScale,
                simulationCalls = simulationCalls,
                worldBodyCount = UnityEngine.Object.FindObjectsOfType<Rigidbody2D>().Length,
                blockedCalls = PlayerControlFixedUpdatePatch.BlockedCalls };
        }

        [System.Serializable]
        public class ResetJoint
        {
            public int agent;
            public string name, type, connected;
            public Vector2 anchor, connectedAnchor;
            public float motorSpeed, maxMotorTorque;
            public bool enabled, enableCollision, useMotor, autoAnchor;
        }

        private PhysicsScene2D physicsScene;

        private void SimulatePhysics(float dt)
        {
            if (PlayerControlFixedUpdatePatch.Enabled) Physics2D.SyncTransforms();
            bool success = PlayerControlFixedUpdatePatch.Enabled
                ? physicsScene.Simulate(dt)
                : Physics2D.Simulate(dt);
            if (PlayerControlFixedUpdatePatch.Enabled) Physics2D.SyncTransforms();
            simulationCalls++;
            if (PlayerControlFixedUpdatePatch.Enabled && !success)
                throw new System.InvalidOperationException("Physics2D.Simulate did not advance");
            GoiRuntime.Testing.DynamicsProbe.AfterManualPhysics();
        }

		public void SetAnchorCalibration(bool available, float localX, float localY)
		{
			anchorCalibrationAvailable = available;
			anchorLocalX = localX;
			anchorLocalY = localY;
		}

		public StepController(int numAgents, int stepFrames, int stateDim = 29, int actionDim = 2)
		{
			if (numAgents != 1) throw new System.ArgumentException("Use IMultiPlayerRuntime for multiple players");
            this.numAgents = 1;
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
			bool normalizePose = false,
			int settleFrames = 120,
			float upScale = 1.0f,
			float offsetX = 0f,
			float gain = 15f, bool nativeNewGame = false)
		{
            if (nativeNewGame && !PlayerControlFixedUpdatePatch.Enabled)
                throw new System.InvalidOperationException("Native new-game initialization requires controlled updates");
			inputServices.Clear();
			stateServices.Clear();
			initialSnapshots.Clear();

			// --- Agent 0：原始 Player ---
			originInputService.AgentIndex = 0;
			inputServices.Add(originInputService);
			stateServices.Add(originStateService);

            physicsScene = originStateService.GetPlayerObject().scene.GetPhysicsScene2D();

			// --- 切换物理引擎到 Script 模式（持久）---
			Physics2D.simulationMode = SimulationMode2D.Script;
			Debug.Log($"[StepController] Physics2D.simulationMode = Script（持久）");
			Debug.Log($"[StepController] fixedDeltaTime = {Time.fixedDeltaTime:F4}s，每 step 推进 {stepFrames} 帧");

			// --- 初始姿态归一化（须在拍快照前）---
            if (PlayerControlFixedUpdatePatch.Enabled)
            {
                if (nativeNewGame)
                {
                    // Preserve the original hierarchy throughout opening physics.
                    // Flatten only at the handover to stop idle render transforms
                    // from changing the ready state between TCP requests.
                    foreach (var input in inputServices) input.BeginNativeOpening();
                    InitializeNativeNewGamePose();
                }
                SeparatePhysicsTransforms();
                DisableBodyInterpolation();
                foreach (var input in inputServices)
                {
                    if (input == null || !input.NativeStartReady)
                        throw new System.InvalidOperationException("Cannot snapshot before native Start");
                    input.GetFakeCursorRB().interpolation = RigidbodyInterpolation2D.None;
                }
            }
			// Legacy normalization uses the R-reset template. Native new-game mode
            // has already reached its input-ready boundary and must not normalize again.
			if (!nativeNewGame && normalizePose)
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
				SimulatePhysics(dt);
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

        private static void RestoreTransform(RigidbodySnapshot snap)
        {
            // PlayerControl reads Transform positions, not just Rigidbody2D.position.
            // A body-only teleport can leave the old geometry visible until physics runs.
            var p = snap.rb.transform.position;
            snap.rb.transform.position = new Vector3(snap.position.x, snap.position.y, p.z);
            snap.rb.transform.rotation = Quaternion.Euler(0f, 0f, snap.rotation);
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
			var allRbs = PlayerBodies(playerObj);

			foreach (var rb in allRbs)
			{
				rb.position += delta;
                RestoreTransform(new RigidbodySnapshot { rb = rb, position = rb.position, rotation = rb.rotation });
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
                    RestoreTransform(new RigidbodySnapshot { rb = fcRB, position = fcRB.position, rotation = fcRB.rotation });
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
			SimulatePhysics(Time.fixedDeltaTime);

			Debug.Log($"[StepController] Teleport agent{agentIndex} → ({target.x:F2}, {target.y:F2}), delta=({delta.x:F2}, {delta.y:F2}), {allRbs.Length} rigidbodies moved");

			return CollectAllStates();
		}

		/// <summary>
		/// 采集各 agent 当前状态，返回 flat float[]
		/// </summary>
		public float[] CollectAllStates()
		{
			float[] result = new float[numAgents * stateDim];
			bool allAnchorsOk = anchorCalibrationAvailable;
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

				float ax = float.NaN, ay = float.NaN;
				Vector2 world = Vector2.zero;
				if (anchorCalibrationAvailable && stateServices[i] != null &&
					stateServices[i].TrySampleReferenceAnchor(anchorLocalX, anchorLocalY, out world))
				{
					ax = world.x;
					ay = world.y;
				}
				else
				{
					allAnchorsOk = false;
				}
				if (stateDim >= LEGACY_WIRE_DIM + ANCHOR_DIM)
				{
					result[baseOffset + LEGACY_WIRE_DIM + 0] = ax;
					result[baseOffset + LEGACY_WIRE_DIM + 1] = ay;
				}
			}
			LastAnchorCollectOk = allAnchorsOk;
			observationSequence++;
            if (observationDiagnosticsEnabled) CaptureObservationDiagnostics(true);
			return result;
		}

        internal GoiRuntime.Testing.ObservationDiagnosticAgent CaptureAgentDiagnostics(Camera camera)
        {
            return GoiRuntime.Testing.ObservationDiagnostics.Capture(inputServices[0], stateServices[0],
                camera, anchorCalibrationAvailable, anchorLocalX, anchorLocalY);
        }

        private void CaptureObservationDiagnostics(bool aligned)
        {
            var packet = new GoiRuntime.Testing.ObservationDiagnosticPacket
            {
                aligned = aligned,
                sequence = observationSequence,
                fixedDeltaTime = Time.fixedDeltaTime,
                stepFrames = stepFrames,
                agents = new GoiRuntime.Testing.ObservationDiagnosticAgent[numAgents]
            };
            Camera cam = Camera.main;
            for (int i = 0; i < numAgents; i++)
                packet.agents[i] = GoiRuntime.Testing.ObservationDiagnostics.Capture(
                    inputServices[i], stateServices[i], cam, anchorCalibrationAvailable, anchorLocalX, anchorLocalY);
            observationDiagnostics = packet;
        }

        public string CollectObservationDiagnostics()
        {
            observationDiagnosticsEnabled = true;
            if (observationDiagnostics == null)
            {
                observationSequence++;
                CaptureObservationDiagnostics(false);
            }
            return GoiRuntime.Testing.ObservationDiagnostics.ToJson(observationDiagnostics);
        }

		public string CollectCalibrationCapture()
		{
			observationDiagnosticsEnabled = true;
			DisableBodyInterpolation();
			observationSequence++;
			CaptureObservationDiagnostics(true);
			Camera cam = Camera.main;
			string dir = Path.Combine(PathManager.DataRootPath, "CalibrationCaptures");
			PathManager.EnsureDirectory(dir);
			string path = Path.Combine(dir, "seq_" + observationSequence.ToString("D6") + ".ppm");
			int width, height;
			if (observationDiagnostics != null &&
				GoiRuntime.Testing.CalibrationCapture.WriteCameraPpm(cam, path, out width, out height))
			{
				observationDiagnostics.screenshotPath = path;
				observationDiagnostics.screenshotWidth = width;
				observationDiagnostics.screenshotHeight = height;
				observationDiagnostics.pixelOrigin = "image_top_left";
			}
			return GoiRuntime.Testing.ObservationDiagnostics.ToJson(observationDiagnostics);
		}

		private void DisableBodyInterpolation()
		{
			for (int i = 0; i < stateServices.Count; i++)
			{
				if (stateServices[i] == null || !stateServices[i].IsReady) continue;
				GameObject go = stateServices[i].GetPlayerObject();
				if (go == null) continue;
				foreach (var rb in PlayerBodies(go))
				{
					if (rb != null)
						rb.interpolation = RigidbodyInterpolation2D.None;
				}
			}
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

    // New-game startup follows the native controller to input readiness without
    // overwriting body poses or adding an arbitrary number of settling steps.
    private void InitializeNativeNewGamePose()
    {
        // Keep the scene's own body layout. Let the original controller finish
        // its opening input pause; never write a reset pose or move the cursor.
        int frames = 0;
        while (true)
        {
            bool waiting = false;
            foreach (var input in inputServices)
                waiting |= input.GetInternalState().pauseInputTimer > 0f;
            if (!waiting) break;
            if (++frames > 2000) throw new System.InvalidOperationException("Native opening did not become ready");
            foreach (var input in inputServices)
            {
                input.SetMouseInput(Vector2.zero);
                input.InvokeFixedUpdate();
            }
            SimulatePhysics(Time.fixedDeltaTime);
        }
        Debug.Log("[StepController] Native new-game input ready after " + frames + " physical steps");
    }

    /// <summary>
    /// Legacy fixed start: write the R-reset template, then settle while pinning
    /// the cursor to the tip. This is not the native NewGame ready state.
    /// upScale/offsetX/gain are retained only for compatibility.
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
			Vector2 body = PlayerControlFixedUpdatePatch.Enabled
                ? new Vector2(-44.2926f, -2.4218f) : rootRb.position;

			foreach (var rb in PlayerBodies(go))
			{
				int idx = MapRigidbodyToNativeIndex(rb, go.transform);
				if (idx < 0) continue;
                if (PlayerControlFixedUpdatePatch.Enabled)
                    RestoreTransform(new RigidbodySnapshot { rb = rb,
                        position = body + NATIVE_REL_POS[idx], rotation = NATIVE_ANGLE[idx] });
				rb.position        = body + NATIVE_REL_POS[idx];
				rb.rotation        = NATIVE_ANGLE[idx];
				rb.velocity        = Vector2.zero;
				rb.angularVelocity = 0f;
			}

			// fakeCursor 吸附到锤头(Tip 目标位)：光标=锤子 → 零牵引
			Rigidbody2D fcRB = isvc?.GetFakeCursorRB();
			if (fcRB != null)
			{
				if (PlayerControlFixedUpdatePatch.Enabled)
                    RestoreTransform(new RigidbodySnapshot { rb = fcRB,
                        position = body + NATIVE_REL_POS[5], rotation = 0f });
				fcRB.position        = body + NATIVE_REL_POS[5];
				fcRB.velocity        = Vector2.zero;
				fcRB.angularVelocity = 0f;
			}
		}

        if (PlayerControlFixedUpdatePatch.Enabled) Physics2D.SyncTransforms();
		if (ResetDiagnosticsEnabled)
        {
            CaptureAllSnapshots();
            initializationFrame = CaptureResetFrame("before_normalization_steps");
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
			SimulatePhysics(Time.fixedDeltaTime);
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


	private void CaptureAllSnapshots()
		{
			initialSnapshots.Clear();
			fakeCursorSnapshots.Clear();
			internalSnapshots.Clear();
			for (int agentIdx = 0; agentIdx < stateServices.Count; agentIdx++)
			{
				var stateSvc = stateServices[agentIdx];
				var list = new List<RigidbodySnapshot>();
				if (stateSvc != null && stateSvc.IsReady)
				{
					GameObject go = stateSvc.GetPlayerObject();
					if (go != null)
					{
						var rbs = PlayerBodies(go);
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
                if (PlayerControlFixedUpdatePatch.Enabled && fcRB == null)
                    throw new System.InvalidOperationException("Missing fakeCursor in snapshot for agent " + agentIdx);
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

				// 修复1：采集 PlayerControl 动作相关内部累积态
				PlayerControlInternalState internalState =
					(agentIdx < inputServices.Count && inputServices[agentIdx] != null)
						? inputServices[agentIdx].GetInternalState()
						: default;
				internalSnapshots.Add(internalState);
			}



			Debug.Log($"[StepController] 已保存 {initialSnapshots.Count} 个 agent 的初始快照（含 fakeCursor + 内部态）");
		}

	}
}
