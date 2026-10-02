using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.SceneManagement;
using GoiRuntime.Core.Configuration;
using GoiRuntime.Core.Interfaces;
using GoiRuntime.Core.Utilities;
using GoiRuntime.PlayerControl;
using GoiRuntime.Testing;
using GoiRuntime.CameraControl;
using GoiRuntime.ColliderCollection;

namespace GoiRuntime.Core
{
    /// <summary>
    /// N native players in N separately loaded Mian PhysicsScene2Ds. No cloned
    /// joints, shared solver or active-scene-dependent stepping. Snapshot is a
    /// replay recipe: rebuilding + replay restores solver history as well as pose.
    /// All entrypoints run on Unity's main thread.
    /// </summary>
    public sealed class IsolatedWorldRuntime : IMultiPlayerRuntime
    {
        sealed class World
        {
            public Scene scene;
            public GameObject player;
            public PlayerInputService input;
            public PlayerStateService state;
            public NativePhysicsInitialization native;
            public StepController controller;
        }
        sealed class Operation
        {
            public float[] actions;
            public int agent;
            public Vector2 target;
        }
        readonly RuntimeConfig config;
        readonly List<World> worlds = new List<World>();
        readonly List<Operation> history = new List<Operation>();
        List<Operation> snapshot = new List<Operation>();
        World pending;
        Exception loadError;
        int epoch, resetNumber;
        bool ready, replaying, cameraFree, visualEnabled = true, diagnosticsEnabled;
        Camera camera;
        FreeCameraController freeCamera;
        ColliderVisualizer visualizer;
        ObservationDiagnosticPacket observations;
        StepController.ResetFrame initialFrame;
        public int NumAgents { get { return config.numAgents; } }
        public int LastCollectSequence { get; private set; }
        public bool LastAnchorCollectOk { get; private set; }

        public IsolatedWorldRuntime(RuntimeConfig config)
        {
            this.config = config;
            if (NumAgents < 1 || NumAgents > 64) throw new ArgumentOutOfRangeException("numAgents", "Expected 1..64 players");
            if (config.stepFrames != 1 || config.actionDimension != 2)
                throw new ArgumentException("Native controlled input requires stepFrames=1 and actionDimension=2");
        }

        public IEnumerator Initialize() { return Rebuild(); }

        void SceneLoaded(Scene scene, LoadSceneMode mode)
        {
            if (scene.name != "Mian") return;
            try
            {
                SceneManager.SetActiveScene(scene); // Native Start creates fakeCursor in this scene.
                if (scene.GetPhysicsScene2D() == Physics2D.defaultPhysicsScene)
                    throw new InvalidOperationException("Player world uses the default physics scene");
                foreach (var prior in worlds)
                    if (prior.scene.GetPhysicsScene2D() == scene.GetPhysicsScene2D())
                        throw new InvalidOperationException("Players share a physics scene");
                GameObject player = null;
                foreach (var root in scene.GetRootGameObjects())
                    foreach (var transform in root.GetComponentsInChildren<Transform>(true))
                        if (transform.name == config.playerObjectName) player = transform.gameObject;
                if (player == null) throw new InvalidOperationException("Native Player missing");
                foreach (var root in scene.GetRootGameObjects())
                    foreach (var behaviour in root.GetComponentsInChildren<Behaviour>(true))
                    {
                        string name = behaviour.GetType().Name;
                        if (name == "Saviour" || name == "CameraControl" || name == "PoseControl" || name == "Animator")
                            behaviour.enabled = false;
                        if (name == "RestartOnContact") UnityEngine.Object.Destroy(behaviour);
                    }
                pending = new World { scene = scene, player = player,
                    input = new PlayerInputService(), state = new PlayerStateService(),
                    native = NativePhysicsInitialization.ApplyBeforeStart(player) };
                if (!pending.input.InitializeFor(player) || !pending.state.InitializeFor(player))
                    throw new InvalidOperationException("Native player binding failed");
                if (!string.IsNullOrEmpty(config.dynamicsProbeMode)) DynamicsProbe.BindScene();
            }
            catch (Exception ex) { loadError = ex; }
        }

        IEnumerator Rebuild()
        {
            Vector3 cameraPosition = camera != null ? camera.transform.position : Vector3.zero;
            Quaternion cameraRotation = camera != null ? camera.transform.rotation : Quaternion.identity;
            float zoom = camera != null ? camera.orthographicSize : 0f;
            bool restoreCamera = camera != null;
            ready = false;
            worlds.Clear();
            observations = null;
            Time.timeScale = 0f;
            Physics2D.simulationMode = SimulationMode2D.Script;
            Physics2D.autoSyncTransforms = false;
            PlayerControlFixedUpdatePatch.Enabled = true;
            PlayerControlUpdatePatch.RlModeActive = true;
            RewiredMouseOverride.NativeInputSmoothing = true;
            RewiredMouseOverride.Active = true;
            RewiredMouseOverride.Reset();
            SceneManager.sceneLoaded += SceneLoaded;
            try
            {
                var calibration = AnchorCalibration.Load();
                for (int i = 0; i < NumAgents; i++)
                {
                    pending = null;
                    loadError = null;
                    var loading = SceneManager.LoadSceneAsync("Mian", new LoadSceneParameters(
                        i == 0 ? LoadSceneMode.Single : LoadSceneMode.Additive, LocalPhysicsMode.Physics2D));
                    float deadline = Time.realtimeSinceStartup + 45f;
                    while (!loading.isDone || pending == null || !pending.input.NativeStartReady)
                    {
                        if (loadError != null) throw loadError;
                        if (Time.realtimeSinceStartup > deadline) throw new TimeoutException("Native world initialization timed out");
                        yield return null;
                    }
                    yield return null;
                    var world = pending;
                    if (world.input.GetFakeCursorRB().gameObject.scene != world.scene)
                        throw new InvalidOperationException("Native cursor belongs to another physics world");
                    PlayerInputService.PrepareControlledPlayer(world.player);
                    world.native.ValidateAfterStart(world.player);
                    world.controller = new StepController(1, config.stepFrames, StepController.STATE_DIM, 2);
                    world.controller.ResetDiagnosticsEnabled = config.experimentalResetDiagnostics;
                    if (!world.controller.Initialize(world.state, world.input, nativeNewGame: true))
                        throw new InvalidOperationException("Native world controller initialization failed");
                    world.input.AgentIndex = i;
                    world.controller.SetAnchorCalibration(calibration.Available, calibration.LocalX, calibration.LocalY);
                    worlds.Add(world);
                    Debug.Log("[MultiPlayer] world " + i + " ready; scene=" + world.scene.handle);
                }
            }
            finally { SceneManager.sceneLoaded -= SceneLoaded; }
            // A common handover boundary for 1, 2, 8... players. Additional scene
            // loads already synchronize old worlds; explicitly synchronize all here.
            Physics2D.SyncTransforms();
            SceneManager.SetActiveScene(worlds[0].scene);
            camera = null;
            foreach (var cam in UnityEngine.Object.FindObjectsOfType<Camera>())
            {
                bool primary = cam.gameObject.scene == worlds[0].scene;
                cam.enabled = primary;
                if (primary && (camera == null || cam.CompareTag("MainCamera"))) camera = cam;
            }
            foreach (var listener in UnityEngine.Object.FindObjectsOfType<AudioListener>())
                listener.enabled = listener.gameObject.scene == worlds[0].scene;
            if (camera != null)
            {
                visualizer = camera.gameObject.AddComponent<ColliderVisualizer>();
                visualizer.SetPlayerRoots(GetVisualizationRoots());
                visualizer.enabled = visualEnabled;
                freeCamera = camera.gameObject.AddComponent<FreeCameraController>();
                freeCamera.SetFreeMode(cameraFree);
                if (restoreCamera)
                {
                    camera.transform.position = cameraPosition;
                    camera.transform.rotation = cameraRotation;
                    camera.orthographicSize = zoom;
                }
            }
            epoch++;
            ready = true;
            initialFrame = config.experimentalResetDiagnostics ? CaptureFrame("initialization") : null;
            CollectAllStates();
        }

        public IEnumerator Reset()
        {
            var rebuild = Rebuild();
            while (rebuild.MoveNext()) yield return rebuild.Current;
            replaying = true;
            try
            {
                for (int i = 0; i < snapshot.Count; i++)
                {
                    Apply(snapshot[i]);
                    if ((i + 1) % 120 == 0) yield return null;
                }
            }
            finally { replaying = false; }
            history.Clear();
            history.AddRange(snapshot);
            resetNumber++;
        }

        void RequireReady()
        {
            if (!ready || worlds.Count != NumAgents) throw new InvalidOperationException("Multi-player runtime is not ready");
        }
        void Apply(Operation operation)
        {
            if (operation.actions != null)
                for (int i = 0; i < NumAgents; i++)
                    worlds[i].controller.ExecuteStep(new float[] { operation.actions[2*i], operation.actions[2*i+1] });
            else worlds[operation.agent].controller.Teleport(0, operation.target);
        }
        public float[] ExecuteStep(float[] actions)
        {
            RequireReady();
            if (actions == null || actions.Length != NumAgents * 2) throw new ArgumentException("Expected numAgents * 2 actions");
            foreach (float value in actions)
                if (float.IsNaN(value) || float.IsInfinity(value)) throw new ArgumentException("Actions must be finite");
            var operation = new Operation { actions = (float[])actions.Clone() };
            Apply(operation);
            if (!replaying) history.Add(operation);
            return CollectAllStates();
        }
        public float[] Teleport(int agent, Vector2 position)
        {
            RequireReady();
            if (agent < 0 || agent >= NumAgents) throw new ArgumentOutOfRangeException("agent");
            if (float.IsNaN(position.x) || float.IsNaN(position.y) || float.IsInfinity(position.x) || float.IsInfinity(position.y))
                throw new ArgumentException("Teleport position must be finite");
            var operation = new Operation { agent = agent, target = position };
            Apply(operation);
            if (!replaying) history.Add(operation);
            return CollectAllStates();
        }
        public void TakeNewSnapshot() { RequireReady(); snapshot = new List<Operation>(history); }
        public void ConfigureInput(bool active, int mouseX, int mouseY)
        {
            // A replayable environment cannot admit live physical mouse input or
            // change input mappings halfway through its saved recipe.
            if (!active || (mouseX >= 0 && mouseX != RewiredMouseOverride.MouseXActionId) ||
                (mouseY >= 0 && mouseY != RewiredMouseOverride.MouseYActionId))
                throw new InvalidOperationException("Multi-player input is fixed to native mouse axes with injected actions");
        }
        public float[] CollectAllStates()
        {
            RequireReady();
            var states = new float[NumAgents * StepController.STATE_DIM];
            LastAnchorCollectOk = true;
            for (int i = 0; i < NumAgents; i++)
            {
                Array.Copy(worlds[i].controller.CollectAllStates(), 0, states, i * StepController.STATE_DIM, StepController.STATE_DIM);
                LastAnchorCollectOk &= worlds[i].controller.LastAnchorCollectOk;
            }
            LastCollectSequence++;
            if (diagnosticsEnabled) observations = CaptureObservations(true);
            return states;
        }
        public GameObject GetPlayer(int agent) { return worlds[agent].player; }
        public GameObject[] GetVisualizationRoots()
        {
            var roots = new List<GameObject>();
            foreach (var world in worlds) roots.AddRange(world.controller.GetVisualizationRoots());
            return roots.ToArray();
        }
        public void SetCameraFree(bool enabled) { cameraFree = enabled; if (freeCamera != null) freeCamera.SetFreeMode(enabled); }
        public void SetVisualization(bool enabled) { visualEnabled = enabled; if (visualizer != null) visualizer.enabled = enabled; }
        public void ExportColliders(string directory)
        {
            RequireReady();
            ColliderExporter.ExportAll(worlds[0].controller.GetVisualizationRoots()[0], directory);
        }

        StepController.ResetFrame CaptureFrame(string phase)
        {
            var bodies = new List<StepController.ResetBody>();
            var controls = new List<PlayerControlInternalState>();
            var states = new List<float>();
            var joints = new List<string>();
            StepController.ResetFrame result = null;
            for (int i = 0; i < NumAgents; i++)
            {
                result = worlds[i].controller.CaptureResetFrame(phase);
                foreach (var body in result.bodies) { body.agent = i; bodies.Add(body); }
                controls.AddRange(result.controls);
                states.AddRange(result.states);
                foreach (var jointJson in result.joints)
                {
                    var joint = JsonUtility.FromJson<StepController.ResetJoint>(jointJson);
                    joint.agent = i;
                    joints.Add(JsonUtility.ToJson(joint));
                }
            }
            result.bodies = bodies.ToArray(); result.controls = controls.ToArray();
            result.states = states.ToArray(); result.joints = joints.ToArray();
            return result;
        }
        public string CollectResetDiagnostics()
        {
            RequireReady();
            if (!config.experimentalResetDiagnostics) throw new InvalidOperationException("Reset diagnostics disabled");
            var handles = new List<string>();
            foreach (var world in worlds) handles.Add(world.scene.handle.ToString());
            return "{\"version\":2,\"kind\":\"reset_diagnostics\",\"backend\":\"isolated_worlds\",\"agentCount\":" + NumAgents
                + ",\"physicsWorldCount\":" + worlds.Count + ",\"sceneHandles\":[" + string.Join(",", handles.ToArray()) + "]"
                + ",\"epoch\":" + epoch + ",\"resetNumber\":" + resetNumber + ",\"snapshotOperations\":" + snapshot.Count
                + ",\"controlledUpdates\":true,\"initialization\":" + StepController.ResetFrameJson(initialFrame)
                + ",\"current\":" + StepController.ResetFrameJson(CaptureFrame("live")) + ",\"stages\":[]}";
        }
        ObservationDiagnosticPacket CaptureObservations(bool aligned)
        {
            var agents = new ObservationDiagnosticAgent[NumAgents];
            for (int i = 0; i < NumAgents; i++) agents[i] = worlds[i].controller.CaptureAgentDiagnostics(camera);
            return new ObservationDiagnosticPacket { aligned = aligned, sequence = LastCollectSequence,
                fixedDeltaTime = Time.fixedDeltaTime, stepFrames = config.stepFrames, agents = agents };
        }
        public string CollectObservationDiagnostics()
        {
            RequireReady();
            diagnosticsEnabled = true;
            if (observations == null) observations = CaptureObservations(false);
            return ObservationDiagnostics.ToJson(observations);
        }
        public string CollectCalibrationCapture()
        {
            RequireReady();
            diagnosticsEnabled = true;
            observations = CaptureObservations(true);
            string dir = Path.Combine(PathManager.DataRootPath, "CalibrationCaptures");
            PathManager.EnsureDirectory(dir);
            string path = Path.Combine(dir, "seq_" + LastCollectSequence.ToString("D6") + ".ppm");
            int width, height;
            if (!CalibrationCapture.WriteCameraPpm(camera, path, out width, out height))
                throw new InvalidOperationException("Camera capture failed");
            observations.screenshotPath = path; observations.screenshotWidth = width; observations.screenshotHeight = height;
            observations.pixelOrigin = "image_top_left";
            return ObservationDiagnostics.ToJson(observations);
        }
    }
}
