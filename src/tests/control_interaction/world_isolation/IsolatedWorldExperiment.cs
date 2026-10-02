using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Reflection;
using BepInEx;
using HarmonyLib;
using UnityEngine;
using UnityEngine.SceneManagement;
using GoiRuntime.Core;
using GoiRuntime.PlayerControl;
using GoiRuntime.Communication;
using GoiRuntime.Core.Interfaces;
using GoiRuntime.Core.Protocol;
using GoiRuntime.ColliderCollection;
using GoiRuntime.CameraControl;

[BepInPlugin("com.symbol.goi.isolated_world_experiment", "Isolated world experiment", "0.1.0")]
[BepInDependency("com.symbol.goi.runtime_v2")]
public class IsolatedWorldExperiment : BaseUnityPlugin
{
    [Serializable] public class ExperimentConfig
    {
        public string output;
        public int worlds = 2, resets = 1, burnSteps = 200;
        public bool otherZero, startupSync, sceneSyncOnly, livePolicy;
        public int liveSteps = 1200;
        public float[] actions;
    }
    [Serializable] public class EndState { public string status, error; public int worlds, epochs, steps; }
    [Serializable] public class WorldInfo
    {
        public int slot, sceneHandle, bodyCount, colliderCount;
        public string sceneName;
        public bool valid, isDefault;
        public float fixedDeltaTime;
    }
    [Serializable] public class ShapeInfo
    {
        public string path, type, material, bodyType, detection;
        public int layer, pathCount;
        public bool enabled, active, trigger;
        public float friction, bounciness, density;
        public Vector3 position, scale, euler;
        public Vector2 offset;
        public float[] points;
    }
    class World
    {
        public Scene scene;
        public StepController controller;
    }
    static readonly BindingFlags Flags = BindingFlags.Instance | BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public;
    static IsolatedWorldExperiment self;
    static ExperimentConfig cfg;
    static readonly List<World> worlds = new List<World>();
    static GameRuntimeManager manager;
    static bool started;

    void Awake()
    {
        string path = Path.Combine(Paths.GameRootPath, "GoiData/isolation_experiment.json");
        if (!File.Exists(path)) return;
        cfg = JsonUtility.FromJson<ExperimentConfig>(File.ReadAllText(path));
        if (cfg.worlds < 1 || cfg.worlds > 8 || (!cfg.livePolicy && (cfg.actions == null || cfg.actions.Length % 2 != 0)))
            throw new InvalidOperationException("Invalid experiment config");
        self = this;
        var h = new Harmony("com.symbol.goi.isolated_world_experiment");
        h.Patch(typeof(GameRuntimeManager).GetMethod("OnSceneLoaded", Flags),
            new HarmonyMethod(typeof(IsolatedWorldExperiment).GetMethod("BeforeSceneLoaded", Flags)));
        h.Patch(typeof(GameRuntimeManager).GetMethod("FinishGameRuntimeMode", Flags), null,
            new HarmonyMethod(typeof(IsolatedWorldExperiment).GetMethod("AfterReady", Flags)));
        if (cfg.sceneSyncOnly)
            h.Patch(typeof(StepController).GetMethod("SimulatePhysics", Flags),
                new HarmonyMethod(typeof(IsolatedWorldExperiment).GetMethod("SimulateSceneOnly", Flags)));
        Logger.LogInfo("Independent-world experiment armed");
    }
    static void Set(string field, object value) { typeof(GameRuntimeManager).GetField(field, Flags).SetValue(manager, value); }
    static bool SimulateSceneOnly(float dt)
    {
        if (!SceneManager.GetActiveScene().GetPhysicsScene2D().Simulate(dt))
            throw new InvalidOperationException("Scene simulation failed");
        var counter = typeof(StepController).GetField("simulationCalls", Flags);
        counter.SetValue(null, (long)counter.GetValue(null) + 1);
        return false;
    }
    static void BeforeSceneLoaded(Scene scene)
    {
        if (scene.name == "Mian" && scene.GetPhysicsScene2D() != Physics2D.defaultPhysicsScene)
            SceneManager.SetActiveScene(scene);
    }
    static void AfterReady(GameRuntimeManager __instance)
    {
        manager = __instance;
        var controller = (StepController)typeof(GameRuntimeManager).GetField("stepController", Flags).GetValue(manager);
        if (controller == null || controller.NumAgents != 1 || !controller.IsReady)
            throw new InvalidOperationException("Each world must initialize exactly one native player");
        var stateServices = (List<PlayerStateService>)typeof(StepController).GetField("stateServices", Flags).GetValue(controller);
        var player = stateServices[0].GetPlayerObject();
        var scene = player.scene;
        if (scene.GetPhysicsScene2D() == Physics2D.defaultPhysicsScene)
            throw new InvalidOperationException("World is not isolated");
        foreach (var world in worlds)
            if (world.scene.GetPhysicsScene2D() == scene.GetPhysicsScene2D())
                throw new InvalidOperationException("Physics worlds share a handle");
        worlds.Add(new World { scene = scene, controller = controller });
        File.WriteAllText(Path.Combine(cfg.output, "ready_" + scene.handle + ".json"), controller.CollectResetDiagnostics());
        // Native references are already bound. Keep later GameObject.Find("Player")
        // scoped by leaving that name only on the next newly loaded scene's player.
        player.name = "ExperimentPlayer_" + (worlds.Count - 1);
        self.Logger.LogInfo("World ready: slot=" + (worlds.Count - 1) + " scene=" + scene.handle);
        if (!started)
        {
            started = true;
            self.StartCoroutine(self.RunGuarded());
        }
    }
    IEnumerator RunGuarded()
    {
        var run = Run();
        while (true)
        {
            bool more = false;
            try { more = run.MoveNext(); }
            catch (Exception e)
            {
                File.WriteAllText(Path.Combine(cfg.output, "finished.json"),
                    JsonUtility.ToJson(new EndState { status = "ERROR", error = e.ToString() }));
                Logger.LogError(e);
                yield break;
            }
            if (!more) break;
            yield return run.Current;
        }
    }
    IEnumerator Run()
    {
        yield return null; // Allow the surrounding native initialization to finish.
        for (int epoch = 0; epoch <= cfg.resets; epoch++)
        {
            if (epoch > 0)
            {
                foreach (var world in worlds)
                    if (!world.scene.IsValid()) throw new InvalidOperationException("Premature world loss");
                worlds.Clear();
                var reset = (IEnumerator)typeof(GameRuntimeManager).GetMethod("ResetSceneAndRespond", Flags).Invoke(manager, null);
                while (reset.MoveNext()) yield return reset.Current;
                if (worlds.Count != 1) throw new InvalidOperationException("RESET did not rebuild the primary world");
            }
            while (worlds.Count < cfg.worlds)
            {
                int expected = worlds.Count + 1;
                Set("isInitialized", false);
                var loading = SceneManager.LoadSceneAsync("Mian", new LoadSceneParameters(LoadSceneMode.Additive, LocalPhysicsMode.Physics2D));
                float deadline = Time.realtimeSinceStartup + 45f;
                while (!loading.isDone || worlds.Count < expected)
                {
                    if (Time.realtimeSinceStartup > deadline) throw new InvalidOperationException("Additional world initialization timed out");
                    yield return null;
                }
                yield return null;
            }
            if (cfg.startupSync) Physics2D.SyncTransforms();
            WriteMetadata(epoch);
            WriteDiagnostics(epoch, 0);
            if (cfg.livePolicy)
            {
                yield return RunLivePolicy();
                yield break;
            }
            string trace = Path.Combine(cfg.output, "trace_epoch_" + epoch.ToString("00") + ".jsonl");
            using (var writer = new StreamWriter(trace, false))
            {
                WriteStates(writer, 0);
                for (int t = 0; t < cfg.actions.Length / 2; t++)
                {
                    for (int i = 0; i < worlds.Count; i++)
                    {
                        SceneManager.SetActiveScene(worlds[i].scene);
                        bool zero = cfg.otherZero && i > 0;
                        worlds[i].controller.ExecuteStep(new float[] { zero ? 0f : cfg.actions[2*t], zero ? 0f : cfg.actions[2*t+1] });
                    }
                    WriteStates(writer, t + 1);
                    if (t + 1 == 30 || t + 1 == 31 || t + 1 == 32 || t + 1 == 40 || t + 1 == cfg.actions.Length / 2)
                        WriteDiagnostics(epoch, t + 1);
                    if ((t + 1) % 120 == 0)
                    {
                        writer.Flush();
                        Logger.LogInfo("Independent worlds epoch=" + epoch + " step=" + (t + 1));
                        yield return null;
                    }
                }
            }
            if (epoch < cfg.resets)
                for (int t = 0; t < cfg.burnSteps; t++)
                    for (int i = 0; i < worlds.Count; i++)
                    {
                        SceneManager.SetActiveScene(worlds[i].scene);
                        worlds[i].controller.ExecuteStep(new float[] { i == 0 ? 40f : -70f, i == 0 ? -30f : 50f });
                    }
        }
        File.WriteAllText(Path.Combine(cfg.output, "finished.json"),
            JsonUtility.ToJson(new EndState { status = "COMPLETE", worlds = worlds.Count, epochs = cfg.resets + 1, steps = cfg.actions.Length / 2 }));
    }
    IEnumerator RunLivePolicy()
    {
        // Native initialization has finished. Replace only command dispatch;
        // the established TCP framing and per-world controller remain unchanged.
        manager.StopAllCoroutines();
        var server = (TcpStepServer)typeof(GameRuntimeManager).GetField("tcpStepServer", Flags).GetValue(manager);
        var oldHello = (HelloSnapshot)typeof(TcpStepServer).GetField("_helloSnapshot", Flags).GetValue(server);
        var hello = new HelloSnapshot {
            NumAgents = worlds.Count, FixedDeltaTime = oldHello.FixedDeltaTime, StepFrames = oldHello.StepFrames,
            AnchorId = oldHello.AnchorId, CalibrationAvailable = oldHello.CalibrationAvailable,
            CalibrationSha256 = oldHello.CalibrationSha256
        };
        if (server.IsConnected) throw new InvalidOperationException("Client connected before worlds were ready");
        server.Stop();
        server = new TcpStepServer(StepController.STATE_DIM, StepController.LEGACY_WIRE_DIM, hello);
        Set("tcpStepServer", server);
        var runtimeCfg = typeof(GameRuntimeManager).GetField("config", Flags).GetValue(manager);
        int port = (int)runtimeCfg.GetType().GetField("tcpPort").GetValue(runtimeCfg);
        server.StartListening(port);

        UnityEngine.Camera view = null;
        foreach (var root in worlds[0].scene.GetRootGameObjects())
            foreach (var cam in root.GetComponentsInChildren<UnityEngine.Camera>(true))
                if (cam.CompareTag("MainCamera")) view = cam;
        if (view == null) throw new InvalidOperationException("Primary camera missing");
        foreach (var visual in UnityEngine.Object.FindObjectsOfType<ColliderVisualizer>()) visual.enabled = false;
        foreach (var camera in UnityEngine.Object.FindObjectsOfType<UnityEngine.Camera>()) camera.enabled = camera == view;
        foreach (var listener in UnityEngine.Object.FindObjectsOfType<AudioListener>()) listener.enabled = listener.gameObject == view.gameObject;
        foreach (var free in UnityEngine.Object.FindObjectsOfType<FreeCameraController>()) free.enabled = false;
        var roots = new List<GameObject>();
        foreach (var world in worlds) roots.AddRange(world.controller.GetVisualizationRoots());
        var visualizer = view.gameObject.AddComponent<ColliderVisualizer>();
        visualizer.SetPlayerRoots(roots.ToArray());
        // Draw one identical terrain copy while drawing all eight physical players.
        var terrain = new List<Collider2D>();
        foreach (var root in worlds[0].scene.GetRootGameObjects())
            foreach (var collider in root.GetComponentsInChildren<PolygonCollider2D>())
                if (!collider.isTrigger && collider.GetComponentInParent<Rigidbody2D>() == null) terrain.Add(collider);
        typeof(ColliderVisualizer).GetField("_envColliders", Flags).SetValue(visualizer, terrain.ToArray());
        var freeCamera = view.gameObject.AddComponent<FreeCameraController>();
        freeCamera.initialZoom = 28f;
        freeCamera.SetFreeMode(true);
        Set("colliderVisualizer", visualizer);
        Set("freeCameraController", freeCamera);
        SceneManager.SetActiveScene(worlds[0].scene);
        // Only camera transforms are adjusted. Player/body transforms stay untouched.
        var firstState = worlds[0].controller.CollectAllStates();
        view.transform.position = new Vector3(firstState[0] + 5f, firstState[1] + 3f, view.transform.position.z);
        Application.targetFrameRate = 120;
        using (var writer = new StreamWriter(Path.Combine(cfg.output, "live_trace.jsonl"), false))
        {
            WriteStates(writer, 0);
            writer.Flush();
            string readyPath = Path.Combine(cfg.output, "worlds_ready.json");
            using (var ready = new StreamWriter(readyPath + ".tmp", false)) WriteStates(ready, 0);
            File.Move(readyPath + ".tmp", readyPath);
            Logger.LogInfo("POLICY READY: " + worlds.Count + " independent worlds, TCP port=" + port + ", free camera enabled");
            int steps = 0, sequence = 0;
            while (server.IsRunning)
            {
                StepCommand command;
                if (server.TryGetCommand(out command))
                {
                    if (command.Type == CommandType.Close)
                    {
                        // CLOSE is posted by the TCP thread without awaiting a response.
                        // Do not overwrite a completed trajectory with an unsupported-command error.
                        string closedPath = Path.Combine(cfg.output, "closed.json");
                        File.WriteAllText(closedPath + ".tmp", JsonUtility.ToJson(new EndState {
                            status = steps == cfg.liveSteps ? "COMPLETE" : "INTERRUPTED",
                            worlds = worlds.Count, epochs = 1, steps = steps }));
                        File.Move(closedPath + ".tmp", closedPath);
                        yield break;
                    }
                    var response = new StepResponse { States = new float[0], Dones = new bool[0], Sequence = ++sequence };
                    try
                    {
                        if (command.Type == CommandType.Step)
                        {
                            if (steps >= cfg.liveSteps || command.Actions == null || command.Actions.Length != worlds.Count * 2)
                                throw new InvalidOperationException("Invalid action batch or test horizon exceeded");
                            var states = new List<float>();
                            foreach (var world in worlds)
                            {
                                int index = worlds.IndexOf(world);
                                SceneManager.SetActiveScene(world.scene);
                                var state = world.controller.ExecuteStep(new float[] { command.Actions[index * 2], command.Actions[index * 2 + 1] });
                                if (!world.controller.LastAnchorCollectOk) throw new InvalidOperationException("Anchor calibration unavailable");
                                states.AddRange(state);
                            }
                            SceneManager.SetActiveScene(worlds[0].scene);
                            response.States = states.ToArray();
                            response.Dones = new bool[worlds.Count];
                            steps++;
                            WriteStates(writer, steps);
                            if (steps == 31 || steps == cfg.liveSteps) WriteDiagnostics(0, steps);
                            if (steps % 120 == 0) { writer.Flush(); Logger.LogInfo("POLICY STEP " + steps); }
                            if (steps == cfg.liveSteps)
                            {
                                writer.Flush();
                                File.WriteAllText(Path.Combine(cfg.output, "finished.json"), JsonUtility.ToJson(new EndState {
                                    status = "COMPLETE", worlds = worlds.Count, epochs = 1, steps = steps }));
                            }
                        }
                        else if (command.Type == CommandType.CameraFree) freeCamera.SetFreeMode(command.CameraFreeEnabled);
                        else if (command.Type == CommandType.Visualize) visualizer.enabled = command.VisualizeEnabled;
                        else if (command.Type == CommandType.Config)
                        {
                            RewiredMouseOverride.Active = command.ConfigRewiredMouseActive;
                            if (command.ConfigMouseXActionId >= 0) RewiredMouseOverride.MouseXActionId = command.ConfigMouseXActionId;
                            if (command.ConfigMouseYActionId >= 0) RewiredMouseOverride.MouseYActionId = command.ConfigMouseYActionId;
                        }
                        else throw new InvalidOperationException("Cold live test does not implement " + command.Type);
                    }
                    catch (Exception error)
                    {
                        Logger.LogError(error);
                        response.ErrorCode = 10;
                        response.ErrorMessage = error.Message;
                        File.WriteAllText(Path.Combine(cfg.output, "finished.json"), JsonUtility.ToJson(new EndState { status = "ERROR", error = error.ToString() }));
                    }
                    server.SendResponse(response);
                }
                yield return null;
            }
        }
    }
    static string Numbers(float[] values)
    {
        string[] strings = new string[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            if (Single.IsNaN(values[i]) || Single.IsInfinity(values[i])) throw new InvalidOperationException("Nonfinite state");
            strings[i] = values[i].ToString("R", CultureInfo.InvariantCulture);
        }
        return "[" + String.Join(",", strings) + "]";
    }
    static void WriteStates(StreamWriter writer, int step)
    {
        var states = new List<string>();
        foreach (var world in worlds) states.Add(Numbers(world.controller.CollectAllStates()));
        writer.WriteLine("{\"step\":" + step + ",\"states\":[" + String.Join(",", states.ToArray()) + "]}");
    }
    static void WriteDiagnostics(int epoch, int step)
    {
        var json = new List<string>();
        foreach (var world in worlds) json.Add(world.controller.CollectResetDiagnostics());
        File.WriteAllText(Path.Combine(cfg.output, "diagnostics_" + epoch + "_" + step + ".json"), "[" + String.Join(",", json.ToArray()) + "]");
    }
    static string PathOf(Transform t)
    {
        string path = t.name;
        while (t.parent != null) { t = t.parent; path = t.name + "/" + path; }
        return path;
    }
    static void WriteMetadata(int epoch)
    {
        var descriptions = new List<string>();
        for (int i = 0; i < worlds.Count; i++)
        {
            var world = worlds[i];
            var shapes = new List<string>();
            int bodyCount = 0, colliderCount = 0;
            foreach (var root in world.scene.GetRootGameObjects())
            {
                bodyCount += root.GetComponentsInChildren<Rigidbody2D>(true).Length;
                foreach (var c in root.GetComponentsInChildren<Collider2D>(true))
                {
                    colliderCount++;
                    var points = new List<float>();
                    var polygon = c as PolygonCollider2D;
                    if (polygon != null)
                        for (int p = 0; p < polygon.pathCount; p++)
                            foreach (var xy in polygon.GetPath(p)) { points.Add(xy.x); points.Add(xy.y); }
                    var edge = c as EdgeCollider2D;
                    if (edge != null) foreach (var xy in edge.points) { points.Add(xy.x); points.Add(xy.y); }
                    var box = c as BoxCollider2D;
                    if (box != null) { points.Add(box.size.x); points.Add(box.size.y); points.Add(box.edgeRadius); }
                    var circle = c as CircleCollider2D;
                    if (circle != null) points.Add(circle.radius);
                    var capsule = c as CapsuleCollider2D;
                    if (capsule != null) { points.Add(capsule.size.x); points.Add(capsule.size.y); points.Add((float)capsule.direction); }
                    var material = c.sharedMaterial;
                    var rb = c.attachedRigidbody;
                    shapes.Add(JsonUtility.ToJson(new ShapeInfo { path = PathOf(c.transform), type = c.GetType().Name,
                        material = material == null ? "default" : material.name,
                        friction = material == null ? -1f : material.friction,
                        bounciness = material == null ? -1f : material.bounciness,
                        bodyType = rb == null ? "none" : rb.bodyType.ToString(),
                        detection = rb == null ? "none" : rb.collisionDetectionMode.ToString(),
                        layer = c.gameObject.layer, pathCount = polygon == null ? 0 : polygon.pathCount,
                        enabled = c.enabled, active = c.gameObject.activeInHierarchy, trigger = c.isTrigger,
                        density = c.density, position = c.transform.position, euler = c.transform.eulerAngles,
                        scale = c.transform.lossyScale, offset = c.offset, points = points.ToArray() }));
                }
            }
            var info = new WorldInfo { slot = i, sceneHandle = world.scene.handle, sceneName = world.scene.name,
                valid = world.scene.GetPhysicsScene2D().IsValid(), isDefault = world.scene.GetPhysicsScene2D() == Physics2D.defaultPhysicsScene,
                bodyCount = bodyCount, colliderCount = colliderCount, fixedDeltaTime = Time.fixedDeltaTime };
            string header = JsonUtility.ToJson(info);
            descriptions.Add(header.Substring(0, header.Length - 1) + ",\"colliders\":[" + String.Join(",", shapes.ToArray()) + "]}");
        }
        File.WriteAllText(Path.Combine(cfg.output, "worlds_epoch_" + epoch + ".json"), "[" + String.Join(",", descriptions.ToArray()) + "]");
    }
}
