using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using HarmonyLib;
using UnityEngine;
using UnityEngine.LowLevel;
using GoiRuntime.Core.Configuration;
using GoiRuntime.PlayerControl;

namespace GoiRuntime.Testing
{
    // Opt-in recorder. Native mode preserves the game's lifecycle and automatic
    // physics. Only Rewired axis input and autosave-to-disk are intercepted.
    public static class DynamicsProbe
    {
        [Serializable] public class Fixture
        {
            public float[] actions;
            public int warmupFrames = 240;
            public string output;
        }
        [Serializable] public class Sample
        {
            public int step, renderFrame, controllerCalls, physicsCalls;
            public string phase, mode;
            public float fixedTime, fixedDeltaTime, timeScale;
            public Vector2 requested, inputBeforeInjection;
            public string[] bodyNames;
            public float[] axes, bodies, bodyConfig, controls, motors, contacts, motorParameters;
            public int[] axisReads;
        }
        private static string mode;
        private static Fixture fixture;
        private static Component pc;
        private static Rigidbody2D[] bodies;
        private static StreamWriter writer;
        private static int warmup, step, calls, physics;
        private static bool active, pending;
        private static readonly float[] axes = new float[4];
        private static readonly int[] reads = new int[4];
        private static Vector2 requested;
        private static Vector2 inputBeforeInjection;
        public static bool IsNative { get { return mode == "native" || mode == "native_dual"; } }
        private const BindingFlags Fields = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

        public static void Configure(RuntimeConfig config)
        {
            mode = config.dynamicsProbeMode;
            if (string.IsNullOrEmpty(mode)) return;
            if (!IsNative && mode != "candidate") throw new InvalidOperationException("Unknown dynamics probe mode");
            if (config.numDuplicates != 1) throw new InvalidOperationException("Dynamics probe requires one agent");
            if (IsNative && (config.experimentalControlledUpdates || config.experimentalSceneReset))
                throw new InvalidOperationException("Native reference cannot enable reset modifications");
            fixture = JsonUtility.FromJson<Fixture>(File.ReadAllText(config.dynamicsProbeFile));
            if (fixture.actions == null || fixture.actions.Length == 0 || fixture.actions.Length % 2 != 0)
                throw new InvalidOperationException("Invalid dynamics actions");
            if (File.Exists(fixture.output)) throw new InvalidOperationException("Refusing to overwrite dynamics trace");
            Directory.CreateDirectory(Path.GetDirectoryName(fixture.output));
            writer = new StreamWriter(fixture.output, false);
            writer.AutoFlush = true;
            var harmony = new Harmony("com.symbol.goi.dynamics.probe");
            var type = FindType("PlayerControl");
            harmony.Patch(type.GetMethod("FixedUpdate", Fields),
                prefix: new HarmonyMethod(typeof(DynamicsProbe), nameof(BeforeControl)) { priority = Priority.Last },
                postfix: new HarmonyMethod(typeof(DynamicsProbe), nameof(AfterControl)));
            var rewired = FindType("Rewired.Player");
            harmony.Patch(rewired.GetMethod("GetAxis", new[] { typeof(string) }),
                prefix: new HarmonyMethod(typeof(DynamicsProbe), nameof(AxisPrefix)) { priority = Priority.First },
                postfix: new HarmonyMethod(typeof(DynamicsProbe), nameof(AxisPostfix)) { priority = Priority.Last });
            // Keep native load/initialization, but don't write experiment progress into user saves.
            harmony.Patch(FindType("Saviour").GetMethod("SaveGameNow", Fields),
                prefix: new HarmonyMethod(typeof(DynamicsProbe), nameof(SkipAutosave)));
            if (IsNative)
            {
                var loop = PlayerLoop.GetCurrentPlayerLoop();
                if (!InsertAfterPhysics(ref loop)) throw new InvalidOperationException("Physics2D player-loop boundary not found");
                PlayerLoop.SetPlayerLoop(loop);
            }
        }

        private static Type FindType(string name)
        {
            foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
            {
                var type = assembly.GetType(name);
                if (type != null) return type;
            }
            throw new InvalidOperationException("Missing type: " + name);
        }

        public static bool SkipAutosave() { return false; }

        private static bool InsertAfterPhysics(ref PlayerLoopSystem system)
        {
            if (system.subSystemList == null) return false;
            var items = new List<PlayerLoopSystem>(system.subSystemList);
            for (int i = 0; i < items.Count; i++)
            {
                if (items[i].type != null && items[i].type.Name == "Physics2DFixedUpdate")
                {
                    items.Insert(i + 1, new PlayerLoopSystem { type = typeof(DynamicsProbe), updateDelegate = AfterNativePhysics });
                    system.subSystemList = items.ToArray();
                    return true;
                }
                var child = items[i];
                if (InsertAfterPhysics(ref child))
                {
                    items[i] = child;
                    system.subSystemList = items.ToArray();
                    return true;
                }
            }
            return false;
        }

        public static void BindScene()
        {
            if (string.IsNullOrEmpty(mode)) return;
            var go = GameObject.Find("Player");
            pc = go.GetComponent("PlayerControl");
            var found = go.GetComponentsInChildren<Rigidbody2D>(true);
            if (found.Length != 6) throw new InvalidOperationException("Expected six native player bodies");
            bodies = new Rigidbody2D[7];
            Array.Copy(found, bodies, 6); // Bind before candidate reparents the bodies.
            warmup = step = calls = physics = 0;
            active = pending = false;
        }

        private static object Read(string name)
        {
            var field = pc.GetType().GetField(name, Fields);
            if (field == null) throw new InvalidOperationException("Missing PlayerControl field: " + name);
            return field.GetValue(pc);
        }

        private static bool Ready()
        {
            return pc != null && (bool)Read("loadFinished") && (float)Read("pauseInputTimer") <= 0f
                && Read("fakeCursorRB") != null;
        }

        public static void BeginCandidate()
        {
            if (mode != "candidate" || !Ready() || active || writer == null)
                throw new InvalidOperationException("Candidate recorder cannot start");
            active = true;
            step = calls = physics = 0;
        }

        public static void BeforeControl(object __instance)
        {
            if (!ReferenceEquals(__instance, pc) || fixture == null || writer == null) return;
            if (mode == "candidate" && !RewiredMouseOverride.InsideInvokeFixedUpdate) return;
            if (IsNative && !active && Ready())
            {
                if (warmup++ >= fixture.warmupFrames) active = true;
            }
            if (!active) { requested = Vector2.zero; return; }
            if (pending) throw new InvalidOperationException("Controller advanced twice before recorded physics");
            requested = new Vector2(fixture.actions[step * 2], fixture.actions[step * 2 + 1]);
            inputBeforeInjection = (Vector2)Read("mouseInput");
            // Single-variable ablation under the native clock/hierarchy: emulate
            // the existing RL field write, while retaining exactly the same axes.
            if (mode == "native_dual") pc.GetType().GetField("mouseInput", Fields).SetValue(pc, requested);
            Array.Clear(axes, 0, axes.Length);
            Array.Clear(reads, 0, reads.Length);
            calls++;
            pending = true;
            Emit("before_control");
        }

        public static void AfterControl(object __instance)
        {
            if (pending && ReferenceEquals(__instance, pc)) Emit("after_control");
        }

        private static int AxisIndex(string name)
        {
            switch (name) { case "mouseX": return 0; case "mouseY": return 1;
                case "StickLX": case "StickRX": return 2;
                case "StickLY": case "StickRY": return 3; default: return -1; }
        }

        public static bool AxisPrefix(string __0, ref float __result)
        {
            int index = AxisIndex(__0);
            if (index < 0 || fixture == null) return true;
            if (index >= 2) { __result = 0f; return false; }
            if (!IsNative) return true;
            __result = active ? (index == 0 ? requested.x : requested.y) : 0f;
            return false;
        }

        public static void AxisPostfix(string __0, float __result)
        {
            int index = AxisIndex(__0);
            if (!pending || index < 0) return;
            axes[index] = __result;
            reads[index]++;
        }

        private static void AfterNativePhysics() { if (IsNative) AfterPhysics(); }
        public static void AfterManualPhysics() { if (mode == "candidate") AfterPhysics(); }

        private static void AfterPhysics()
        {
            if (!pending) return;
            physics++;
            Emit("after_physics");
            pending = false;
            step++;
            if (step * 2 == fixture.actions.Length)
            {
                active = false;
                writer.Close();
                writer = null;
                File.WriteAllText(fixture.output + ".done", step.ToString());
            }
        }

        private static void AppendVector(List<float> values, Vector2 v) { values.Add(v.x); values.Add(v.y); }

        private static void Emit(string phase)
        {
            bodies[6] = (Rigidbody2D)Read("fakeCursorRB");
            var states = new List<float>();
            var settings = new List<float>();
            var contacts = new List<float>();
            var names = new string[bodies.Length];
            for (int i = 0; i < bodies.Length; i++)
            {
                var rb = bodies[i];
                names[i] = rb.name;
                AppendVector(states, rb.position); states.Add(rb.rotation);
                AppendVector(states, rb.velocity); states.Add(rb.angularVelocity);
                settings.Add(rb.mass); settings.Add(rb.inertia); settings.Add(rb.gravityScale);
                settings.Add(rb.drag); settings.Add(rb.angularDrag); AppendVector(settings, rb.centerOfMass);
                settings.Add((float)rb.bodyType); settings.Add((float)rb.constraints);
                var points = new ContactPoint2D[32];
                int count = rb.GetContacts(points);
                contacts.Add(i); contacts.Add(count);
                for (int j = 0; j < count; j++)
                {
                    AppendVector(contacts, points[j].point); AppendVector(contacts, points[j].normal);
                    contacts.Add(points[j].normalImpulse); contacts.Add(points[j].tangentImpulse);
                }
            }
            var controls = new List<float>();
            AppendVector(controls, (Vector2)Read("mouseInput"));
            AppendVector(controls, (Vector2)Read("oldMouse"));
            controls.Add((float)Read("mouseVelocityAverage"));
            AppendVector(controls, (Vector2)Read("mw"));
            controls.Add((float)Read("oldAngle")); controls.Add((float)Read("pauseInputTimer"));
            controls.Add((float)Read("mouseSensitivity")); controls.Add((bool)Read("trackpad") ? 1 : 0);
            var hinge = (HingeJoint2D)Read("hj");
            var slider = (SliderJoint2D)Read("sj");
            float[] motors = { hinge.motor.motorSpeed, hinge.motor.maxMotorTorque,
                slider.motor.motorSpeed, slider.motor.maxMotorTorque, hinge.jointAngle, slider.jointTranslation };
            float[] motorParameters = { hinge.motor.maxMotorTorque, slider.motor.maxMotorTorque,
                bodies[0].GetComponent<HingeJoint2D>().motor.maxMotorTorque,
                ((JointMotor2D)Read("motor")).maxMotorTorque, ((JointMotor2D)Read("slider")).maxMotorTorque };
            var sample = new Sample { step = step, phase = phase, mode = mode,
                renderFrame = Time.frameCount, controllerCalls = calls, physicsCalls = physics,
                fixedTime = Time.fixedTime, fixedDeltaTime = Time.fixedDeltaTime, timeScale = Time.timeScale,
                requested = requested, inputBeforeInjection = inputBeforeInjection, bodyNames = names,
                axes = axes, axisReads = reads, bodies = states.ToArray(),
                bodyConfig = settings.ToArray(), controls = controls.ToArray(), motors = motors,
                motorParameters = motorParameters, contacts = contacts.ToArray() };
            writer.WriteLine(JsonUtility.ToJson(sample));
        }
    }
}
