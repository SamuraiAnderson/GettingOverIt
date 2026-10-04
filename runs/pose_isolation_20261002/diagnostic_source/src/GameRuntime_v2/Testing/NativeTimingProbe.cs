using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using HarmonyLib;
using UnityEngine;

namespace GoiRuntime.Testing
{
    // Test-only native lifecycle audit and explicitly labelled ablations. No TCP
    // or production reset behavior is changed. All audit serialization is deferred.
    public static class NativeTimingProbe
    {
        [Serializable] public class Event
        {
            public int sequence, renderFrame, controllers, physics, targetFrameRate, captureFramerate, vSyncCount;
            public string phase;
            public float fixedTime, time, realtime, deltaTime, fixedDeltaTime, timeScale;
            public bool loadFinished, mouseSnap, skipped, autoSync;
            public int inputsToSkip, simulationMode;
            public float pause, gain;
            public float[] bodies, transforms;
            public int[] interpolation;
            public string operationTarget;
            public float[] operationValue;
        }
        private static DynamicsProbe.Fixture fixture;
        private static Component pc;
        private static Rigidbody2D[] bodies;
        private static readonly List<Event> events = new List<Event>();
        private static int controllers, physics;
        private static bool finished;
        private const BindingFlags Flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

        public static void Configure(DynamicsProbe.Fixture settings, Harmony harmony)
        {
            if (!settings.timingAudit) return;
            fixture = settings;
            PoseWriteProbe.Configure(settings, harmony);
            foreach (string typeName in new[] { "PlayerControl", "PoseControl", "Saviour" })
            {
                var type = AccessTools.TypeByName(typeName);
                foreach (string methodName in new[] { "Start", "PlayOpeningAnimation", "Update", "LateUpdate" })
                {
                    var method = type.GetMethod(methodName, Flags);
                    if (method == null) continue;
                    harmony.Patch(method,
                        prefix: new HarmonyMethod(typeof(NativeTimingProbe), nameof(BeforeMethod)) { priority = Priority.First },
                        postfix: new HarmonyMethod(typeof(NativeTimingProbe), nameof(AfterMethod)) { priority = Priority.Last });
                }
            }
            ApplyClockOptions();
        }

        private static void ApplyClockOptions()
        {
            if (fixture.renderFps > 0)
            {
                if (QualitySettings.vSyncCount != 0) QualitySettings.vSyncCount = 0;
                if (Application.targetFrameRate != fixture.renderFps) Application.targetFrameRate = fixture.renderFps;
            }
            if (fixture.captureFps > 0 && Time.captureFramerate != fixture.captureFps)
                Time.captureFramerate = fixture.captureFps;
        }

        public static void Bind(Component controller, Rigidbody2D[] actorBodies)
        {
            if (fixture == null) return;
            pc = controller; bodies = actorBodies;
            events.Clear(); controllers = physics = 0; finished = false;
            ApplyOptions();
            Capture("scene_bound");
        }

        private static object Read(string name) { return pc.GetType().GetField(name, Flags).GetValue(pc); }

        private static void ApplyOptions()
        {
            ApplyClockOptions();
            bodies[6] = (Rigidbody2D)Read("fakeCursorRB");
            if (fixture.noInterpolation)
                foreach (var body in bodies)
                    if (body != null && body.interpolation != RigidbodyInterpolation2D.None)
                        body.interpolation = RigidbodyInterpolation2D.None;
        }

        private static bool Relevant(object instance)
        {
            var component = instance as Component;
            return fixture != null && pc != null && !finished && component != null
                && component.gameObject.scene == pc.gameObject.scene;
        }

        public static bool IsRecording(object instance) { return Relevant(instance); }

        public static bool BeforeMethod(object __instance, MethodBase __originalMethod)
        {
            if (!Relevant(__instance)) return true;
            string name = __originalMethod.DeclaringType.Name + "." + __originalMethod.Name;
            bool skip = (fixture.skipPlayerUpdate && name == "PlayerControl.Update")
                || (fixture.skipPoseLateUpdate && name == "PoseControl.LateUpdate");
            Capture("before_" + name, skip);
            return !skip;
        }

        public static void AfterMethod(object __instance, MethodBase __originalMethod)
        {
            if (!Relevant(__instance)) return;
            Capture("after_" + __originalMethod.DeclaringType.Name + "." + __originalMethod.Name);
        }

        public static void BeforeController()
        {
            if (fixture == null || pc == null || finished) return;
            ApplyOptions();
            controllers++;
            Capture("before_controller");
        }

        public static void AfterController() { Capture("after_controller"); }
        public static void BeforePhysics() { Capture("before_physics"); }
        public static void AfterPhysics()
        {
            if (fixture == null || pc == null || finished) return;
            physics++;
            Capture("after_physics");
        }

        public static void CaptureOperation(string phase, bool skipped, string target, float[] value)
        {
            Capture(phase, skipped, target, value);
        }

        private static void Capture(string phase, bool skipped = false, string target = null, float[] value = null)
        {
            if (fixture == null || pc == null || finished) return;
            if (events.Count > 100000) throw new InvalidOperationException("Timing audit exceeded its bounded event budget");
            bodies[6] = (Rigidbody2D)Read("fakeCursorRB");
            var states = new float[42];
            var transforms = new float[42];
            var interpolation = new int[7];
            for (int i = 0; i < bodies.Length; i++)
            {
                var rb = bodies[i];
                if (rb == null) { interpolation[i] = -1; continue; }
                int j = i * 6;
                states[j] = rb.position.x; states[j + 1] = rb.position.y; states[j + 2] = rb.rotation;
                states[j + 3] = rb.velocity.x; states[j + 4] = rb.velocity.y; states[j + 5] = rb.angularVelocity;
                var pose = rb.transform;
                transforms[j] = pose.position.x; transforms[j + 1] = pose.position.y;
                transforms[j + 2] = pose.position.z; transforms[j + 3] = pose.eulerAngles.z;
                transforms[j + 4] = pose.lossyScale.x; transforms[j + 5] = pose.lossyScale.y;
                interpolation[i] = (int)rb.interpolation;
            }
            events.Add(new Event { sequence = events.Count, renderFrame = Time.frameCount,
                controllers = controllers, physics = physics, phase = phase, skipped = skipped,
                fixedTime = Time.fixedTime, time = Time.time, realtime = Time.realtimeSinceStartup,
                deltaTime = Time.deltaTime, fixedDeltaTime = Time.fixedDeltaTime, timeScale = Time.timeScale,
                loadFinished = (bool)Read("loadFinished"), mouseSnap = (bool)Read("mouseSnap"),
                inputsToSkip = (int)Read("inputsToSkip"), pause = (float)Read("pauseInputTimer"),
                gain = (float)Read("mouseVelocityAverage"), targetFrameRate = Application.targetFrameRate,
                captureFramerate = Time.captureFramerate, vSyncCount = QualitySettings.vSyncCount,
                autoSync = Physics2D.autoSyncTransforms,
                simulationMode = (int)Physics2D.simulationMode,
                bodies = states, transforms = transforms, interpolation = interpolation,
                operationTarget = target, operationValue = value });
        }

        public static void Finish()
        {
            if (fixture == null) return;
            finished = true;
            using (var output = new StreamWriter(fixture.output + ".timeline.jsonl", false))
                foreach (var entry in events) output.WriteLine(JsonUtility.ToJson(entry));
            File.WriteAllText(fixture.output + ".timeline.done", events.Count.ToString());
            PoseWriteProbe.Finish();
            events.Clear();
        }
    }
}
