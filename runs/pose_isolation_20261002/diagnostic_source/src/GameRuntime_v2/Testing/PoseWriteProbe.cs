using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Reflection.Emit;
using HarmonyLib;
using UnityEngine;

namespace GoiRuntime.Testing
{
    // Replaces only call sites inside PoseControl.LateUpdate, never Unity's
    // global setters. Wrappers pass the original arguments through unchanged
    // unless a named operation is explicitly selected for a diagnostic ablation.
    public static class PoseWriteProbe
    {
        [Serializable] public class Site
        {
            public int index, calls, skippedCalls;
            public string method, operation, targetPath;
            public string[] descendantBodies, targetAliases;
        }
        private static readonly List<Site> sites = new List<Site>();
        // Names follow the checked IL call-site order. Runtime object identity
        // alone is ambiguous: center and hammerMeshHub alias the same Transform.
        private static readonly string[] targetFields = { "lookTarget", "lookTarget", "dudeMeshHub", "potMeshHub",
            "leftHand", "rightHand", "leftHandGripCenter", "rightHandGripCenter", "anim", "hammerMeshHub",
            "leftHand", "rightHand", "leftElbow", "rightElbow", "leftHand", "rightHand",
            "leftElbow", "rightElbow", "leftHand", "rightHand", "leftHand", "rightHand" };
        private static DynamicsProbe.Fixture fixture;
        private static bool finished;

        public static void Configure(DynamicsProbe.Fixture settings, Harmony harmony)
        {
            if (!settings.poseWriteAudit && (settings.skipPoseOperations == null || settings.skipPoseOperations.Length == 0)) return;
            fixture = settings;
            finished = false; sites.Clear();
            harmony.Patch(AccessTools.Method(AccessTools.TypeByName("PoseControl"), "LateUpdate"),
                transpiler: new HarmonyMethod(typeof(PoseWriteProbe), nameof(Transpile)));
        }

        public static IEnumerable<CodeInstruction> Transpile(IEnumerable<CodeInstruction> instructions)
        {
            // Harmony can rerun transpilers when another patch is added.
            sites.Clear();
            int position = 0, localPosition = 0, rotation = 0, animator = 0;
            foreach (var instruction in instructions)
            {
                var method = instruction.operand as MethodInfo;
                string wrapper = null;
                if (method != null && (instruction.opcode == OpCodes.Callvirt || instruction.opcode == OpCodes.Call))
                {
                    if (method.DeclaringType == typeof(Transform))
                    {
                        if (method.Name == "set_position") { wrapper = nameof(Position); position++; }
                        else if (method.Name == "set_localPosition") { wrapper = nameof(LocalPosition); localPosition++; }
                        else if (method.Name == "set_rotation") { wrapper = nameof(Rotation); rotation++; }
                    }
                    else if (method.DeclaringType == typeof(Animator) && method.Name == "Update")
                    { wrapper = nameof(AnimationUpdate); animator++; }
                }
                if (wrapper == null) { yield return instruction; continue; }
                int index = sites.Count;
                sites.Add(new Site { index = index, method = method.DeclaringType.Name + "." + method.Name });
                var owner = new CodeInstruction(OpCodes.Ldarg_0);
                owner.labels.AddRange(instruction.labels);
                owner.blocks.AddRange(instruction.blocks);
                yield return owner;
                yield return new CodeInstruction(OpCodes.Ldc_I4, index);
                yield return new CodeInstruction(OpCodes.Call, AccessTools.Method(typeof(PoseWriteProbe), wrapper));
            }
            if (position != 9 || localPosition != 4 || rotation != 8 || animator != 1)
                throw new InvalidOperationException("Unexpected PoseControl.LateUpdate call-site layout: "
                    + position + "/" + localPosition + "/" + rotation + "/" + animator);
        }

        private static string PathOf(Transform target)
        {
            string path = target.name;
            for (var parent = target.parent; parent != null; parent = parent.parent) path = parent.name + "/" + path;
            return path;
        }

        private static string[] Aliases(object owner, UnityEngine.Object target)
        {
            var names = new List<string>();
            foreach (var field in owner.GetType().GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic))
                if (typeof(UnityEngine.Object).IsAssignableFrom(field.FieldType) && (UnityEngine.Object)field.GetValue(owner) == target)
                    names.Add(field.Name);
            return names.ToArray();
        }

        private static bool Begin(Component target, object owner, int index, string property, float[] value)
        {
            // Match the lifecycle audit's scene boundary. Native Awake can call
            // LateUpdate before sceneLoaded; preserve those calls unchanged.
            if (finished || !NativeTimingProbe.IsRecording(owner)) return false;
            var site = sites[index];
            if (site.operation == null)
            {
                string fieldName = targetFields[index];
                if ((UnityEngine.Object)AccessTools.Field(owner.GetType(), fieldName).GetValue(owner) != target)
                    throw new InvalidOperationException("Pose call site disagrees with IL target: " + index);
                site.operation = fieldName + "." + property;
                site.targetAliases = Aliases(owner, target);
                site.targetPath = PathOf(target.transform);
                var descendants = new List<string>();
                foreach (var rb in target.GetComponentsInChildren<Rigidbody2D>(true)) descendants.Add(PathOf(rb.transform));
                site.descendantBodies = descendants.ToArray();
            }
            bool skip = fixture.skipPoseOperations != null && Array.IndexOf(fixture.skipPoseOperations, site.operation) >= 0;
            site.calls++; if (skip) site.skippedCalls++;
            if (fixture.poseWriteAudit)
                NativeTimingProbe.CaptureOperation("before_PoseWrite." + index + "." + site.operation, skip, site.targetPath, value);
            return skip;
        }

        private static void End(int index, bool skip, float[] value)
        {
            if (finished || !fixture.poseWriteAudit) return;
            var site = sites[index];
            if (site.operation == null) return;
            NativeTimingProbe.CaptureOperation("after_PoseWrite." + index + "." + site.operation, skip, site.targetPath, value);
        }

        public static void Position(Transform target, Vector3 value, object owner, int index)
        {
            var values = fixture.poseWriteAudit ? new[] { value.x, value.y, value.z } : null;
            bool skip = Begin(target, owner, index, "position", values);
            if (!skip) target.position = value;
            End(index, skip, values);
        }
        public static void LocalPosition(Transform target, Vector3 value, object owner, int index)
        {
            var values = fixture.poseWriteAudit ? new[] { value.x, value.y, value.z } : null;
            bool skip = Begin(target, owner, index, "localPosition", values);
            if (!skip) target.localPosition = value;
            End(index, skip, values);
        }
        public static void Rotation(Transform target, Quaternion value, object owner, int index)
        {
            var values = fixture.poseWriteAudit ? new[] { value.x, value.y, value.z, value.w } : null;
            bool skip = Begin(target, owner, index, "rotation", values);
            if (!skip) target.rotation = value;
            End(index, skip, values);
        }
        public static void AnimationUpdate(Animator target, float value, object owner, int index)
        {
            var values = fixture.poseWriteAudit ? new[] { value } : null;
            bool skip = Begin(target, owner, index, "Update", values);
            if (!skip) target.Update(value);
            End(index, skip, values);
        }

        public static void Finish()
        {
            if (fixture == null) return;
            finished = true;
            var records = new List<string>();
            foreach (var site in sites)
            {
                string record = JsonUtility.ToJson(site, true);
                if (record == "{}") throw new InvalidOperationException("Pose manifest serialization failed");
                records.Add(record);
            }
            File.WriteAllText(fixture.output + ".pose_sites.json", "[" + string.Join(",", records.ToArray()) + "]");
        }
    }
}
