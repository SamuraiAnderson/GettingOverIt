using System;
using System.Collections.Generic;
using System.Reflection;
using System.Reflection.Emit;
using HarmonyLib;
using UnityEngine;

namespace GoiRuntime.PlayerControl
{
    // PoseControl uses a physics parent (Player/Hub) as its visual depth hub.
    // Updating its world Z can resynchronize descendant XY into Physics2D.
    // Suppress only that setter, retaining all controller and animation work.
    public static class HubDepthWritePatch
    {
        private static readonly HashSet<object> initializing = new HashSet<object>();
        private static bool installed;

        public static void Apply()
        {
            if (installed) return;
            var pose = AccessTools.TypeByName("PoseControl");
            if (pose == null) throw new InvalidOperationException("Hub depth fix: PoseControl missing");
            var awake = AccessTools.DeclaredMethod(pose, "Awake");
            var late = AccessTools.DeclaredMethod(pose, "LateUpdate");
            if (awake == null || late == null) throw new InvalidOperationException("Hub depth fix: native lifecycle missing");
            var harmony = new Harmony("com.symbol.goi.hub.depth.write");
            harmony.Patch(awake,
                prefix: new HarmonyMethod(typeof(HubDepthWritePatch), nameof(BeforeAwake)),
                finalizer: new HarmonyMethod(typeof(HubDepthWritePatch), nameof(AfterAwake)));
            harmony.Patch(late, transpiler: new HarmonyMethod(typeof(HubDepthWritePatch), nameof(Transpile)));
            installed = true;
        }

        // Awake calls LateUpdate twice to establish the native hierarchy. Keep
        // those writes for every fresh scene and clone, including exceptional exits.
        public static void BeforeAwake(object __instance, out bool __state)
        { __state = initializing.Add(__instance); }

        public static Exception AfterAwake(object __instance, bool __state, Exception __exception)
        {
            if (__state) initializing.Remove(__instance);
            return __exception;
        }

        public static void WriteDepth(Transform target, Vector3 value, object owner)
        {
            if (initializing.Contains(owner)) target.position = value;
        }

        private static bool Field(CodeInstruction instruction, Type owner, string name)
        {
            var field = instruction.operand as FieldInfo;
            return instruction.opcode == OpCodes.Ldfld && field != null
                && field.DeclaringType == owner && field.Name == name;
        }

        public static IEnumerable<CodeInstruction> Transpile(IEnumerable<CodeInstruction> instructions, MethodBase __originalMethod)
        {
            var code = new List<CodeInstruction>(instructions);
            var pose = __originalMethod.DeclaringType;
            var getPosition = AccessTools.PropertyGetter(typeof(Transform), "position");
            var setPosition = AccessTools.PropertySetter(typeof(Transform), "position");
            var vector = AccessTools.Constructor(typeof(Vector3), new[] { typeof(float), typeof(float), typeof(float) });
            int match = -1, count = 0;
            for (int i = 0; i + 15 < code.Count; i++)
            {
                // Exact expression: hub.position = Vector3(hub.position.x,
                // hub.position.y, rightHandGripCenter.position.z).
                if (code[i].opcode != OpCodes.Ldarg_0 || !Field(code[i + 1], pose, "hammerMeshHub")
                    || code[i + 2].opcode != OpCodes.Ldarg_0 || !Field(code[i + 3], pose, "hammerMeshHub")
                    || !code[i + 4].Calls(getPosition) || !Field(code[i + 5], typeof(Vector3), "x")
                    || code[i + 6].opcode != OpCodes.Ldarg_0 || !Field(code[i + 7], pose, "hammerMeshHub")
                    || !code[i + 8].Calls(getPosition) || !Field(code[i + 9], typeof(Vector3), "y")
                    || code[i + 10].opcode != OpCodes.Ldarg_0 || !Field(code[i + 11], pose, "rightHandGripCenter")
                    || !code[i + 12].Calls(getPosition) || !Field(code[i + 13], typeof(Vector3), "z")
                    || code[i + 14].opcode != OpCodes.Newobj || !Equals(code[i + 14].operand, vector)
                    || !code[i + 15].Calls(setPosition)) continue;
                match = i + 15; count++;
            }
            if (count != 1) throw new InvalidOperationException("Hub depth fix: expected one native depth write, found " + count);
            var original = code[match];
            var owner = new CodeInstruction(OpCodes.Ldarg_0);
            owner.labels.AddRange(original.labels);
            owner.blocks.AddRange(original.blocks);
            code[match] = owner;
            code.Insert(match + 1, new CodeInstruction(OpCodes.Call,
                AccessTools.Method(typeof(HubDepthWritePatch), nameof(WriteDepth))));
            return code;
        }
    }
}
