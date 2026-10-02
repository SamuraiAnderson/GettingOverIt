// Managed-only checks for the exact IL rewrite and nested/exceptional Awake guard.
// Arguments: isolated game directory, directory containing the built plugin.
using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using GoiRuntime.PlayerControl;
using HarmonyLib;
using UnityEngine;

public class HubDepthPatchChecks
{
    public class PoseFixture
    {
        public Transform hammerMeshHub, rightHandGripCenter, other;
        public void LateUpdate() { }
    }

    public static int Main(string[] args)
    {
        AppDomain.CurrentDomain.AssemblyResolve += delegate(object sender, ResolveEventArgs e)
        {
            string name = new AssemblyName(e.Name).Name + ".dll";
            foreach (string folder in new[] { args[1], Path.Combine(args[0], "BepInEx/core"),
                                             Path.Combine(args[0], "GettingOverIt_Data/Managed") })
            {
                string path = Path.Combine(folder, name);
                if (File.Exists(path)) return Assembly.LoadFrom(path);
            }
            return null;
        };
        try { Check(); Console.WriteLine("PASS: exact rewrite, ambiguity rejection, metadata and Awake lifecycle"); return 0; }
        catch (Exception e) { Console.Error.WriteLine(e); return 1; }
    }

    static void Require(bool condition, string message)
    { if (!condition) throw new Exception(message); }

    static CodeInstruction Field(Type type, string name)
    { return new CodeInstruction(OpCodes.Ldfld, type.GetField(name)); }

    static List<CodeInstruction> Expression(string hub)
    {
        var code = new List<CodeInstruction>();
        code.Add(new CodeInstruction(OpCodes.Ldarg_0)); code.Add(Field(typeof(PoseFixture), hub));
        foreach (string axis in new[] { "x", "y", "z" })
        {
            code.Add(new CodeInstruction(OpCodes.Ldarg_0));
            code.Add(Field(typeof(PoseFixture), axis == "z" ? "rightHandGripCenter" : hub));
            code.Add(new CodeInstruction(OpCodes.Callvirt, AccessTools.PropertyGetter(typeof(Transform), "position")));
            code.Add(Field(typeof(Vector3), axis));
        }
        code.Add(new CodeInstruction(OpCodes.Newobj, typeof(Vector3).GetConstructor(new[] { typeof(float), typeof(float), typeof(float) })));
        code.Add(new CodeInstruction(OpCodes.Callvirt, AccessTools.PropertySetter(typeof(Transform), "position")));
        return code;
    }

    static List<CodeInstruction> Rewrite(List<CodeInstruction> code)
    { return new List<CodeInstruction>(HubDepthWritePatch.Transpile(code, typeof(PoseFixture).GetMethod("LateUpdate"))); }

    static void Reject(List<CodeInstruction> code)
    {
        try { Rewrite(code); }
        catch (InvalidOperationException) { return; }
        throw new Exception("Missing or ambiguous depth write was accepted");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    static void Check()
    {
        var original = Expression("hammerMeshHub");
        original.AddRange(Expression("other")); // An unrelated setter must survive.
        var label = new DynamicMethod("labels", typeof(void), Type.EmptyTypes).GetILGenerator().DefineLabel();
        original[15].labels.Add(label);
        original[15].blocks.Add(new ExceptionBlock(ExceptionBlockType.BeginExceptionBlock));
        var rewritten = Rewrite(original);
        Require(rewritten.Count == original.Count + 1, "Unexpected instruction count");
        for (int i = 0; i < original.Count; i++)
            if (i != 15) Require(ReferenceEquals(original[i], rewritten[i < 15 ? i : i + 1]), "Unrelated instruction changed");
        Require(rewritten[15].opcode == OpCodes.Ldarg_0 && rewritten[16].Calls(
            typeof(HubDepthWritePatch).GetMethod("WriteDepth")), "Wrong replacement");
        Require(rewritten[15].labels.Contains(label) && rewritten[15].blocks.Count == 1, "Lost IL metadata");
        Reject(Expression("other"));
        var duplicate = Expression("hammerMeshHub"); duplicate.AddRange(Expression("hammerMeshHub")); Reject(duplicate);
        var wrongAxis = Expression("hammerMeshHub"); wrongAxis[13] = Field(typeof(Vector3), "x"); Reject(wrongAxis);

        var active = (HashSet<object>)typeof(HubDepthWritePatch).GetField("initializing", BindingFlags.NonPublic | BindingFlags.Static).GetValue(null);
        var owner = new object(); var clone = new object(); bool outer, inner, cloneState;
        HubDepthWritePatch.BeforeAwake(owner, out outer);
        HubDepthWritePatch.BeforeAwake(owner, out inner);
        HubDepthWritePatch.BeforeAwake(clone, out cloneState);
        Require(outer && !inner && cloneState && active.Count == 2, "Awake nesting or clone guard failed");
        HubDepthWritePatch.AfterAwake(owner, inner, null);
        Require(active.Contains(owner), "Inner Awake removed outer guard");
        var error = new Exception("native Awake failure");
        Require(ReferenceEquals(error, HubDepthWritePatch.AfterAwake(owner, outer, error)), "Native exception swallowed");
        HubDepthWritePatch.AfterAwake(clone, cloneState, null);
        Require(active.Count == 0, "Awake guard leaked");
    }
}
