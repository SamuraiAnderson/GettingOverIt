using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using BepInEx;
using HarmonyLib;
using UnityEngine;
using GoiRuntime.Core;

[BepInPlugin("com.symbol.goi.callback_isolation_probe", "Callback isolation probe", "0.1.0")]
[BepInDependency("com.symbol.goi.runtime_v2")]
public class CallbackIsolationProbe : BaseUnityPlugin
{
    [Serializable] public class ProbeConfig { public string output; public string mode; }
    [Serializable] public class EventRow
    {
        public int step;
        public string method, instance;
        public bool skipped;
    }
    static ProbeConfig cfg;
    static int step;
    static string output;
    static readonly BindingFlags Flags = BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;

    void Awake()
    {
        string path = Path.Combine(Paths.GameRootPath, "GoiData/clone_probe.json");
        if (!File.Exists(path)) return;
        cfg = JsonUtility.FromJson<ProbeConfig>(File.ReadAllText(path));
        if (cfg == null || String.IsNullOrEmpty(cfg.output)) return;
        output = Path.Combine(Path.GetDirectoryName(cfg.output), "callbacks.jsonl");
        var harmony = new Harmony("com.symbol.goi.callback_isolation_probe");
        harmony.Patch(typeof(IsolatedWorldRuntime).GetMethod("ExecuteStep", Flags),
            new HarmonyMethod(typeof(CallbackIsolationProbe).GetMethod("BeforeStep", Flags)));
        var patched = new List<string>();
        foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
        {
            string name = assembly.GetName().Name;
            if (name != "Assembly-CSharp" && name != "Assembly-CSharp-firstpass" && name != "GameRuntime_v2") continue;
            foreach (var type in assembly.GetTypes())
            {
                if (!typeof(MonoBehaviour).IsAssignableFrom(type)) continue;
                foreach (var method in type.GetMethods(Flags | BindingFlags.DeclaredOnly))
                {
                    string m = method.Name;
                    if (m != "OnCollisionEnter2D" && m != "OnCollisionStay2D" && m != "OnCollisionExit2D" &&
                        m != "OnTriggerEnter2D" && m != "OnTriggerStay2D" && m != "OnTriggerExit2D") continue;
                    if (method.IsAbstract || method.GetMethodBody() == null) continue;
                    harmony.Patch(method, new HarmonyMethod(typeof(CallbackIsolationProbe).GetMethod("BeforeCallback", Flags)));
                    patched.Add(type.FullName + "." + m);
                }
            }
        }
        File.WriteAllLines(Path.Combine(Path.GetDirectoryName(output), "patched_callbacks.txt"), patched.ToArray());
        Logger.LogInfo("Callback probe patched " + patched.Count + " methods; mode=" + cfg.mode);
    }
    static void BeforeStep() { step++; }
    static bool BeforeCallback(MethodBase __originalMethod, object __instance)
    {
        bool skip = cfg.mode == "skip_callbacks31" && step >= 31;
        if (step >= 28 && step <= 34)
        {
            var component = __instance as Component;
            File.AppendAllText(output, JsonUtility.ToJson(new EventRow { step = step,
                method = __originalMethod.DeclaringType.FullName + "." + __originalMethod.Name,
                instance = component == null ? "none" : component.transform.root.name + "/" + component.name,
                skipped = skip }) + "\n");
        }
        return !skip;
    }
}
