using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using BepInEx;
using HarmonyLib;
using UnityEngine;
using GoiRuntime.Core;

[BepInPlugin("com.symbol.goi.clone_divergence_probe", "Clone divergence probe", "0.1.0")]
[BepInDependency("com.symbol.goi.runtime_v2")]
public class CloneDivergenceProbe : BaseUnityPlugin
{
    [Serializable] public class Config { public string output; public string mode; }
    [Serializable] public class ContactRow
    {
        public string collider, other;
        public int owner, otherOwner;
        public Vector2 point, normal, relativeVelocity;
        public float separation, normalImpulse, tangentImpulse;
        public bool enabled;
    }
    [Serializable] public class ColliderRow
    {
        public int agent, layer;
        public string name, body, detection;
        public Vector2 offset;
        public Vector3 scale;
        public bool enabled, trigger;
        public ContactRow[] contacts;
    }
    [Serializable] public class PairRow
    {
        public int a, b;
        public string ca, cb;
        public bool ignored, layerIgnored;
    }
    [Serializable] public class TerrainRow
    {
        public string name;
        public int paths;
        public Vector3 position, scale;
        public float[] worldPoints;
    }
    static Config cfg;
    static StepController current;
    static int step;
    static readonly BindingFlags Flags = BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;

    void Awake()
    {
        string path = Path.Combine(Paths.GameRootPath, "GoiData/clone_probe.json");
        if (!File.Exists(path)) return;
        cfg = JsonUtility.FromJson<Config>(File.ReadAllText(path));
        if (cfg == null || String.IsNullOrEmpty(cfg.output)) return;
        var harmony = new Harmony("com.symbol.goi.clone_divergence_probe");
        harmony.Patch(typeof(StepController).GetMethod("ExecuteStep", Flags),
            new HarmonyMethod(typeof(CloneDivergenceProbe).GetMethod("BeforeStep", Flags)),
            new HarmonyMethod(typeof(CloneDivergenceProbe).GetMethod("AfterStep", Flags)));
        harmony.Patch(typeof(StepController).GetMethod("SimulatePhysics", Flags),
            new HarmonyMethod(typeof(CloneDivergenceProbe).GetMethod("BeforeSimulation", Flags)),
            new HarmonyMethod(typeof(CloneDivergenceProbe).GetMethod("AfterSimulation", Flags)));
        Logger.LogInfo("Clone probe enabled: " + cfg.mode);
    }
    static bool Recording { get { return step == 1 || (step >= 28 && step <= 34); } }
    static void BeforeStep(StepController __instance)
    {
        current = __instance;
        step++;
        if (Recording) Capture("before_control");
        if (step == 31 && cfg.mode != "observe")
        {
            foreach (var root in current.GetVisualizationRoots())
                foreach (var col in root.GetComponentsInChildren<Collider2D>(true))
                    if (col.name == "Tip")
                    {
                        if (cfg.mode == "tip_discrete") col.attachedRigidbody.collisionDetectionMode = CollisionDetectionMode2D.Discrete;
                        string suffix = cfg.mode == "ignore_rock3" ? " (3)" : cfg.mode == "ignore_rock2" ? " (2)" : null;
                        if (suffix != null)
                            foreach (var rock in UnityEngine.Object.FindObjectsOfType<PolygonCollider2D>())
                                if (rock.name == "Intresto_rock055_252vert_500face_scale_1m_tex_sandstone1" + suffix)
                                    Physics2D.IgnoreCollision(col, rock, true);
                    }
            Capture("after_intervention");
        }
    }
    static void AfterStep() { current = null; }
    static void BeforeSimulation()
    {
        if (current == null) return;
        if (Recording) Capture("before_simulation");
    }
    static void AfterSimulation()
    {
        if (current != null && Recording) Capture("after_simulation");
    }
    static string PathOf(Collider2D col)
    {
        if (col == null) return "null";
        return col.transform.root.name + "/" + col.name;
    }
    static void Capture(string phase)
    {
        var roots = current.GetVisualizationRoots();
        var owners = new Dictionary<Collider2D, int>();
        var colliders = new List<Collider2D[]>();
        for (int i = 0; i < roots.Length; i++)
        {
            var cols = roots[i].GetComponentsInChildren<Collider2D>(true);
            colliders.Add(cols);
            foreach (var col in cols) owners[col] = i;
        }
        var rows = new List<string>();
        var pairs = new List<string>();
        var terrain = new List<string>();
        if (step == 1 && phase == "before_control")
            foreach (var rock in UnityEngine.Object.FindObjectsOfType<PolygonCollider2D>())
                if (rock.name == "Intresto_rock055_252vert_500face_scale_1m_tex_sandstone1 (2)" ||
                    rock.name == "Intresto_rock055_252vert_500face_scale_1m_tex_sandstone1 (3)")
                {
                    var points = new List<float>();
                    for (int path = 0; path < rock.pathCount; path++)
                        foreach (var p in rock.GetPath(path))
                        {
                            var world = rock.transform.TransformPoint(p + rock.offset);
                            points.Add(world.x); points.Add(world.y);
                        }
                    terrain.Add(JsonUtility.ToJson(new TerrainRow { name = rock.name, paths = rock.pathCount,
                        position = rock.transform.position, scale = rock.transform.lossyScale,
                        worldPoints = points.ToArray() }));
                }
        for (int i = 0; i < roots.Length; i++)
        {
            foreach (var col in colliders[i])
            {
                var contacts = new ContactPoint2D[128];
                int count = col.GetContacts(contacts);
                var cr = new ContactRow[count];
                for (int c = 0; c < count; c++)
                {
                    var p = contacts[c];
                    int owner, other;
                    owners.TryGetValue(p.collider, out owner);
                    owners.TryGetValue(p.otherCollider, out other);
                    cr[c] = new ContactRow { collider = PathOf(p.collider), other = PathOf(p.otherCollider),
                        owner = owners.ContainsKey(p.collider) ? owner : -1,
                        otherOwner = owners.ContainsKey(p.otherCollider) ? other : -1,
                        point = p.point, normal = p.normal, relativeVelocity = p.relativeVelocity,
                        separation = p.separation, normalImpulse = p.normalImpulse,
                        tangentImpulse = p.tangentImpulse, enabled = p.enabled };
                }
                var rb = col.attachedRigidbody;
                var row = new ColliderRow { agent = i, name = col.name, layer = col.gameObject.layer,
                    body = rb == null ? "none" : rb.name,
                    detection = rb == null ? "none" : rb.collisionDetectionMode.ToString(),
                    enabled = col.enabled, trigger = col.isTrigger,
                    offset = col.offset, scale = col.transform.lossyScale, contacts = cr };
                // Unity's serializer may omit nested arrays of plugin-defined classes.
                string header = JsonUtility.ToJson(row);
                var contactJson = new List<string>();
                foreach (var c in cr) contactJson.Add(JsonUtility.ToJson(c));
                rows.Add(header.Substring(0, header.Length - 1) + ",\"contacts\":[" + String.Join(",", contactJson.ToArray()) + "]}");
                if (step == 1 && phase == "before_control")
                    for (int j = i + 1; j < roots.Length; j++)
                        foreach (var other in colliders[j])
                            pairs.Add(JsonUtility.ToJson(new PairRow { a = i, b = j,
                                ca = col.name, cb = other.name,
                                ignored = Physics2D.GetIgnoreCollision(col, other),
                                layerIgnored = Physics2D.GetIgnoreLayerCollision(col.gameObject.layer, other.gameObject.layer) }));
            }
        }
        object frame = typeof(StepController).GetMethod("CaptureResetFrame", Flags).Invoke(current, new object[] { phase });
        string frameJson = (string)typeof(StepController).GetMethod("ResetFrameJson", Flags).Invoke(null, new object[] { frame });
        string line = "{\"step\":" + step + ",\"phase\":\"" + phase + "\",\"frame\":" + frameJson
            + ",\"colliders\":[" + String.Join(",", rows.ToArray()) + "]"
            + ",\"terrain\":[" + String.Join(",", terrain.ToArray()) + "]"
            + ",\"pairs\":[" + String.Join(",", pairs.ToArray()) + "]}";
        File.AppendAllText(cfg.output, line + "\n");
    }
}
