using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using System.Globalization;
using System.Text;
using UnityEngine;
using GoiRuntime.PlayerControl;

namespace GoiRuntime.Testing
{
    // Optional, read-only diagnostic response. Never changes the 33D state packet.
    [Serializable]
    public class ObservationDiagnosticPacket
    {
        public int version = 2;
        public bool aligned;
        public int sequence;
        public float fixedDeltaTime;
        public int stepFrames;
        public ObservationDiagnosticAgent[] agents;
    }

    [Serializable]
    public class ObservationDiagnosticAgent
    {
        public float[] control = new float[6];
        public bool[] controlKnown = new bool[6];
        // [tip callback contact, tip callback grip, PlayerSounds.isTouching]
        public float[] contact = new float[3];
        public bool[] contactKnown = new bool[3];
        // Direct physics query by collider part: [tip, body, pot, pot_sides].
        public float[] physicsContact = new float[4];
        public bool[] physicsKnown = new bool[4];
        // World x, y, z-angle, scale x, scale y for each corresponding part.
        public float[] partPose = new float[20];
    }

    public static class ObservationDiagnostics
    {
        private const BindingFlags Flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

        private static string Number(float value)
        {
            return float.IsNaN(value) || float.IsInfinity(value) ? "0" : value.ToString("R", CultureInfo.InvariantCulture);
        }

        public static string ToJson(ObservationDiagnosticPacket packet)
        {
            // Unity's runtime serializer omits arrays of custom plugin-defined classes.
            // This fixed primitive-only schema needs no general-purpose JSON dependency.
            var json = new StringBuilder("{\"version\":2,\"fixedDeltaTime\":");
            json.Append(Number(packet.fixedDeltaTime)).Append(",\"stepFrames\":").Append(packet.stepFrames);
            json.Append(",\"aligned\":").Append(packet.aligned ? "true" : "false");
            json.Append(",\"sequence\":").Append(packet.sequence);
            json.Append(",\"agents\":[");
            for (int i = 0; i < packet.agents.Length; i++)
            {
                if (i > 0) json.Append(',');
                var agent = packet.agents[i];
                json.Append("{\"control\":[").Append(string.Join(",", Array.ConvertAll(agent.control, Number)));
                json.Append("],\"controlKnown\":[").Append(string.Join(",", Array.ConvertAll(agent.controlKnown, b => b ? "true" : "false")));
                json.Append("],\"contact\":[").Append(string.Join(",", Array.ConvertAll(agent.contact, Number)));
                json.Append("],\"contactKnown\":[").Append(string.Join(",", Array.ConvertAll(agent.contactKnown, b => b ? "true" : "false")));
                json.Append("],\"physicsContact\":[").Append(string.Join(",", Array.ConvertAll(agent.physicsContact, Number)));
                json.Append("],\"physicsKnown\":[").Append(string.Join(",", Array.ConvertAll(agent.physicsKnown, b => b ? "true" : "false")));
                json.Append("],\"partPose\":[").Append(string.Join(",", Array.ConvertAll(agent.partPose, Number)));
                json.Append("]}");
            }
            return json.Append("]}").ToString();
        }

        private static object Read(Component component, string field)
        {
            if (component == null) return null;
            try { return component.GetType().GetField(field, Flags)?.GetValue(component); }
            catch { return null; }
        }

        public static ObservationDiagnosticAgent Capture(PlayerInputService input)
        {
            var result = new ObservationDiagnosticAgent();
            var root = input?.GetPlayerObject();
            if (root == null) return result;
            Component player = null, hammer = null, sounds = null;
            foreach (var component in root.GetComponentsInChildren<Component>(true))
            {
                if (component == null) continue;
                switch (component.GetType().Name)
                {
                    case "PlayerControl": player = component; break;
                    case "HammerCollisions": hammer = component; break;
                    case "PlayerSounds": sounds = component; break;
                }
            }
            object oldMouse = Read(player, "oldMouse");
            if (oldMouse is Vector2)
            {
                var mouse = (Vector2)oldMouse;
                result.control[0] = mouse.x; result.control[1] = mouse.y;
                result.controlKnown[0] = !float.IsNaN(mouse.x) && !float.IsInfinity(mouse.x);
                result.controlKnown[1] = !float.IsNaN(mouse.y) && !float.IsInfinity(mouse.y);
            }
            string[] names = { "mouseVelocityAverage", "mouseSnap", "inputsToSkip", "pauseInputTimer" };
            for (int i = 0; i < names.Length; i++)
            {
                object value = Read(player, names[i]);
                if (value == null) continue;
                try
                {
                    result.control[i + 2] = Convert.ToSingle(value);
                    result.controlKnown[i + 2] = !float.IsNaN(result.control[i + 2]) && !float.IsInfinity(result.control[i + 2]);
                }
                catch { }
            }
            var points = Read(hammer, "collisionPoints") as IDictionary;
            int terrainLayer = LayerMask.NameToLayer("Terrain");
            if (terrainLayer >= 0)
            {
                foreach (var collider in root.GetComponentsInChildren<Collider2D>(true))
                {
                    int part = collider.gameObject == root ? 1 :
                        collider.name == "Tip" ? 0 : collider.name == "PotCollider" ? 2 :
                        collider.name == "Sides" ? 3 : -1;
                    if (part < 0 || collider.isTrigger) continue;
                    result.physicsKnown[part] = true;
                    if (collider.enabled && collider.gameObject.activeInHierarchy && collider.IsTouchingLayers(1 << terrainLayer))
                        result.physicsContact[part] = 1f;
                    var transform = collider.transform;
                    result.partPose[part * 5] = transform.position.x;
                    result.partPose[part * 5 + 1] = transform.position.y;
                    result.partPose[part * 5 + 2] = transform.eulerAngles.z;
                    result.partPose[part * 5 + 3] = transform.lossyScale.x;
                    result.partPose[part * 5 + 4] = transform.lossyScale.y;
                }
            }
            if (points != null && terrainLayer >= 0)
            {
                result.contactKnown[0] = true;
                foreach (DictionaryEntry entry in points)
                {
                    var collider = entry.Key as Collider2D;
                    var contacts = entry.Value as ContactPoint2D[];
                    if (collider != null && collider.gameObject.layer == terrainLayer && contacts != null && contacts.Length > 0)
                        result.contact[0] = 1f;
                }
                object slide = Read(hammer, "slide");
                if (slide is bool)
                {
                    result.contactKnown[1] = true;
                    result.contact[1] = result.contact[0] > 0 && !(bool)slide ? 1f : 0f;
                }
            }
            object touching = Read(sounds, "isTouching");
            if (touching is bool)
            {
                result.contactKnown[2] = true;
                result.contact[2] = (bool)touching ? 1f : 0f;
            }
            return result;
        }
    }
}
