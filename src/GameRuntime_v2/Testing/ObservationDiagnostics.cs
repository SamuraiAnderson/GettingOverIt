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
        public int version = 4;
        public bool aligned;
        public int sequence;
        public float fixedDeltaTime;
        public int stepFrames;
        public string screenshotPath = "";
        public int screenshotWidth;
        public int screenshotHeight;
        public string pixelOrigin = "image_top_left";
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
        public float bodyX, bodyY, bodyZ, bodyRz, bodySx, bodySy, bodySz;
        public bool bodyKnown;
        public float anchorX, anchorY;
        public bool anchorKnown;
        public float camPosX, camPosY, camPosZ, camOrthoSize, camPixelW, camPixelH;
        public bool camKnown;
        public float unityScreenBodyX, unityScreenBodyY;
        public float unityScreenAxisXX, unityScreenAxisXY;
        public float unityScreenAxisYX, unityScreenAxisYY;
        public bool unityScreenKnown;
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
            var json = new StringBuilder("{\"version\":4,\"fixedDeltaTime\":");
            json.Append(Number(packet.fixedDeltaTime)).Append(",\"stepFrames\":").Append(packet.stepFrames);
            json.Append(",\"aligned\":").Append(packet.aligned ? "true" : "false");
            json.Append(",\"sequence\":").Append(packet.sequence);
            json.Append(",\"screenshotPath\":\"").Append(Escape(packet.screenshotPath)).Append('"');
            json.Append(",\"screenshotWidth\":").Append(packet.screenshotWidth);
            json.Append(",\"screenshotHeight\":").Append(packet.screenshotHeight);
            json.Append(",\"pixelOrigin\":\"").Append(Escape(packet.pixelOrigin)).Append('"');
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
                json.Append("],\"bodyX\":").Append(Number(agent.bodyX)).Append(",\"bodyY\":").Append(Number(agent.bodyY));
                json.Append(",\"bodyZ\":").Append(Number(agent.bodyZ)).Append(",\"bodyRz\":").Append(Number(agent.bodyRz));
                json.Append(",\"bodySx\":").Append(Number(agent.bodySx)).Append(",\"bodySy\":").Append(Number(agent.bodySy));
                json.Append(",\"bodySz\":").Append(Number(agent.bodySz)).Append(",\"bodyKnown\":").Append(agent.bodyKnown ? "true" : "false");
                json.Append(",\"anchorX\":").Append(Number(agent.anchorX)).Append(",\"anchorY\":").Append(Number(agent.anchorY));
                json.Append(",\"anchorKnown\":").Append(agent.anchorKnown ? "true" : "false");
                json.Append(",\"camPosX\":").Append(Number(agent.camPosX)).Append(",\"camPosY\":").Append(Number(agent.camPosY));
                json.Append(",\"camPosZ\":").Append(Number(agent.camPosZ)).Append(",\"camOrthoSize\":").Append(Number(agent.camOrthoSize));
                json.Append(",\"camPixelW\":").Append(Number(agent.camPixelW)).Append(",\"camPixelH\":").Append(Number(agent.camPixelH));
                json.Append(",\"camKnown\":").Append(agent.camKnown ? "true" : "false");
                json.Append(",\"unityScreenBodyX\":").Append(Number(agent.unityScreenBodyX));
                json.Append(",\"unityScreenBodyY\":").Append(Number(agent.unityScreenBodyY));
                json.Append(",\"unityScreenAxisXX\":").Append(Number(agent.unityScreenAxisXX));
                json.Append(",\"unityScreenAxisXY\":").Append(Number(agent.unityScreenAxisXY));
                json.Append(",\"unityScreenAxisYX\":").Append(Number(agent.unityScreenAxisYX));
                json.Append(",\"unityScreenAxisYY\":").Append(Number(agent.unityScreenAxisYY));
                json.Append(",\"unityScreenKnown\":").Append(agent.unityScreenKnown ? "true" : "false");
                json.Append("}");
            }
            return json.Append("]}").ToString();
        }

        private static string Escape(string value)
        {
            if (string.IsNullOrEmpty(value)) return "";
            return value.Replace("\\", "\\\\").Replace("\"", "\\\"");
        }

        private static object Read(Component component, string field)
        {
            if (component == null) return null;
            try { return component.GetType().GetField(field, Flags)?.GetValue(component); }
            catch { return null; }
        }

        public static ObservationDiagnosticAgent Capture(PlayerInputService input)
        {
            return Capture(input, null, null, false, 0f, 0f);
        }

        public static ObservationDiagnosticAgent Capture(
            PlayerInputService input,
            PlayerStateService state,
            Camera camera,
            bool calibrationAvailable,
            float anchorLocalX,
            float anchorLocalY)
        {
            var result = new ObservationDiagnosticAgent();
            var root = input != null ? input.GetPlayerObject() : null;
            if (state != null)
            {
                Vector3 pos;
                float rotZ;
                Vector3 scale;
                result.bodyKnown = state.TryGetBodyPose(out pos, out rotZ, out scale);
                if (result.bodyKnown)
                {
                    result.bodyX = pos.x; result.bodyY = pos.y; result.bodyZ = pos.z;
                    result.bodyRz = rotZ;
                    result.bodySx = scale.x; result.bodySy = scale.y; result.bodySz = scale.z;
                }
                Vector2 anchor = Vector2.zero;
                result.anchorKnown = calibrationAvailable &&
                    state.TrySampleReferenceAnchor(anchorLocalX, anchorLocalY, out anchor);
                if (result.anchorKnown)
                {
                    result.anchorX = anchor.x;
                    result.anchorY = anchor.y;
                }
            }
            if (camera != null)
            {
                result.camKnown = true;
                result.camPosX = camera.transform.position.x;
                result.camPosY = camera.transform.position.y;
                result.camPosZ = camera.transform.position.z;
                result.camOrthoSize = camera.orthographic ? camera.orthographicSize : 0f;
                result.camPixelW = camera.pixelWidth;
                result.camPixelH = camera.pixelHeight;
                if (state != null)
                {
                    Transform body = state.GetPlayerTransform();
                    if (body != null)
                    {
                        Vector3 s0 = camera.WorldToScreenPoint(body.position);
                        Vector3 sx = camera.WorldToScreenPoint(body.TransformPoint(new Vector3(1f, 0f, 0f)));
                        Vector3 sy = camera.WorldToScreenPoint(body.TransformPoint(new Vector3(0f, 1f, 0f)));
                        result.unityScreenBodyX = s0.x;
                        result.unityScreenBodyY = s0.y;
                        result.unityScreenAxisXX = sx.x;
                        result.unityScreenAxisXY = sx.y;
                        result.unityScreenAxisYX = sy.x;
                        result.unityScreenAxisYY = sy.y;
                        result.unityScreenKnown = s0.z > 0f && !float.IsNaN(s0.x) && !float.IsInfinity(s0.x);
                    }
                }
            }
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
