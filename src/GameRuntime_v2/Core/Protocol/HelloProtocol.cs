using System;
using System.Globalization;
using System.Text;

namespace GoiRuntime.Core.Protocol
{
	/// <summary>
	/// Immutable HELLO description prepared on the Unity main thread.
	/// The TCP background thread may only read this snapshot.
	/// </summary>
	public sealed class HelloSnapshot
	{
		public string ProtocolVersion = "v2";
		public string StateSchemaId = "anchor35";
		public int RawDim = 35;
		public string FieldHash = "goi-state-v1-35";
		public string AnchorId = "";
		public string CalibrationSha256 = "";
		public bool CalibrationAvailable;
		public int NumAgents;
        public string PhysicsBackend = "isolated_worlds";
        public int PhysicsWorldCount;
		public float FixedDeltaTime;
		public int StepFrames;
	}

	[Serializable]
	public sealed class HelloRequest
	{
		public string protocol;
		public string state_schema;
		public string anchor_id;
		public string calibration_sha256;
		public bool require_calibration;
	}

	public static class HelloProtocol
	{
		public const byte Marker = 0xFE;
		public const byte V2State = 0x01;
		public const byte V2Error = 0x02;
		public const byte V2Diagnostic = 0x03;

		public static bool TryAccept(HelloRequest request, HelloSnapshot snapshot, out string rejectReason)
		{
			rejectReason = null;
			if (request == null)
			{
				rejectReason = "hello_request_missing";
				return false;
			}
			if (request.protocol != "v2")
			{
				rejectReason = "unknown_protocol";
				return false;
			}
			if (request.state_schema != "anchor35")
			{
				rejectReason = "unknown_schema";
				return false;
			}
			if (request.require_calibration && !snapshot.CalibrationAvailable)
			{
				rejectReason = "calibration_unavailable";
				return false;
			}
			if (!string.IsNullOrEmpty(request.calibration_sha256) &&
				request.calibration_sha256 != snapshot.CalibrationSha256)
			{
				rejectReason = "calibration_hash_mismatch";
				return false;
			}
			if (!string.IsNullOrEmpty(request.anchor_id) &&
				!string.IsNullOrEmpty(snapshot.AnchorId) &&
				request.anchor_id != snapshot.AnchorId)
			{
				rejectReason = "anchor_id_mismatch";
				return false;
			}
			return true;
		}

		public static string ToJson(HelloSnapshot snapshot, bool accepted, string rejectReason)
		{
			var json = new StringBuilder();
			json.Append("{\"accepted\":").Append(accepted ? "true" : "false");
			json.Append(",\"protocol\":\"").Append(Escape(snapshot.ProtocolVersion)).Append('"');
			json.Append(",\"state_schema\":\"").Append(Escape(snapshot.StateSchemaId)).Append('"');
			json.Append(",\"raw_dim\":").Append(snapshot.RawDim);
			json.Append(",\"field_hash\":\"").Append(Escape(snapshot.FieldHash)).Append('"');
			json.Append(",\"anchor_id\":\"").Append(Escape(snapshot.AnchorId ?? "")).Append('"');
			json.Append(",\"calibration_sha256\":\"").Append(Escape(snapshot.CalibrationSha256 ?? "")).Append('"');
			json.Append(",\"calibration_available\":").Append(snapshot.CalibrationAvailable ? "true" : "false");
			json.Append(",\"num_agents\":").Append(snapshot.NumAgents);
            json.Append(",\"physics_backend\":\"").Append(Escape(snapshot.PhysicsBackend)).Append('"');
            json.Append(",\"physics_world_count\":").Append(snapshot.PhysicsWorldCount);
			json.Append(",\"fixed_delta_time\":").Append(snapshot.FixedDeltaTime.ToString("R", CultureInfo.InvariantCulture));
			json.Append(",\"step_frames\":").Append(snapshot.StepFrames);
			if (!string.IsNullOrEmpty(rejectReason))
				json.Append(",\"reject_reason\":\"").Append(Escape(rejectReason)).Append('"');
			return json.Append('}').ToString();
		}

		private static string Escape(string value)
		{
			if (string.IsNullOrEmpty(value)) return "";
			return value.Replace("\\", "\\\\").Replace("\"", "\\\"");
		}
	}
}
