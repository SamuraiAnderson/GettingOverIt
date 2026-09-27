using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using UnityEngine;
using GoiRuntime.Core.Utilities;

namespace GoiRuntime.Core.Configuration
{
	[Serializable]
	public sealed class AnchorCalibrationFile
	{
		public string anchor_id = "";
		public float local_x;
		public float local_y;
		public string body_frame = "Player";
		public string units = "game_world_m";
		public string status = "unverified";
		public string sha256 = "";
	}

	/// <summary>
	/// Read-only calibration loaded once on the main thread. Unverified files
	/// are kept for diagnostics and never advertised as calibrated.
	/// </summary>
	public sealed class AnchorCalibration
	{
		public bool FilePresent;
		public bool Available;
		public string AnchorId = "";
		public string Sha256 = "";
		public string Status = "missing";
		public float LocalX;
		public float LocalY;
		public string BodyFrame = "Player";

		public static string FilePath
		{
			get { return Path.Combine(PathManager.DataRootPath, "anchor_calibration.json"); }
		}

		public static AnchorCalibration Load()
		{
			var result = new AnchorCalibration();
			if (!File.Exists(FilePath))
				return result;
			try
			{
				string json = File.ReadAllText(FilePath);
				var parsed = JsonUtility.FromJson<AnchorCalibrationFile>(json);
				if (parsed == null)
					return result;
				result.FilePresent = true;
				result.AnchorId = parsed.anchor_id ?? "";
				result.LocalX = parsed.local_x;
				result.LocalY = parsed.local_y;
				result.BodyFrame = string.IsNullOrEmpty(parsed.body_frame) ? "Player" : parsed.body_frame;
				result.Status = string.IsNullOrEmpty(parsed.status) ? "unverified" : parsed.status;
				result.Sha256 = !string.IsNullOrEmpty(parsed.sha256) ? parsed.sha256 : Sha256Hex(json);
				result.Available = result.Status == "verified"
					&& result.BodyFrame == "Player"
					&& parsed.units == "game_world_m"
					&& IsFinite(result.LocalX) && IsFinite(result.LocalY);
			}
			catch (Exception e)
			{
				Debug.LogError("[AnchorCalibration] load failed: " + e.Message);
			}
			return result;
		}

		private static bool IsFinite(float value)
		{
			return !float.IsNaN(value) && !float.IsInfinity(value);
		}

		private static string Sha256Hex(string text)
		{
				using (var sha = new SHA256Managed())
			{
				byte[] bytes = sha.ComputeHash(Encoding.UTF8.GetBytes(text));
				var sb = new StringBuilder(bytes.Length * 2);
				for (int i = 0; i < bytes.Length; i++)
					sb.Append(bytes[i].ToString("x2"));
				return sb.ToString();
			}
		}
	}
}
