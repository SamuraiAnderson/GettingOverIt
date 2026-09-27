using System.IO;
using System.Text;
using UnityEngine;

namespace GoiRuntime.Testing
{
	/// <summary>
	/// Same-thread camera dump. Physics is not advanced. Image origin is top-left.
	/// </summary>
	public static class CalibrationCapture
	{
		public static bool WriteCameraPpm(Camera camera, string path, out int width, out int height)
		{
			width = 0;
			height = 0;
			if (camera == null) return false;
			width = camera.pixelWidth;
			height = camera.pixelHeight;
			if (width < 8 || height < 8) return false;

			RenderTexture rt = new RenderTexture(width, height, 24);
			RenderTexture previous = camera.targetTexture;
			camera.targetTexture = rt;
			camera.Render();
			RenderTexture.active = rt;
			Texture2D tex = new Texture2D(width, height, TextureFormat.RGB24, false);
			tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
			tex.Apply();
			camera.targetTexture = previous;
			RenderTexture.active = null;
			Color[] pixels = tex.GetPixels();
			Object.Destroy(tex);
			Object.Destroy(rt);

			using (FileStream stream = File.Create(path))
			{
				byte[] header = Encoding.ASCII.GetBytes("P6\n" + width + " " + height + "\n255\n");
				stream.Write(header, 0, header.Length);
				byte[] row = new byte[width * 3];
				for (int y = height - 1; y >= 0; y--)
				{
					int src = y * width;
					for (int x = 0; x < width; x++)
					{
						Color color = pixels[src + x];
						row[x * 3] = (byte)Mathf.Clamp(Mathf.RoundToInt(color.r * 255f), 0, 255);
						row[x * 3 + 1] = (byte)Mathf.Clamp(Mathf.RoundToInt(color.g * 255f), 0, 255);
						row[x * 3 + 2] = (byte)Mathf.Clamp(Mathf.RoundToInt(color.b * 255f), 0, 255);
					}
					stream.Write(row, 0, row.Length);
				}
			}
			return true;
		}
	}
}
