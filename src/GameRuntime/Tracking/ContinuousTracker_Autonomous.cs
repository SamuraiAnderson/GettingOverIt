using UnityEngine;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System;
using System.Linq;

namespace GoiHitboxLogger
{
	/// <summary>
	/// 简化的连续状态跟踪器 - 专注于数据采集
	/// 移除所有IPC相关功能，纯粹的数据采集器
	/// </summary>
	public static class ContinuousTrackerAutonomous
	{
		#region 测试模式支持
		
		/// <summary>
		/// 外部测试指令数据点
		/// </summary>
		public struct TestMouseCommand
		{
			public float timestamp;
			public float mouseX;
			public float mouseY;
		}
		
		// 测试模式相关变量
		private static bool testModeEnabled = false;
		private static List<TestMouseCommand> testCommands = new List<TestMouseCommand>();
		private static float testStartTime = 0f;
		private static int currentTestCommandIndex = 0;
		
		#endregion
		
		#region 数据结构
		
		/// <summary>
		/// 鼠标输入状态
		/// </summary>
		[Serializable]
		public struct MouseInput
		{
			public float mouseX;
			public float mouseY;
			public float timestamp;
			
			public Vector2 mousePosition => new Vector2(mouseX, mouseY);
		}

		/// <summary>
		/// 游戏物理状态
		/// </summary>
		[Serializable]
		public struct PhysicsState
		{
			public float playerX;
			public float playerY;
			public float velocityX;
			public float velocityY;
			public float hammerAngle;
			public float hammerAngularVel;
			public bool isGrounded;
			public float timestamp;
			
			public Vector2 playerPosition => new Vector2(playerX, playerY);
			public Vector2 velocity => new Vector2(velocityX, velocityY);
			public float speed => velocity.magnitude;
		}

		/// <summary>
		/// 跟踪数据点
		/// </summary>
		[Serializable]
		public struct TrackingDataPoint
		{
			public MouseInput mouseInput;
			public PhysicsState physicsState;
			public float deltaTime;
			
			// 计算属性
			public float mouseDeltaX;
			public float mouseDeltaY;
			public float mouseMoveDistance;
			public float mouseMoveSpeed;
			
			public float positionDelta;
			public float velocityDelta;
			public float angleDelta;
			public float physicsResponseDelay;
		}
		
		#endregion
		
		#region 配置和状态
		
		private static float trackingInterval = 1f / 30f; // 30Hz
		private static int maxDataPoints = 1000; // 最大数据点数
		
		private static List<TrackingDataPoint> trackingData = new List<TrackingDataPoint>();
		private static TrackingDataPoint lastDataPoint;
		private static float lastSampleTime = 0f;
		
		private static bool isTracking = false;
		private static bool isInitialized = false;
		
		// 组件引用
		private static GameObject player;
		private static Transform playerTransform;
		private static Rigidbody2D playerRigidbody;
		private static Transform hammerTip;
		private static HingeJoint2D mainHinge;
		private static PolygonCollider2D potCollider;
		
		#endregion
		
		#region 公共接口
		
		/// <summary>
		/// 初始化跟踪器
		/// </summary>
		public static bool Initialize()
		{
			Debug.Log("🚀🚀🚀 [TRACKER] 开始初始化连续跟踪器... 🚀🚀🚀");
			
			if (isInitialized) 
			{
				Debug.Log("🔄🔄🔄 [TRACKER] 已初始化，跳过重复初始化 🔄🔄🔄");
				return true;
			}
			
			Debug.Log("🔍🔍🔍 [TRACKER] 开始查找游戏组件... 🔍🔍🔍");
			bool findResult = FindGameComponents();
			Debug.Log($"🔍🔍🔍 [TRACKER] FindGameComponents 结果: {findResult} 🔍🔍🔍");
			
			if (!findResult)
			{
				Debug.LogError("❌❌❌ [TRACKER] 连续跟踪器初始化失败：未找到必要的游戏组件 ❌❌❌");
				return false;
			}
			
			Debug.Log("📂📂📂 [TRACKER] 组件查找成功，开始加载测试指令... 📂📂📂");
			
			// 尝试加载测试指令
			LoadTestCommands();
			
			isInitialized = true;
			Debug.Log($"✅✅✅ [TRACKER] 连续跟踪器初始化完成！testModeEnabled={testModeEnabled} ✅✅✅");
			return true;
		}
		
		/// <summary>
		/// 设置采样频率
		/// </summary>
		public static void SetSamplingRate(float frequency)
		{
			trackingInterval = 1f / Mathf.Clamp(frequency, 1f, 200f);
			Debug.Log($"📊 采样频率设置为: {frequency:F0}Hz (间隔{trackingInterval * 1000f:F1}ms)");
		}
		
		/// <summary>
		/// 开始跟踪
		/// </summary>
		public static void StartTracking()
		{
			if (!isInitialized)
			{
				Debug.LogError("❌ 连续跟踪器未初始化，无法开始跟踪");
				return;
			}
			
			// 重置测试模式状态
			if (testModeEnabled)
			{
				testStartTime = 0f; // 下次调用GetTestMousePosition时会重新设置
				currentTestCommandIndex = 0;
				Debug.Log("🧪 测试模式重置，准备回放鼠标指令");
			}
			
			isTracking = true;
			lastSampleTime = Time.time;
			Debug.Log("🚀 开始连续数据跟踪");
		}
		
		/// <summary>
		/// 停止跟踪
		/// </summary>
		public static void StopTracking()
		{
			isTracking = false;
			Debug.Log($"⏸️ 停止连续数据跟踪 - 已记录 {trackingData.Count} 个数据点");
		}
		
		/// <summary>
		/// 清空数据
		/// </summary>
		public static void ClearData()
		{
			trackingData.Clear();
			Debug.Log("🗑️ 已清空跟踪数据");
		}
		
		/// <summary>
		/// 获取数据点数量
		/// </summary>
		public static int GetDataPointCount()
		{
			return trackingData.Count;
		}
		
		/// <summary>
		/// 主要更新方法 - 由Unity每帧调用
		/// </summary>
		public static void Update()
		{
			if (!isInitialized || !isTracking) return;

			float currentTime = Time.time;
			
			if (currentTime - lastSampleTime >= trackingInterval)
			{
				// 采样新数据点
				TrackingDataPoint newDataPoint = SampleCurrentData();
				
				// 计算变化率
				if (trackingData.Count > 0)
				{
					CalculateDeltas(ref newDataPoint, lastDataPoint);
				}
				
				// 添加到数据集
				trackingData.Add(newDataPoint);
				lastDataPoint = newDataPoint;
				lastSampleTime = currentTime;
				
				// 维护数据点数量上限
				if (trackingData.Count > maxDataPoints)
				{
					trackingData.RemoveAt(0);
				}
			}
		}
		
		/// <summary>
		/// 导出跟踪数据到CSV
		/// </summary>
		public static string ExportTrackingData()
		{
			if (trackingData.Count == 0)
			{
				Debug.LogWarning("⚠️ 没有数据可导出");
				return null;
			}

			try
			{
				string fileName = $"ContinuousTracking_{DateTime.Now:yyyyMMdd_HHmmss}.csv";
				string basePath = Path.Combine(Application.dataPath, "..");
				string trackingDumpPath = Path.Combine(basePath, "TrackingDump");
				
				if (!Directory.Exists(trackingDumpPath))
				{
					Directory.CreateDirectory(trackingDumpPath);
				}
				
				string fullPath = Path.Combine(trackingDumpPath, fileName);
				
				StringBuilder csv = new StringBuilder();
				
				// CSV头部
				csv.AppendLine("timestamp,deltaTime," +
							  "mouseX,mouseY," +
							  "playerX,playerY,velocityX,velocityY,hammerAngle,hammerAngularVel,isGrounded," +
							  "mouseDeltaX,mouseDeltaY,mouseMoveDistance,mouseMoveSpeed," +
							  "positionDelta,velocityDelta,angleDelta,physicsResponseDelay");
				
				// 数据行
				foreach (var dataPoint in trackingData)
				{
					csv.AppendLine(
						$"{dataPoint.mouseInput.timestamp:F3},{dataPoint.deltaTime:F4}," +
						$"{dataPoint.mouseInput.mouseX:F2},{dataPoint.mouseInput.mouseY:F2}," +
						$"{dataPoint.physicsState.playerX:F3},{dataPoint.physicsState.playerY:F3}," +
						$"{dataPoint.physicsState.velocityX:F3},{dataPoint.physicsState.velocityY:F3}," +
						$"{dataPoint.physicsState.hammerAngle:F1},{dataPoint.physicsState.hammerAngularVel:F2}," +
						$"{(dataPoint.physicsState.isGrounded ? 1 : 0)}," +
						$"{dataPoint.mouseDeltaX:F2},{dataPoint.mouseDeltaY:F2}," +
						$"{dataPoint.mouseMoveDistance:F2},{dataPoint.mouseMoveSpeed:F2}," +
						$"{dataPoint.positionDelta:F3},{dataPoint.velocityDelta:F3}," +
						$"{dataPoint.angleDelta:F2},{dataPoint.physicsResponseDelay:F4}");
				}
				
				File.WriteAllText(fullPath, csv.ToString());
				
				Debug.Log($"📁 跟踪数据已导出: {fullPath}");
				Debug.Log($"📊 数据点数量: {trackingData.Count}");
				
				return fullPath;
			}
			catch (Exception e)
			{
				Debug.LogError($"❌ 导出跟踪数据失败: {e.Message}");
				return null;
			}
		}
		
		/// <summary>
		/// 获取数据摘要
		/// </summary>
		public static string GetDataSummary()
		{
			if (trackingData.Count == 0) return "无数据";
			
			var speeds = trackingData.Select(d => d.mouseMoveSpeed).Where(s => s > 0f);
			var angles = trackingData.Select(d => d.physicsState.hammerAngle);
			
			float minSpeed = speeds.Any() ? speeds.Min() : 0f;
			float maxSpeed = speeds.Any() ? speeds.Max() : 0f;
			float avgSpeed = speeds.Any() ? speeds.Average() : 0f;
			
			float minAngle = angles.Min();
			float maxAngle = angles.Max();
			
			return $"速度范围{minSpeed:F1}-{maxSpeed:F1}(平均{avgSpeed:F1}), " +
				   $"角度范围{minAngle:F0}°-{maxAngle:F0}°";
		}
		
		#endregion
		
		#region 内部方法
		
		/// <summary>
		/// 加载测试指令
		/// </summary>
		private static void LoadTestCommands()
		{
			Debug.Log("🧪🧪🧪 [TEST] 开始加载测试指令... 🧪🧪🧪");
			try
			{
				string basePath = Path.Combine(Application.dataPath, "..");
				string srcPath = Path.Combine(basePath, "src");
				string dataPath = Path.Combine(srcPath, "Data");
				string csvPath = Path.Combine(dataPath, "input_commands.csv");
				
				Debug.Log($"🗂️ 计算的CSV路径: {csvPath}");
				Debug.Log($"📁 Application.dataPath: {Application.dataPath}");
				Debug.Log($"📂 basePath: {basePath}");
				Debug.Log($"📂 srcPath: {srcPath}");
				Debug.Log($"📂 dataPath: {dataPath}");
				
				// 检查各级目录是否存在
				Debug.Log($"📋 basePath 存在: {Directory.Exists(basePath)}");
				Debug.Log($"📋 srcPath 存在: {Directory.Exists(srcPath)}");
				Debug.Log($"📋 dataPath 存在: {Directory.Exists(dataPath)}");
				
				if (!File.Exists(csvPath))
				{
					Debug.LogError($"❌ 测试指令文件不存在: {csvPath}");
					Debug.LogError("❌ 使用实时鼠标输入模式");
					testModeEnabled = false;
					return;
				}
				
				Debug.Log("✅ 测试指令文件存在，开始加载...");
				
				testCommands.Clear();
				string[] lines = File.ReadAllLines(csvPath);
				
				// 跳过标题行
				for (int i = 1; i < lines.Length; i++)
				{
					string[] parts = lines[i].Split(',');
					if (parts.Length >= 3)
					{
						TestMouseCommand cmd = new TestMouseCommand();
						if (float.TryParse(parts[0], out cmd.timestamp) &&
							float.TryParse(parts[1], out cmd.mouseX) &&
							float.TryParse(parts[2], out cmd.mouseY))
						{
							testCommands.Add(cmd);
						}
					}
				}
				
				if (testCommands.Count > 0)
				{
					testModeEnabled = true;
					currentTestCommandIndex = 0;
					Debug.Log($"🧪 测试模式已启用：加载了 {testCommands.Count} 个鼠标指令");
					Debug.Log($"📊 指令时长: {testCommands[testCommands.Count-1].timestamp:F1}秒");
				}
				else
				{
					Debug.LogWarning("⚠️ 测试指令文件为空，使用实时鼠标输入模式");
					testModeEnabled = false;
				}
			}
			catch (System.Exception e)
			{
				Debug.LogError($"❌ 加载测试指令失败: {e.Message}");
				Debug.LogError($"❌ 异常详细: {e}");
				Debug.LogError("❌ 使用实时鼠标输入模式");
				testModeEnabled = false;
			}
			
			Debug.Log($"🏁🏁🏁 [TEST] LoadTestCommands 完成，testModeEnabled = {testModeEnabled} 🏁🏁🏁");
		}
		
		/// <summary>
		/// 获取当前鼠标位置（考虑测试模式）
		/// </summary>
		public static Vector2 GetCurrentMousePosition()
		{
			return GetTestMousePosition(Time.time);
		}
		
		/// <summary>
		/// 尝试获取测试模式鼠标位置（用于Input hook）
		/// </summary>
		public static bool TryGetTestMousePosition(out Vector2 testPos)
		{
			if (testModeEnabled && testCommands.Count > 0)
			{
				testPos = GetTestMousePosition(Time.time);
				return true;
			}
			
			testPos = Vector2.zero;
			return false;
		}
		
		/// <summary>
		/// 获取测试模式鼠标位置
		/// </summary>
		private static Vector2 GetTestMousePosition(float currentTime)
		{
			if (!testModeEnabled || testCommands.Count == 0)
			{
				// 不在测试模式，返回实际鼠标位置
				Vector3 realPos = Input.mousePosition;
				if (testModeEnabled && testCommands.Count == 0)
				{
					Debug.LogWarning("⚠️ testModeEnabled=true 但 testCommands 为空");
				}
				// Debug.Log($"🔧 使用真实鼠标位置: ({realPos.x:F1}, {realPos.y:F1})");
				return new Vector2(realPos.x, realPos.y);
			}
			
			// 第一次调用时记录开始时间
			if (testStartTime == 0f)
			{
				testStartTime = currentTime;
				currentTestCommandIndex = 0;
				Debug.Log($"🚀 测试模式开始，基准时间: {currentTime:F3}");
			}
			
			float relativeTime = currentTime - testStartTime;
			
			// 在测试指令中查找合适的时间点
			while (currentTestCommandIndex < testCommands.Count - 1)
			{
				if (testCommands[currentTestCommandIndex + 1].timestamp > relativeTime)
				{
					break;
				}
				currentTestCommandIndex++;
			}
			
			// 如果超出测试指令范围，使用最后一个指令
			if (currentTestCommandIndex >= testCommands.Count)
			{
				currentTestCommandIndex = testCommands.Count - 1;
			}
			
			TestMouseCommand currentCmd = testCommands[currentTestCommandIndex];
			
			// 如果还有下一个指令，进行线性插值以平滑过渡
			if (currentTestCommandIndex < testCommands.Count - 1)
			{
				TestMouseCommand nextCmd = testCommands[currentTestCommandIndex + 1];
				float t = (relativeTime - currentCmd.timestamp) / (nextCmd.timestamp - currentCmd.timestamp);
				t = Mathf.Clamp01(t);
				
				float interpolatedX = Mathf.Lerp(currentCmd.mouseX, nextCmd.mouseX, t);
				float interpolatedY = Mathf.Lerp(currentCmd.mouseY, nextCmd.mouseY, t);
				
				Vector2 testPos = new Vector2(interpolatedX, interpolatedY);
				return testPos;
			}
			
			Vector2 finalPos = new Vector2(currentCmd.mouseX, currentCmd.mouseY);
			Debug.Log($"🎯 测试模式返回最终位置: ({finalPos.x:F1}, {finalPos.y:F1}) [索引{currentTestCommandIndex}]");
			return finalPos;
		}
		
		/// <summary>
		/// 查找游戏组件
		/// </summary>
		private static bool FindGameComponents()
		{
			Debug.Log("🎮🎮🎮 [TRACKER] 查找Player对象... 🎮🎮🎮");
			player = GameObject.Find("Player");
			if (player == null) 
			{
				Debug.LogError("❌❌❌ [TRACKER] 未找到Player对象！ ❌❌❌");
				return false;
			}
			Debug.Log("✅✅✅ [TRACKER] Player对象找到了！ ✅✅✅");
			
			playerTransform = player.transform;
			playerRigidbody = player.GetComponent<Rigidbody2D>();
			
			// 查找锤子尖端
			hammerTip = FindDeepChild(player.transform, "Tip");
			
			// 查找主要铰链
			mainHinge = player.GetComponent<HingeJoint2D>();
			
			// 查找锅的碰撞体
			Transform potColliderTransform = FindDeepChild(player.transform, "PotCollider");
			if (potColliderTransform != null)
			{
				potCollider = potColliderTransform.GetComponent<PolygonCollider2D>();
			}
			
			return playerRigidbody != null;
		}
		
		/// <summary>
		/// 深度查找子对象
		/// </summary>
		private static Transform FindDeepChild(Transform parent, string name)
		{
			Queue<Transform> queue = new Queue<Transform>();
			queue.Enqueue(parent);
			
			while (queue.Count > 0)
			{
				Transform current = queue.Dequeue();
				
				if (current.name == name)
				{
					return current;
				}
				
				foreach (Transform child in current)
				{
					queue.Enqueue(child);
				}
			}
			
			return null;
		}
		
		/// <summary>
		/// 采样当前数据
		/// </summary>
		private static TrackingDataPoint SampleCurrentData()
		{
			TrackingDataPoint dataPoint = new TrackingDataPoint();
			float currentTime = Time.time;
			
			// 采样鼠标输入（支持测试模式）
			Vector2 mousePos = GetTestMousePosition(currentTime);
			dataPoint.mouseInput = new MouseInput
			{
				mouseX = mousePos.x,
				mouseY = mousePos.y,
				timestamp = currentTime
			};
			
			// 采样物理状态
			dataPoint.physicsState = new PhysicsState
			{
				playerX = playerTransform.position.x,
				playerY = playerTransform.position.y,
				velocityX = playerRigidbody.velocity.x,
				velocityY = playerRigidbody.velocity.y,
				hammerAngle = GetHammerAngle(),
				hammerAngularVel = GetHammerAngularVelocity(),
				isGrounded = CheckGroundedStatus(),
				timestamp = currentTime
			};
			
			dataPoint.deltaTime = trackingData.Count > 0 ? 
				currentTime - lastDataPoint.physicsState.timestamp : 0f;
			
			return dataPoint;
		}
		
		/// <summary>
		/// 计算变化量
		/// </summary>
		private static void CalculateDeltas(ref TrackingDataPoint current, TrackingDataPoint previous)
		{
			// 鼠标移动计算
			current.mouseDeltaX = current.mouseInput.mouseX - previous.mouseInput.mouseX;
			current.mouseDeltaY = current.mouseInput.mouseY - previous.mouseInput.mouseY;
			current.mouseMoveDistance = Mathf.Sqrt(current.mouseDeltaX * current.mouseDeltaX + 
												  current.mouseDeltaY * current.mouseDeltaY);
			current.mouseMoveSpeed = current.deltaTime > 0 ? 
				current.mouseMoveDistance / current.deltaTime : 0f;
			
			// 物理变化计算
			Vector2 currentPos = current.physicsState.playerPosition;
			Vector2 previousPos = previous.physicsState.playerPosition;
			current.positionDelta = Vector2.Distance(currentPos, previousPos);
			
			Vector2 currentVel = current.physicsState.velocity;
			Vector2 previousVel = previous.physicsState.velocity;
			current.velocityDelta = Vector2.Distance(currentVel, previousVel);
			
			current.angleDelta = Mathf.Abs(current.physicsState.hammerAngle - previous.physicsState.hammerAngle);
			
			// 物理响应延迟（简化版本）
			current.physicsResponseDelay = current.deltaTime;
		}
		
		/// <summary>
		/// 获取锤子角度
		/// </summary>
		private static float GetHammerAngle()
		{
			if (hammerTip != null && playerTransform != null)
			{
				Vector3 hammerDirection = hammerTip.position - playerTransform.position;
				return Mathf.Atan2(hammerDirection.y, hammerDirection.x) * Mathf.Rad2Deg;
			}
			return 0f;
		}
		
		/// <summary>
		/// 获取锤子角速度
		/// </summary>
		private static float GetHammerAngularVelocity()
		{
			if (mainHinge != null)
			{
				return mainHinge.jointSpeed;
			}
			return 0f;
		}
		
		/// <summary>
		/// 检查接地状态
		/// </summary>
		private static bool CheckGroundedStatus()
		{
			if (potCollider != null)
			{
				// 简化的接地检测：检查碰撞体是否与地面接触
				ContactFilter2D filter = new ContactFilter2D();
				filter.SetLayerMask(LayerMask.GetMask("Terrain"));
				
				Collider2D[] results = new Collider2D[5];
				int contactCount = potCollider.OverlapCollider(filter, results);
				return contactCount > 0;
			}
			
			// 基于速度的简单检测
			return Mathf.Abs(playerRigidbody.velocity.y) < 0.1f;
		}
		
		#endregion
	}
}
