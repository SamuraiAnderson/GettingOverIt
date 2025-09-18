using UnityEngine;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System;
using System.Linq;

namespace GoiHitboxLogger
{
	/// <summary>
	/// 连续状态跟踪器 - 高频采样，精简参数，连续记录
	/// 专注于最小有效参数集的连续跟踪分析
	/// </summary>
	public static class ContinuousTracker
	{
		/// <summary>
		/// 鼠标输入状态 - 2自由度
		/// Getting Over It的控制完全基于鼠标移动，没有按钮输入
		/// </summary>
		[Serializable]
		public struct MouseInput
		{
			public float mouseX;        // 鼠标X坐标（屏幕坐标）
			public float mouseY;        // 鼠标Y坐标（屏幕坐标）
			public float timestamp;     // 时间戳
			
			// 计算属性
			public Vector2 mousePosition => new Vector2(mouseX, mouseY);
		}

		/// <summary>
		/// 游戏物理状态 - 6自由度
		/// 专注于物理响应和锤子控制状态
		/// </summary>
		[Serializable]
		public struct PhysicsState
		{
			public float playerX;       // Player X坐标
			public float playerY;       // Player Y坐标
			public float velocityX;     // Player X速度
			public float velocityY;     // Player Y速度
			public float hammerAngle;   // 锤子相对角度（相对Player）
			public float hammerAngularVel; // 锤子角速度
			public bool isGrounded;     // 接地状态
			public float timestamp;     // 时间戳
			
			// 计算属性
			public Vector2 playerPosition => new Vector2(playerX, playerY);
			public Vector2 velocity => new Vector2(velocityX, velocityY);
			public float speed => velocity.magnitude;
		}

		/// <summary>
		/// 鼠标-物理响应跟踪数据点
		/// 专注于鼠标移动与物理响应的映射关系
		/// </summary>
		[Serializable]
		public struct TrackingDataPoint
		{
			public MouseInput mouseInput;
			public PhysicsState physicsState;
			public float deltaTime;     // 与上一帧的时间差
			
			// 鼠标移动分析
			public float mouseDeltaX;   // 鼠标X变化
			public float mouseDeltaY;   // 鼠标Y变化
			public float mouseMoveDistance; // 鼠标移动距离
			public float mouseMoveSpeed;    // 鼠标移动速度
			
			// 物理响应分析
			public float positionDelta;  // 位置变化量
			public float velocityDelta;  // 速度变化量
			public float angleDelta;     // 角度变化量
			public float physicsResponseDelay; // 物理响应延迟
		}

		// 跟踪设置
		private static float trackingInterval = 0.02f; // 50Hz采样
		private static int maxDataPoints = 10000; // 最大数据点数（约3.3分钟@50Hz）
		
		// 数据存储
		private static List<TrackingDataPoint> trackingData = new List<TrackingDataPoint>();
		private static TrackingDataPoint lastDataPoint;
		private static float lastSampleTime = 0f;
		
		// 系统状态
		private static bool isTracking = false;
		private static bool isInitialized = false;
		
		// 自动化采集支持
		private static bool autoCollectionMode = false;
		private static float autoCollectionStartTime = 0f;
		private static float autoCollectionDuration = 5f;
		private static int autoCollectionSessionId = 0;
		
		// 游戏采样频率检测
		private static List<float> gameFrameTimes = new List<float>();
		private static List<Vector2> gameMousePositions = new List<Vector2>();
		private static float lastGameFrameTime = 0f;
		private static Vector2 lastGameMousePosition = Vector2.zero;
		
		// 组件引用
		private static GameObject player;
		private static Transform playerTransform;
		private static Rigidbody2D playerRigidbody;
		private static Transform hammerTip;
		private static HingeJoint2D mainHinge;
		private static PolygonCollider2D potCollider;

		/// <summary>
		/// 初始化连续跟踪器
		/// </summary>
		public static bool Initialize()
		{
			try
			{
				// 查找Player对象
				player = GameObject.Find("Player");
				if (player == null)
				{
					Debug.LogError("❌ ContinuousTracker: 未找到Player对象");
					return false;
				}

				// 获取核心组件引用
				playerTransform = player.transform;
				playerRigidbody = player.GetComponent<Rigidbody2D>();
				mainHinge = player.GetComponent<HingeJoint2D>();

				// 查找锤子头
				hammerTip = player.transform.Find("Hub/Slider/Handle/PoleMiddle/Tip");

				// 查找锅碰撞器
				Transform potColliderTransform = player.transform.Find("PotCollider");
				if (potColliderTransform != null)
				{
					potCollider = potColliderTransform.GetComponent<PolygonCollider2D>();
				}

				isInitialized = (playerTransform != null && playerRigidbody != null);
				
				if (isInitialized)
				{
					trackingData.Clear();
					lastSampleTime = Time.time;
					Debug.Log("✅ ContinuousTracker: 初始化成功");
					Debug.Log($"   - 采样频率: {1f/trackingInterval:F0}Hz");
					Debug.Log($"   - 最大数据点: {maxDataPoints}");
					Debug.Log($"   - 预计记录时长: {maxDataPoints * trackingInterval / 60f:F1}分钟");
				}
				else
				{
					Debug.LogError("❌ ContinuousTracker: 初始化失败 - 缺少关键组件");
				}

				return isInitialized;
			}
			catch (Exception e)
			{
				Debug.LogError($"❌ ContinuousTracker初始化异常: {e.Message}");
				return false;
			}
		}

		/// <summary>
		/// 开始连续跟踪
		/// </summary>
		public static void StartTracking()
		{
			if (!isInitialized)
			{
				Debug.LogWarning("⚠️ ContinuousTracker未初始化");
				return;
			}

			isTracking = true;
			trackingData.Clear();
			lastSampleTime = Time.time;
			Debug.Log("🔄 连续跟踪已开始");
		}

		/// <summary>
		/// 停止连续跟踪
		/// </summary>
		public static void StopTracking()
		{
			isTracking = false;
			Debug.Log($"⏹️ 连续跟踪已停止 - 共记录 {trackingData.Count} 个数据点");
		}

		/// <summary>
		/// 更新连续跟踪（需要在Update中调用）
		/// </summary>
		public static void UpdateTracking()
		{
			if (!isInitialized) return;

			float currentTime = Time.time;
			
			// 检测游戏本身的帧率和鼠标采样频率
			DetectGameSamplingFrequency(currentTime);
			
			// 检查自动化采集信号
			CheckAutoCollectionSignal();
			
			// 检查自动化采集是否完成
			if (autoCollectionMode && currentTime - autoCollectionStartTime >= autoCollectionDuration)
			{
				StopAutoCollection();
			}
			
			if (!isTracking) return;

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
					trackingData.RemoveAt(0); // 移除最旧的数据点
				}
			}
		}

		/// <summary>
		/// 采样当前鼠标-物理响应数据
		/// </summary>
		private static TrackingDataPoint SampleCurrentData()
		{
			TrackingDataPoint dataPoint = new TrackingDataPoint();
			float currentTime = Time.time;

			// 采样鼠标输入数据
			dataPoint.mouseInput = new MouseInput
			{
				mouseX = Input.mousePosition.x,
				mouseY = Input.mousePosition.y,
				timestamp = currentTime
			};

			// 采样物理状态数据
			dataPoint.physicsState = new PhysicsState
			{
				playerX = playerTransform.position.x,
				playerY = playerTransform.position.y,
				velocityX = playerRigidbody.velocity.x,
				velocityY = playerRigidbody.velocity.y,
				timestamp = currentTime
			};

			// 计算锤子相对角度
			if (hammerTip != null)
			{
				Vector3 hammerDirection = hammerTip.position - playerTransform.position;
				dataPoint.physicsState.hammerAngle = Mathf.Atan2(hammerDirection.y, hammerDirection.x) * Mathf.Rad2Deg;
			}

			// 计算锤子角速度（通过主关节）
			if (mainHinge != null)
			{
				dataPoint.physicsState.hammerAngularVel = mainHinge.jointSpeed;
			}

			// 检查接地状态
			dataPoint.physicsState.isGrounded = CheckGroundContact();

			// 时间差
			if (trackingData.Count > 0)
			{
				dataPoint.deltaTime = currentTime - lastDataPoint.physicsState.timestamp;
			}

			return dataPoint;
		}

		/// <summary>
		/// 计算鼠标移动模式和物理响应变化率
		/// </summary>
		private static void CalculateDeltas(ref TrackingDataPoint current, TrackingDataPoint previous)
		{
			// 鼠标移动分析
			current.mouseDeltaX = current.mouseInput.mouseX - previous.mouseInput.mouseX;
			current.mouseDeltaY = current.mouseInput.mouseY - previous.mouseInput.mouseY;
			current.mouseMoveDistance = Mathf.Sqrt(current.mouseDeltaX * current.mouseDeltaX + current.mouseDeltaY * current.mouseDeltaY);
			current.mouseMoveSpeed = current.mouseMoveDistance / current.deltaTime;

			// 物理响应分析
			Vector2 positionDelta = current.physicsState.playerPosition - previous.physicsState.playerPosition;
			current.positionDelta = positionDelta.magnitude;
			
			Vector2 velocityDelta = current.physicsState.velocity - previous.physicsState.velocity;
			current.velocityDelta = velocityDelta.magnitude;
			
			current.angleDelta = Mathf.DeltaAngle(previous.physicsState.hammerAngle, current.physicsState.hammerAngle);
			
			// 物理响应延迟分析（简化：基于鼠标移动与角色速度的相关性）
			current.physicsResponseDelay = CalculateResponseDelay(current, previous);
		}

		/// <summary>
		/// 检测游戏本身的采样频率
		/// 分析游戏帧率和鼠标输入采样频率
		/// </summary>
		private static void DetectGameSamplingFrequency(float currentTime)
		{
			// 记录游戏帧时间
			if (lastGameFrameTime > 0)
			{
				float frameTime = currentTime - lastGameFrameTime;
				gameFrameTimes.Add(frameTime);
				
				// 保持最近100帧的数据
				if (gameFrameTimes.Count > 100)
				{
					gameFrameTimes.RemoveAt(0);
				}
			}
			lastGameFrameTime = currentTime;
			
			// 记录鼠标位置变化
			Vector2 currentMousePosition = Input.mousePosition;
			if (lastGameMousePosition != Vector2.zero)
			{
				float mouseMoveDistance = Vector2.Distance(currentMousePosition, lastGameMousePosition);
				if (mouseMoveDistance > 0.1f) // 只记录有意义的移动
				{
					gameMousePositions.Add(currentMousePosition);
					
					// 保持最近100个鼠标位置
					if (gameMousePositions.Count > 100)
					{
						gameMousePositions.RemoveAt(0);
					}
				}
			}
			lastGameMousePosition = currentMousePosition;
		}

		/// <summary>
		/// 计算物理响应延迟
		/// 分析鼠标移动与角色物理响应的延迟关系
		/// </summary>
		private static float CalculateResponseDelay(TrackingDataPoint current, TrackingDataPoint previous)
		{
			// 简化的响应延迟计算：基于鼠标移动强度与角色速度变化的关系
			float mouseMoveIntensity = current.mouseMoveDistance;
			float playerSpeedChange = current.velocityDelta;
			
			// 如果鼠标移动强度高但角色速度变化小，可能存在延迟
			if (mouseMoveIntensity > 1.0f && playerSpeedChange < 0.5f)
			{
				return 1.0f; // 表示可能存在响应延迟
			}
			
			return 0.0f; // 响应正常
		}

		/// <summary>
		/// 检查地面接触
		/// </summary>
		private static bool CheckGroundContact()
		{
			if (potCollider == null || playerRigidbody == null) return false;
			
			// 简化的接地检测：基于垂直速度和速度变化
			return Mathf.Abs(playerRigidbody.velocity.y) < 0.1f && 
				   playerRigidbody.velocity.magnitude < 1.0f;
		}

		/// <summary>
		/// 获取跟踪统计信息
		/// </summary>
		public static string GetTrackingStats()
		{
			if (!isInitialized) return "❌ 未初始化";
			
			StringBuilder stats = new StringBuilder();
			stats.AppendLine("📊 连续跟踪统计:");
			stats.AppendLine($"  状态: {(isTracking ? "🔄 跟踪中" : "⏹️ 已停止")}");
			stats.AppendLine($"  数据点数: {trackingData.Count}");
			stats.AppendLine($"  采样频率: {1f/trackingInterval:F0}Hz");
			
			if (trackingData.Count > 0)
			{
				float duration = trackingData[trackingData.Count - 1].physicsState.timestamp - trackingData[0].physicsState.timestamp;
				stats.AppendLine($"  记录时长: {duration:F1}秒");
				stats.AppendLine($"  实际频率: {trackingData.Count / duration:F1}Hz");
			}

			return stats.ToString();
		}

		/// <summary>
		/// 获取游戏本身的采样频率信息
		/// </summary>
		public static string GetGameSamplingInfo()
		{
			if (!isInitialized) return "❌ 未初始化";
			
			StringBuilder info = new StringBuilder();
			info.AppendLine("🎮 游戏采样频率分析:");
			
			// 游戏帧率分析
			if (gameFrameTimes.Count > 10)
			{
				float avgFrameTime = gameFrameTimes.Average();
				float gameFrameRate = 1f / avgFrameTime;
				float minFrameTime = gameFrameTimes.Min();
				float maxFrameTime = gameFrameTimes.Max();
				float minFrameRate = 1f / maxFrameTime;
				float maxFrameRate = 1f / minFrameTime;
				
				info.AppendLine($"  游戏帧率: {gameFrameRate:F1}Hz (平均)");
				info.AppendLine($"  帧率范围: {minFrameRate:F1}-{maxFrameRate:F1}Hz");
				info.AppendLine($"  帧时间: {avgFrameTime*1000f:F1}ms (平均)");
				info.AppendLine($"  帧时间范围: {minFrameTime*1000f:F1}-{maxFrameTime*1000f:F1}ms");
			}
			else
			{
				info.AppendLine("  游戏帧率: 数据不足，需要更多采样");
			}
			
			// 鼠标采样分析
			if (gameMousePositions.Count > 10)
			{
				info.AppendLine($"  鼠标位置采样: {gameMousePositions.Count} 个有效位置");
				info.AppendLine($"  鼠标移动检测: 已检测到 {gameMousePositions.Count} 次移动");
			}
			else
			{
				info.AppendLine("  鼠标采样: 数据不足，需要更多鼠标移动");
			}
			
			// Unity系统信息
			info.AppendLine($"  Unity帧率: {1f/Time.deltaTime:F1}Hz (当前)");
			info.AppendLine($"  Unity帧时间: {Time.deltaTime*1000f:F1}ms (当前)");
			
			return info.ToString();
		}

		/// <summary>
		/// 导出连续跟踪数据
		/// </summary>
		public static void ExportTrackingData()
		{
			if (trackingData.Count == 0)
			{
				Debug.LogWarning("⚠️ 没有跟踪数据可导出");
				return;
			}

			try
			{
				StringBuilder report = new StringBuilder();
				
				// 文件头信息
				report.AppendLine("=== Getting Over It 连续跟踪数据 ===");
				report.AppendLine($"导出时间: {DateTime.Now}");
				report.AppendLine($"数据点数: {trackingData.Count}");
				report.AppendLine($"采样频率: {1f/trackingInterval:F0}Hz");
				
				if (trackingData.Count > 0)
				{
					float duration = trackingData[trackingData.Count - 1].physicsState.timestamp - trackingData[0].physicsState.timestamp;
					report.AppendLine($"记录时长: {duration:F2}秒");
				}
				
				report.AppendLine();
				
				// CSV格式头部
				report.AppendLine("# 数据格式: CSV - 鼠标移动-物理响应分析");
				report.AppendLine("# 鼠标输入(2DOF): mouseX, mouseY");
				report.AppendLine("# 物理状态(6DOF): playerX, playerY, velocityX, velocityY, hammerAngle, hammerAngularVel, isGrounded");
				report.AppendLine("# 鼠标移动分析: mouseDeltaX, mouseDeltaY, mouseMoveDistance, mouseMoveSpeed");
				report.AppendLine("# 物理响应分析: positionDelta, velocityDelta, angleDelta, physicsResponseDelay");
				report.AppendLine("# 注意: Getting Over It的控制完全基于鼠标移动，没有按钮输入");
				report.AppendLine();
				
				// CSV表头
				report.AppendLine("timestamp,deltaTime," +
					"mouseX,mouseY," +
					"playerX,playerY,velocityX,velocityY,hammerAngle,hammerAngularVel,isGrounded," +
					"mouseDeltaX,mouseDeltaY,mouseMoveDistance,mouseMoveSpeed," +
					"positionDelta,velocityDelta,angleDelta,physicsResponseDelay");

				// 数据行
				foreach (var dataPoint in trackingData)
				{
					report.AppendLine($"{dataPoint.physicsState.timestamp:F3},{dataPoint.deltaTime:F4}," +
						$"{dataPoint.mouseInput.mouseX:F1},{dataPoint.mouseInput.mouseY:F1}," +
						$"{dataPoint.physicsState.playerX:F3},{dataPoint.physicsState.playerY:F3}," +
						$"{dataPoint.physicsState.velocityX:F3},{dataPoint.physicsState.velocityY:F3}," +
						$"{dataPoint.physicsState.hammerAngle:F2},{dataPoint.physicsState.hammerAngularVel:F3}," +
						$"{(dataPoint.physicsState.isGrounded ? 1 : 0)}," +
						$"{dataPoint.mouseDeltaX:F2},{dataPoint.mouseDeltaY:F2}," +
						$"{dataPoint.mouseMoveDistance:F2},{dataPoint.mouseMoveSpeed:F2}," +
						$"{dataPoint.positionDelta:F4},{dataPoint.velocityDelta:F4},{dataPoint.angleDelta:F3},{dataPoint.physicsResponseDelay:F2}");
				}

				// 保存文件
				string fileName = $"ContinuousTracking_{DateTime.Now:yyyyMMdd_HHmmss}.csv";
				string baseDir = Path.Combine(Application.dataPath, "..");
				string trackingDumpDir = Path.Combine(baseDir, "TrackingDump");
				string filePath = Path.Combine(trackingDumpDir, fileName);

				Directory.CreateDirectory(Path.GetDirectoryName(filePath));
				File.WriteAllText(filePath, report.ToString());
				
				Debug.Log($"✅ 连续跟踪数据已导出到: {filePath}");
				Debug.Log($"📊 数据摘要: {trackingData.Count}个数据点，{GetDataSummary()}");
			}
			catch (Exception e)
			{
				Debug.LogError($"❌ 导出连续跟踪数据失败: {e.Message}");
			}
		}

		/// <summary>
		/// 获取数据摘要
		/// </summary>
		private static string GetDataSummary()
		{
			if (trackingData.Count == 0) return "无数据";

			// 计算基础统计
			float minSpeed = float.MaxValue, maxSpeed = 0f, avgSpeed = 0f;
			float minAngle = float.MaxValue, maxAngle = float.MinValue;
			
			foreach (var data in trackingData)
			{
				float speed = data.physicsState.speed;
				minSpeed = Mathf.Min(minSpeed, speed);
				maxSpeed = Mathf.Max(maxSpeed, speed);
				avgSpeed += speed;
				
				minAngle = Mathf.Min(minAngle, data.physicsState.hammerAngle);
				maxAngle = Mathf.Max(maxAngle, data.physicsState.hammerAngle);
				
				// 移除抓握变化统计，专注于鼠标移动分析
				// if (data.grippingChanged) grippingChanges++;
			}
			
			avgSpeed /= trackingData.Count;
			
			return $"速度范围{minSpeed:F1}-{maxSpeed:F1}(平均{avgSpeed:F1}), " +
				   $"角度范围{minAngle:F0}°-{maxAngle:F0}°";
		}

		/// <summary>
		/// 设置采样频率
		/// </summary>
		public static void SetSamplingRate(float frequency)
		{
			// 在自动化采集模式下强制使用30Hz
			if (autoCollectionMode)
			{
				trackingInterval = 1f / 30f; // 固定30Hz
				Debug.Log($"🤖 自动化模式: 强制使用30Hz采样频率 (间隔{trackingInterval * 1000f:F1}ms)");
			}
			else
			{
				trackingInterval = 1f / Mathf.Clamp(frequency, 1f, 200f); // 1-200Hz
				Debug.Log($"📊 采样频率已设置为: {frequency:F0}Hz (间隔{trackingInterval * 1000f:F1}ms)");
			}
		}

		/// <summary>
		/// 获取当前数据点数
		/// </summary>
		public static int GetDataPointCount()
		{
			return trackingData.Count;
		}

		/// <summary>
		/// 是否正在跟踪
		/// </summary>
		public static bool IsTracking()
		{
			return isTracking;
		}

		/// <summary>
		/// 是否已初始化
		/// </summary>
		public static bool IsInitialized()
		{
			return isInitialized;
		}

		/// <summary>
		/// 清空跟踪数据
		/// </summary>
		public static void ClearData()
		{
			trackingData.Clear();
			Debug.Log("🗑️ 跟踪数据已清空");
		}

		/// <summary>
		/// 检查自动化采集信号
		/// </summary>
		private static void CheckAutoCollectionSignal()
		{
			string basePath = Path.Combine(Application.dataPath, "..");
			string srcPath = Path.Combine(basePath, "src");
			string dataPath = Path.Combine(srcPath, "Data");
			string signalPath = Path.Combine(dataPath, "collection_signal.json");
			
			if (File.Exists(signalPath))
			{
				try
				{
					string jsonContent = File.ReadAllText(signalPath);
					var signalData = JsonUtility.FromJson<AutoCollectionSignal>(jsonContent);
					
					if (signalData.signal == "start" && !autoCollectionMode)
					{
						StartAutoCollection(signalData.session_id, signalData.duration, signalData.frequency);
					}
					else if (signalData.signal == "stop" && autoCollectionMode)
					{
						StopAutoCollection();
					}
					
					// 删除信号文件
					File.Delete(signalPath);
				}
				catch (Exception e)
				{
					Debug.LogError($"❌ 读取自动化采集信号失败: {e.Message}");
				}
			}
		}
		
		/// <summary>
		/// 开始自动化采集
		/// </summary>
		private static void StartAutoCollection(int sessionId, float duration, int frequency)
		{
			autoCollectionMode = true;
			autoCollectionStartTime = Time.time;
			autoCollectionDuration = duration;
			autoCollectionSessionId = sessionId;
			
			// 设置采样频率
			SetSamplingRate(frequency);
			
			// 开始跟踪
			StartTracking();
			
			// 发送响应信号
			SendAutoCollectionResponse("started");
			
			Debug.Log($"🤖 自动化采集已开始 - 会话#{sessionId}, 时长{duration}秒, 频率{frequency}Hz");
		}
		
		/// <summary>
		/// 停止自动化采集
		/// </summary>
		private static void StopAutoCollection()
		{
			if (!autoCollectionMode) return;
			
			autoCollectionMode = false;
			
			// 停止跟踪
			StopTracking();
			
			// 导出数据
			ExportTrackingData();
			
			// 发送响应信号
			SendAutoCollectionResponse("completed");
			
			Debug.Log($"🤖 自动化采集已停止 - 会话#{autoCollectionSessionId}, 数据点{trackingData.Count}");
		}
		
		/// <summary>
		/// 发送自动化采集响应
		/// </summary>
		private static void SendAutoCollectionResponse(string response)
		{
			try
			{
				var responseData = new AutoCollectionResponse
				{
					response = response,
					session_id = autoCollectionSessionId,
					data_points = trackingData.Count,
					timestamp = Time.time
				};
				
				string jsonContent = JsonUtility.ToJson(responseData, true);
				string basePath = Path.Combine(Application.dataPath, "..");
				string srcPath = Path.Combine(basePath, "src");
				string dataPath = Path.Combine(srcPath, "Data");
				string responsePath = Path.Combine(dataPath, "unity_response.json");
				
				File.WriteAllText(responsePath, jsonContent);
			}
			catch (Exception e)
			{
				Debug.LogError($"❌ 发送自动化采集响应失败: {e.Message}");
			}
		}
		
		/// <summary>
		/// 重置跟踪器
		/// </summary>
		public static void Reset()
		{
			StopTracking();
			ClearData();
			isInitialized = false;
			autoCollectionMode = false;
			player = null;
			playerTransform = null;
			playerRigidbody = null;
			hammerTip = null;
			mainHinge = null;
			potCollider = null;
			
			Debug.Log("🔄 ContinuousTracker已重置");
		}
		
		/// <summary>
		/// 自动化采集信号数据结构
		/// </summary>
		[Serializable]
		public class AutoCollectionSignal
		{
			public string signal;
			public int session_id;
			public float duration;
			public int frequency;
			public float timestamp;
		}
		
		/// <summary>
		/// 自动化采集响应数据结构
		/// </summary>
		[Serializable]
		public class AutoCollectionResponse
		{
			public string response;
			public int session_id;
			public int data_points;
			public float timestamp;
		}
	}
}
