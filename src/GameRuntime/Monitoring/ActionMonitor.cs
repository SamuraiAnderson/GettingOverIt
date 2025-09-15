using UnityEngine;
using System.Text;
using System.IO;
using System.Collections.Generic;

namespace GoiHitboxLogger
{
	/// <summary>
	/// 动作监控器 - 监控Player的输入控制和动作执行
	/// 专注于输入信号和控制状态，而非最终结果状态
	/// </summary>
	public static class ActionMonitor
	{
		/// <summary>
		/// Player动作数据结构
		/// </summary>
		public struct PlayerAction
		{
			// 控制器状态
			public bool playerControlActive;    // PlayerControl是否激活
			public Vector2 inputVector;         // 输入向量（如果可获取）
			public bool isControllerResponding; // 控制器是否响应
			
			// 关节控制状态
			public float mainHingeAngle;        // 主铰链角度
			public float mainHingeVelocity;     // 主铰链角速度
			public float sliderPosition;        // 滑动关节位置
			public float sliderVelocity;        // 滑动关节速度
			public bool hingeMotorActive;       // 铰链马达是否激活
			
			// 动画状态
			public string currentAnimation;     // 当前动画状态
			public float animationTime;         // 动画时间
			public bool animatorEnabled;        // 动画器是否启用
			public int animationLayers;         // 动画层数
			
			// 音效状态
			public bool audioPlaying;           // 是否正在播放音频
			public float audioVolume;           // 音频音量
			public string currentAudioClip;     // 当前音频片段
			
			// 物理控制状态
			public bool gravityEnabled;         // 重力是否启用
			public float drag;                  // 阻力系数
			public float angularDrag;           // 角阻力系数
			public bool kinematicMode;          // 是否为运动学模式
			
			// 交互状态
			public bool handsGripping;          // 手是否抓握
			public Vector3 leftHandTarget;      // 左手目标位置
			public Vector3 rightHandTarget;     // 右手目标位置
			public bool lookTargetActive;       // 视线目标是否激活
			
			// 时间戳
			public float timestamp;             // 动作采样时间
		}

		// 关键组件引用
		private static GameObject player;
		private static MonoBehaviour playerControl;
		private static HingeJoint2D mainHinge;
		private static HingeJoint2D handleHinge;
		private static SliderJoint2D sliderJoint;
		private static Animator dudeAnimator;
		private static Animator handleAnimator;
		private static AudioSource playerAudio;
		private static Rigidbody2D playerRigidbody;
		
		// IK和控制目标
		private static Transform leftTarget;
		private static Transform rightTarget;
		private static Transform lookTarget;
		
		// 动作缓存
		private static PlayerAction currentAction;
		private static PlayerAction previousAction;
		private static bool isInitialized = false;

		/// <summary>
		/// 初始化动作监控器
		/// </summary>
		public static bool Initialize()
		{
			try
			{
				// 查找Player对象
				player = GameObject.Find("Player");
				if (player == null)
				{
					Debug.LogError("❌ ActionMonitor: 未找到Player对象");
					return false;
				}

				// 获取PlayerControl组件
				playerControl = player.GetComponent<MonoBehaviour>();
				playerRigidbody = player.GetComponent<Rigidbody2D>();
				playerAudio = player.GetComponent<AudioSource>();

				// 查找关节组件
				mainHinge = player.GetComponent<HingeJoint2D>();
				
				Transform hubTransform = player.transform.Find("Hub");
				if (hubTransform != null)
				{
					Transform sliderTransform = hubTransform.Find("Slider");
					if (sliderTransform != null)
					{
						sliderJoint = sliderTransform.GetComponent<SliderJoint2D>();
						
						Transform handleTransform = sliderTransform.Find("Handle");
						if (handleTransform != null)
						{
							handleHinge = handleTransform.GetComponent<HingeJoint2D>();
						}
					}
				}

				// 查找动画器
				Transform dudeTransform = player.transform.Find("dude");
				if (dudeTransform != null)
				{
					dudeAnimator = dudeTransform.GetComponent<Animator>();
					
					// 查找IK目标
					leftTarget = dudeTransform.Find("leftCenter");
					rightTarget = dudeTransform.Find("rightCenter");
					lookTarget = dudeTransform.Find("LookTarget");
				}

				Transform handleTransformForAnim = player.transform.Find("handle");
				if (handleTransformForAnim != null)
				{
					handleAnimator = handleTransformForAnim.GetComponent<Animator>();
				}

				isInitialized = (player != null && playerRigidbody != null);
				
				if (isInitialized)
				{
					Debug.Log("✅ ActionMonitor: 初始化成功");
					Debug.Log($"   - PlayerControl: {(playerControl != null ? "找到" : "未找到")}");
					Debug.Log($"   - MainHinge: {(mainHinge != null ? "找到" : "未找到")}");
					Debug.Log($"   - SliderJoint: {(sliderJoint != null ? "找到" : "未找到")}");
					Debug.Log($"   - DudeAnimator: {(dudeAnimator != null ? "找到" : "未找到")}");
					Debug.Log($"   - HandTargets: L:{(leftTarget != null ? "找到" : "未找到")} R:{(rightTarget != null ? "找到" : "未找到")}");
				}
				else
				{
					Debug.LogError("❌ ActionMonitor: 初始化失败 - 缺少关键组件");
				}

				return isInitialized;
			}
			catch (System.Exception e)
			{
				Debug.LogError($"❌ ActionMonitor初始化异常: {e.Message}");
				return false;
			}
		}

		/// <summary>
		/// 更新动作监控
		/// </summary>
		public static PlayerAction UpdateAction()
		{
			if (!isInitialized || player == null)
			{
				Debug.LogWarning("⚠️ ActionMonitor: 未初始化或Player对象丢失");
				return currentAction;
			}

			// 保存上一帧动作
			previousAction = currentAction;

			// 采样新动作状态
			currentAction = SampleCurrentAction();
			
			return currentAction;
		}

		/// <summary>
		/// 采样当前动作状态
		/// </summary>
		private static PlayerAction SampleCurrentAction()
		{
			PlayerAction action = new PlayerAction
			{
				timestamp = Time.time
			};

			// 控制器状态
			if (playerControl != null)
			{
				action.playerControlActive = playerControl.enabled;
				action.isControllerResponding = true; // 简化判断
			}

			// 物理控制状态
			if (playerRigidbody != null)
			{
				action.gravityEnabled = playerRigidbody.gravityScale > 0;
				action.drag = playerRigidbody.drag;
				action.angularDrag = playerRigidbody.angularDrag;
				action.kinematicMode = playerRigidbody.isKinematic;
			}

			// 关节控制状态
			if (mainHinge != null)
			{
				action.mainHingeAngle = mainHinge.jointAngle;
				action.mainHingeVelocity = mainHinge.jointSpeed;
				action.hingeMotorActive = mainHinge.useMotor;
			}

			if (sliderJoint != null)
			{
				action.sliderPosition = sliderJoint.jointTranslation;
				action.sliderVelocity = sliderJoint.jointSpeed;
			}

			// 动画状态
			if (dudeAnimator != null && dudeAnimator.enabled)
			{
				action.animatorEnabled = true;
				action.animationLayers = dudeAnimator.layerCount;
				
				if (dudeAnimator.layerCount > 0)
				{
					AnimatorStateInfo stateInfo = dudeAnimator.GetCurrentAnimatorStateInfo(0);
					action.currentAnimation = GetAnimationStateName(stateInfo);
					action.animationTime = stateInfo.normalizedTime;
				}
			}

			// 音效状态
			if (playerAudio != null)
			{
				action.audioPlaying = playerAudio.isPlaying;
				action.audioVolume = playerAudio.volume;
				action.currentAudioClip = playerAudio.clip != null ? playerAudio.clip.name : "None";
			}

			// IK目标状态
			if (leftTarget != null)
			{
				action.leftHandTarget = leftTarget.position;
			}
			if (rightTarget != null)
			{
				action.rightHandTarget = rightTarget.position;
			}
			if (lookTarget != null)
			{
				action.lookTargetActive = lookTarget.gameObject.activeInHierarchy;
			}

			// 抓握状态（简化判断）
			action.handsGripping = CheckGrippingState();

			return action;
		}

		/// <summary>
		/// 获取动画状态名称
		/// </summary>
		private static string GetAnimationStateName(AnimatorStateInfo stateInfo)
		{
			// 由于无法直接获取状态名，使用hash值代替
			return $"State_{stateInfo.shortNameHash}";
		}

		/// <summary>
		/// 检查抓握状态
		/// </summary>
		private static bool CheckGrippingState()
		{
			// 简化实现：基于手部目标位置和主关节状态
			if (mainHinge != null && leftTarget != null && rightTarget != null)
			{
				float targetDistance = Vector3.Distance(leftTarget.position, rightTarget.position);
				return targetDistance < 1.5f; // 阈值可调整
			}
			return false;
		}

		/// <summary>
		/// 获取当前动作
		/// </summary>
		public static PlayerAction GetCurrentAction()
		{
			return currentAction;
		}

		/// <summary>
		/// 获取上一帧动作
		/// </summary>
		public static PlayerAction GetPreviousAction()
		{
			return previousAction;
		}

		/// <summary>
		/// 检查是否已初始化
		/// </summary>
		public static bool IsInitialized()
		{
			return isInitialized;
		}

		/// <summary>
		/// 打印当前动作信息
		/// </summary>
		public static void PrintCurrentAction()
		{
			if (!isInitialized)
			{
				Debug.Log("⚠️ ActionMonitor未初始化");
				return;
			}

			PlayerAction action = currentAction;
			StringBuilder sb = new StringBuilder();
			
			sb.AppendLine("🎮 === Player动作监控 ===");
			sb.AppendLine($"⏰ 时间戳: {action.timestamp:F2}s");
			sb.AppendLine();
			
			// 控制器状态
			sb.AppendLine("🎛️ 控制器状态:");
			sb.AppendLine($"  PlayerControl激活: {(action.playerControlActive ? "是" : "否")}");
			sb.AppendLine($"  控制器响应: {(action.isControllerResponding ? "是" : "否")}");
			sb.AppendLine($"  重力启用: {(action.gravityEnabled ? "是" : "否")}");
			sb.AppendLine($"  运动学模式: {(action.kinematicMode ? "是" : "否")}");
			sb.AppendLine();
			
			// 关节控制
			sb.AppendLine("🔗 关节控制:");
			sb.AppendLine($"  主铰链角度: {action.mainHingeAngle:F1}°");
			sb.AppendLine($"  主铰链速度: {action.mainHingeVelocity:F2}");
			sb.AppendLine($"  滑动位置: {action.sliderPosition:F2}");
			sb.AppendLine($"  滑动速度: {action.sliderVelocity:F2}");
			sb.AppendLine($"  马达激活: {(action.hingeMotorActive ? "是" : "否")}");
			sb.AppendLine();
			
			// 动画状态
			if (action.animatorEnabled)
			{
				sb.AppendLine("🎭 动画状态:");
				sb.AppendLine($"  当前动画: {action.currentAnimation}");
				sb.AppendLine($"  动画时间: {action.animationTime:F2}");
				sb.AppendLine($"  动画层数: {action.animationLayers}");
			}
			sb.AppendLine();
			
			// 交互状态
			sb.AppendLine("🤲 交互状态:");
			sb.AppendLine($"  手部抓握: {(action.handsGripping ? "是" : "否")}");
			sb.AppendLine($"  视线目标激活: {(action.lookTargetActive ? "是" : "否")}");
			sb.AppendLine($"  左手目标: ({action.leftHandTarget.x:F1}, {action.leftHandTarget.y:F1})");
			sb.AppendLine($"  右手目标: ({action.rightHandTarget.x:F1}, {action.rightHandTarget.y:F1})");
			sb.AppendLine();
			
			// 音效状态
			sb.AppendLine("🔊 音效状态:");
			sb.AppendLine($"  播放中: {(action.audioPlaying ? "是" : "否")}");
			sb.AppendLine($"  音量: {action.audioVolume:F2}");
			sb.AppendLine($"  当前片段: {action.currentAudioClip}");

			Debug.Log(sb.ToString());
		}

		/// <summary>
		/// 导出动作数据到文件
		/// </summary>
		public static void ExportActionData(PlayerAction action)
		{
			try
			{
				StringBuilder report = new StringBuilder();
				report.AppendLine("=== Player动作数据导出 ===");
				report.AppendLine($"导出时间: {System.DateTime.Now}");
				report.AppendLine($"游戏时间戳: {action.timestamp:F3}s");
				report.AppendLine();
				
				// 详细动作数据
				report.AppendLine("🎛️ 控制器数据:");
				report.AppendLine($"PlayerControl激活: {action.playerControlActive}");
				report.AppendLine($"控制器响应: {action.isControllerResponding}");
				report.AppendLine($"重力启用: {action.gravityEnabled}");
				report.AppendLine($"阻力系数: {action.drag:F3}");
				report.AppendLine($"角阻力系数: {action.angularDrag:F3}");
				report.AppendLine($"运动学模式: {action.kinematicMode}");
				report.AppendLine();
				
				report.AppendLine("🔗 关节数据:");
				report.AppendLine($"主铰链角度: {action.mainHingeAngle:F3}");
				report.AppendLine($"主铰链速度: {action.mainHingeVelocity:F3}");
				report.AppendLine($"滑动位置: {action.sliderPosition:F3}");
				report.AppendLine($"滑动速度: {action.sliderVelocity:F3}");
				report.AppendLine($"马达激活: {action.hingeMotorActive}");
				report.AppendLine();
				
				report.AppendLine("🎭 动画数据:");
				report.AppendLine($"动画器启用: {action.animatorEnabled}");
				report.AppendLine($"当前动画: {action.currentAnimation}");
				report.AppendLine($"动画时间: {action.animationTime:F3}");
				report.AppendLine($"动画层数: {action.animationLayers}");
				report.AppendLine();
				
				report.AppendLine("🤲 交互数据:");
				report.AppendLine($"手部抓握: {action.handsGripping}");
				report.AppendLine($"视线目标激活: {action.lookTargetActive}");
				report.AppendLine($"左手目标: {action.leftHandTarget.x:F3}, {action.leftHandTarget.y:F3}, {action.leftHandTarget.z:F3}");
				report.AppendLine($"右手目标: {action.rightHandTarget.x:F3}, {action.rightHandTarget.y:F3}, {action.rightHandTarget.z:F3}");
				report.AppendLine();
				
				report.AppendLine("🔊 音效数据:");
				report.AppendLine($"播放中: {action.audioPlaying}");
				report.AppendLine($"音量: {action.audioVolume:F3}");
				report.AppendLine($"当前片段: {action.currentAudioClip}");
				
				string fileName = $"PlayerAction_{System.DateTime.Now:yyyyMMdd_HHmmss}.txt";
				string baseDir = Path.Combine(Application.dataPath, "..");
				string actionDumpDir = Path.Combine(baseDir, "ActionDump");
				string filePath = Path.Combine(actionDumpDir, fileName);
				
				Directory.CreateDirectory(Path.GetDirectoryName(filePath));
				File.WriteAllText(filePath, report.ToString());
				Debug.Log($"✅ Player动作数据已导出到: {filePath}");
			}
			catch (System.Exception e)
			{
				Debug.LogError($"❌ 导出Player动作数据失败: {e.Message}");
			}
		}

		/// <summary>
		/// 重置监控器
		/// </summary>
		public static void Reset()
		{
			isInitialized = false;
			player = null;
			playerControl = null;
			mainHinge = null;
			handleHinge = null;
			sliderJoint = null;
			dudeAnimator = null;
			handleAnimator = null;
			playerAudio = null;
			playerRigidbody = null;
			leftTarget = null;
			rightTarget = null;
			lookTarget = null;
			
			currentAction = new PlayerAction();
			previousAction = new PlayerAction();
			
			Debug.Log("🔄 ActionMonitor已重置");
		}
	}
}
