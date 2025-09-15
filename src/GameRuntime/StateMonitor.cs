using UnityEngine;
using System.Text;
using System.IO;

namespace GoiHitboxLogger
{
    /// <summary>
    /// 游戏状态监控器 - 监控Player的最终状态结果
    /// 专注于Action执行后的状态变化，而非输入控制
    /// </summary>
    public static class StateMonitor
    {
        /// <summary>
        /// Player状态数据结构
        /// </summary>
        public struct PlayerState
        {
            // 整体状态
            public Vector3 position;           // 玩家位置
            public Vector3 velocity;           // 玩家速度
            public float rotation;             // 玩家旋转角度
            public float angularVelocity;      // 角速度
            
            // 锤子状态
            public Vector3 hammerPosition;     // 锤子头位置
            public Vector3 hammerVelocity;     // 锤子头速度
            public float hammerAngle;          // 锤子角度（相对于玩家）
            public bool hammerColliding;       // 锤子是否正在碰撞
            
            // 碰撞状态
            public bool isGrounded;            // 是否接触地面
            public bool isPotColliding;        // 锅是否在碰撞
            public int activeCollisions;       // 活跃碰撞数量
            public Vector3 lastCollisionPoint; // 最后碰撞点
            
            // 运动状态
            public bool isMoving;              // 是否在移动
            public bool isFalling;             // 是否在下落
            public float movementSpeed;        // 移动速度
            public float verticalSpeed;        // 垂直速度
            
            // 时间戳
            public float timestamp;            // 状态采样时间
        }

        // 关键GameObject引用
        private static GameObject player;
        private static Transform hammerTip;
        private static Rigidbody2D playerRigidbody;
        private static PolygonCollider2D potCollider;
        private static Rigidbody2D hammerRigidbody;
        private static Transform playerTransform;
        
        // 状态缓存
        private static PlayerState currentState;
        private static PlayerState previousState;
        private static bool isInitialized = false;

        /// <summary>
        /// 初始化状态监控器
        /// </summary>
        public static bool Initialize()
        {
            try
            {
                // 查找Player对象
                player = GameObject.Find("Player");
                if (player == null)
                {
                    Debug.LogError("❌ StateMonitor: 未找到Player对象");
                    return false;
                }

                // 获取核心组件引用
                playerTransform = player.transform;
                playerRigidbody = player.GetComponent<Rigidbody2D>();
                
                // 查找锤子头
                hammerTip = player.transform.Find("Hub/Slider/Handle/PoleMiddle/Tip");
                if (hammerTip != null)
                {
                    hammerRigidbody = hammerTip.GetComponent<Rigidbody2D>();
                }

                // 查找锅碰撞器
                Transform potColliderTransform = player.transform.Find("PotCollider");
                if (potColliderTransform != null)
                {
                    potCollider = potColliderTransform.GetComponent<PolygonCollider2D>();
                }

                isInitialized = (playerRigidbody != null);
                
                if (isInitialized)
                {
                    Debug.Log("✅ StateMonitor: 初始化成功");
                    Debug.Log($"   - Player: {player.name}");
                    Debug.Log($"   - HammerTip: {(hammerTip != null ? "找到" : "未找到")}");
                    Debug.Log($"   - PotCollider: {(potCollider != null ? "找到" : "未找到")}");
                }
                else
                {
                    Debug.LogError("❌ StateMonitor: 初始化失败 - 缺少关键组件");
                }

                return isInitialized;
            }
            catch (System.Exception e)
            {
                Debug.LogError($"❌ StateMonitor初始化异常: {e.Message}");
                return false;
            }
        }

        /// <summary>
        /// 更新状态监控
        /// </summary>
        public static PlayerState UpdateState()
        {
            if (!isInitialized || player == null)
            {
                Debug.LogWarning("⚠️ StateMonitor: 未初始化或Player对象丢失");
                return currentState;
            }

            // 保存上一帧状态
            previousState = currentState;

            // 采样新状态
            currentState = SampleCurrentState();
            
            return currentState;
        }

        /// <summary>
        /// 采样当前状态
        /// </summary>
        private static PlayerState SampleCurrentState()
        {
            PlayerState state = new PlayerState
            {
                timestamp = Time.time
            };

            // 玩家整体状态
            if (playerTransform != null)
            {
                state.position = playerTransform.position;
                state.rotation = playerTransform.eulerAngles.z;
            }

            if (playerRigidbody != null)
            {
                state.velocity = playerRigidbody.velocity;
                state.angularVelocity = playerRigidbody.angularVelocity;
                state.movementSpeed = state.velocity.magnitude;
                state.verticalSpeed = state.velocity.y;
                state.isMoving = state.movementSpeed > 0.1f;
                state.isFalling = state.verticalSpeed < -0.5f;
            }

            // 锤子状态
            if (hammerTip != null)
            {
                state.hammerPosition = hammerTip.position;
                
                // 计算锤子相对角度
                Vector3 hammerDirection = state.hammerPosition - state.position;
                state.hammerAngle = Mathf.Atan2(hammerDirection.y, hammerDirection.x) * Mathf.Rad2Deg;

                if (hammerRigidbody != null)
                {
                    state.hammerVelocity = hammerRigidbody.velocity;
                }

                // 检测锤子碰撞（简单方法：检查速度变化）
                if (previousState.timestamp > 0)
                {
                    float velocityChange = (state.hammerVelocity - previousState.hammerVelocity).magnitude;
                    state.hammerColliding = velocityChange > 5.0f; // 速度突变阈值
                }
            }

            // 碰撞状态检测
            if (potCollider != null)
            {
                // 检查地面接触
                state.isGrounded = CheckGroundContact();
                state.isPotColliding = potCollider.IsTouching(potCollider);
                
                // 统计活跃碰撞
                state.activeCollisions = CountActiveCollisions();
            }

            return state;
        }

        /// <summary>
        /// 检查地面接触
        /// </summary>
        private static bool CheckGroundContact()
        {
            if (potCollider == null) return false;

            // 使用射线检测或碰撞器接触检测
            // 这里简化为检查垂直速度
            return playerRigidbody != null && 
                   Mathf.Abs(playerRigidbody.velocity.y) < 0.1f && 
                   playerRigidbody.velocity.magnitude < 1.0f;
        }

        /// <summary>
        /// 统计活跃碰撞数量
        /// </summary>
        private static int CountActiveCollisions()
        {
            if (potCollider == null) return 0;

            // 简化实现：基于速度和加速度变化判断
            if (playerRigidbody != null)
            {
                float speed = playerRigidbody.velocity.magnitude;
                return speed > 0.5f ? 1 : 0;
            }
            return 0;
        }

        /// <summary>
        /// 获取当前状态
        /// </summary>
        public static PlayerState GetCurrentState()
        {
            return currentState;
        }

        /// <summary>
        /// 获取上一帧状态
        /// </summary>
        public static PlayerState GetPreviousState()
        {
            return previousState;
        }

        /// <summary>
        /// 检查是否已初始化
        /// </summary>
        public static bool IsInitialized()
        {
            return isInitialized;
        }

        /// <summary>
        /// 打印当前状态信息
        /// </summary>
        public static void PrintCurrentState()
        {
            if (!isInitialized)
            {
                Debug.Log("⚠️ StateMonitor未初始化");
                return;
            }

            PlayerState state = currentState;
            StringBuilder sb = new StringBuilder();
            
            sb.AppendLine("🎮 === Player状态监控 ===");
            sb.AppendLine($"⏰ 时间戳: {state.timestamp:F2}s");
            sb.AppendLine();
            
            // 整体状态
            sb.AppendLine("📍 整体状态:");
            sb.AppendLine($"  位置: ({state.position.x:F2}, {state.position.y:F2}, {state.position.z:F2})");
            sb.AppendLine($"  速度: ({state.velocity.x:F2}, {state.velocity.y:F2}) 总速度: {state.movementSpeed:F2}");
            sb.AppendLine($"  旋转: {state.rotation:F1}° 角速度: {state.angularVelocity:F2}");
            sb.AppendLine($"  运动状态: {(state.isMoving ? "移动中" : "静止")} | {(state.isFalling ? "下落中" : "稳定")}");
            sb.AppendLine();
            
            // 锤子状态
            if (hammerTip != null)
            {
                sb.AppendLine("🔨 锤子状态:");
                sb.AppendLine($"  位置: ({state.hammerPosition.x:F2}, {state.hammerPosition.y:F2}, {state.hammerPosition.z:F2})");
                sb.AppendLine($"  速度: ({state.hammerVelocity.x:F2}, {state.hammerVelocity.y:F2})");
                sb.AppendLine($"  角度: {state.hammerAngle:F1}°");
                sb.AppendLine($"  碰撞: {(state.hammerColliding ? "是" : "否")}");
            }
            sb.AppendLine();
            
            // 碰撞状态
            sb.AppendLine("💥 碰撞状态:");
            sb.AppendLine($"  接地: {(state.isGrounded ? "是" : "否")}");
            sb.AppendLine($"  锅碰撞: {(state.isPotColliding ? "是" : "否")}");
            sb.AppendLine($"  活跃碰撞数: {state.activeCollisions}");

            Debug.Log(sb.ToString());
        }

        /// <summary>
        /// 导出状态数据到文件
        /// </summary>
        public static void ExportStateData(PlayerState state)
        {
            try
            {
                StringBuilder report = new StringBuilder();
                report.AppendLine("=== Player状态数据导出 ===");
                report.AppendLine($"导出时间: {System.DateTime.Now}");
                report.AppendLine($"游戏时间戳: {state.timestamp:F3}s");
                report.AppendLine();

                // 详细状态数据
                report.AppendLine("📊 详细状态数据:");
                report.AppendLine($"位置: {state.position.x:F3}, {state.position.y:F3}, {state.position.z:F3}");
                report.AppendLine($"速度: {state.velocity.x:F3}, {state.velocity.y:F3}");
                report.AppendLine($"旋转角度: {state.rotation:F3}");
                report.AppendLine($"角速度: {state.angularVelocity:F3}");
                report.AppendLine($"移动速度: {state.movementSpeed:F3}");
                report.AppendLine($"垂直速度: {state.verticalSpeed:F3}");
                report.AppendLine();

                report.AppendLine("🔨 锤子数据:");
                report.AppendLine($"锤子位置: {state.hammerPosition.x:F3}, {state.hammerPosition.y:F3}, {state.hammerPosition.z:F3}");
                report.AppendLine($"锤子速度: {state.hammerVelocity.x:F3}, {state.hammerVelocity.y:F3}");
                report.AppendLine($"锤子角度: {state.hammerAngle:F3}");
                report.AppendLine();

                report.AppendLine("💥 碰撞数据:");
                report.AppendLine($"接地状态: {state.isGrounded}");
                report.AppendLine($"锅碰撞: {state.isPotColliding}");
                report.AppendLine($"锤子碰撞: {state.hammerColliding}");
                report.AppendLine($"活跃碰撞数: {state.activeCollisions}");
                report.AppendLine();

                report.AppendLine("🏃 运动状态:");
                report.AppendLine($"是否移动: {state.isMoving}");
                report.AppendLine($"是否下落: {state.isFalling}");

                string fileName = $"PlayerState_{System.DateTime.Now:yyyyMMdd_HHmmss}.txt";
                string baseDir = Path.Combine(Application.dataPath, "..");
                string stateDumpDir = Path.Combine(baseDir, "StateDump");
                string filePath = Path.Combine(stateDumpDir, fileName);

                Directory.CreateDirectory(Path.GetDirectoryName(filePath));
                File.WriteAllText(filePath, report.ToString());
                Debug.Log($"✅ Player状态数据已导出到: {filePath}");
            }
            catch (System.Exception e)
            {
                Debug.LogError($"❌ 导出Player状态数据失败: {e.Message}");
            }
        }

        /// <summary>
        /// 重置监控器
        /// </summary>
        public static void Reset()
        {
            isInitialized = false;
            player = null;
            hammerTip = null;
            playerRigidbody = null;
            potCollider = null;
            hammerRigidbody = null;
            playerTransform = null;
            
            currentState = new PlayerState();
            previousState = new PlayerState();
            
            Debug.Log("🔄 StateMonitor已重置");
        }
    }
}
