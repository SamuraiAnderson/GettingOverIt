using BepInEx;
using UnityEngine;
using System.Collections.Generic;

namespace GoiHitboxLogger
{
    [BepInPlugin("com.symbol.goi.hitboxlogger", "GOI Hitbox Logger", "1.0.0")]
    public class GoiHitboxLogger : BaseUnityPlugin
    {
        private bool dataCollected = false;
        private bool playerAnalyzed = false;
        private List<HitboxCollector.ColliderInfo> mountainColliders;
        private PlayerAnalyzer.PlayerInfo playerInfo;
        
        // 监控器状态
        private bool stateMonitorEnabled = false;
        private bool actionMonitorEnabled = false;
        private bool stateMonitorInitialized = false;
        private bool actionMonitorInitialized = false;
        private StateMonitor.PlayerState currentState;
        private ActionMonitor.PlayerAction currentAction;
        
        // 连续跟踪器状态
        private bool continuousTrackerEnabled = false;
        private bool continuousTrackerInitialized = false;
        private bool isRecording = false;
        
        // 自动化流程状态
        private bool autoModeEnabled = false;
        private bool autoModeInitialized = false;
        private bool hasStartedNewGame = false;
        private bool waitingForMianScene = false;
        private float mianSceneLoadStartTime = 0f;
        private float mianSceneLoadTimeout = 30f; // 30秒超时
        
        void Start()
        {
            Logger.LogInfo("GOI Hitbox Logger 已启动!");
        }
        
        /// <summary>
        /// 初始化监控器
        /// </summary>
        private void InitializeMonitors()
        {
            if (!IsInGameScene()) return;
            
            // 初始化StateMonitor
            if (!stateMonitorInitialized)
            {
                stateMonitorInitialized = StateMonitor.Initialize();
                if (stateMonitorInitialized)
                {
                    Logger.LogInfo("✅ StateMonitor初始化成功");
                }
            }
            
            // 初始化ActionMonitor
            if (!actionMonitorInitialized)
            {
                actionMonitorInitialized = ActionMonitor.Initialize();
                if (actionMonitorInitialized)
                {
                    Logger.LogInfo("✅ ActionMonitor初始化成功");
                }
            }
            
            // 初始化ContinuousTracker
            if (!continuousTrackerInitialized)
            {
                continuousTrackerInitialized = ContinuousTracker.Initialize();
                if (continuousTrackerInitialized)
                {
                    Logger.LogInfo("✅ ContinuousTracker初始化成功");
                }
            }
        }
        
        void Update()
        {
            // 检查自动化流程信号
            CheckAutoModeSignal();
            
            // 处理自动化流程
            if (autoModeEnabled)
            {
                HandleAutoMode();
                return; // 自动化模式下不执行其他功能
            }
            
            // 检查是否在游戏界面
            if (!IsInGameScene()) 
            {
                dataCollected = false;
                playerAnalyzed = false;
                return;
            }
            
            // 只收集一次数据
            if (!dataCollected)
            {
                CollectHitboxData();
                dataCollected = true;
            }
            
            // 只分析一次Player
            if (!playerAnalyzed)
            {
                AnalyzePlayerData();
                playerAnalyzed = true;
            }
            
            // 移除原有的F8、F9热键（改为连续跟踪功能）
            // 移除原有的碰撞体数据功能，专注于连续跟踪
            
            // 按F10键分析Player对象
            if (Input.GetKeyDown(KeyCode.F10))
            {
                AnalyzePlayerData();
            }
            
            // 按F11键导出Player分析
            if (Input.GetKeyDown(KeyCode.F11))
            {
                ExportPlayerAnalysis();
            }
            
            // 监控器相关热键
            // 按F12键切换State监控
            if (Input.GetKeyDown(KeyCode.F12))
            {
                ToggleStateMonitor();
            }
            
            // 按F1键切换Action监控
            if (Input.GetKeyDown(KeyCode.F1))
            {
                ToggleActionMonitor();
            }
            
            // 按F2键打印当前State
            if (Input.GetKeyDown(KeyCode.F2))
            {
                PrintCurrentState();
            }
            
            // 按F3键打印当前Action
            if (Input.GetKeyDown(KeyCode.F3))
            {
                PrintCurrentAction();
            }
            
            // 按F4键导出State数据
            if (Input.GetKeyDown(KeyCode.F4))
            {
                ExportStateData();
            }
            
            // 按F5键导出Action数据
            if (Input.GetKeyDown(KeyCode.F5))
            {
                ExportActionData();
            }
            
            // 连续跟踪相关热键
            // 按F6键切换连续跟踪
            if (Input.GetKeyDown(KeyCode.F6))
            {
                ToggleContinuousTracking();
            }
            
            // 按F7键开始/停止记录
            if (Input.GetKeyDown(KeyCode.F7))
            {
                ToggleRecording();
            }
            
            // 按F8键导出连续跟踪数据
            if (Input.GetKeyDown(KeyCode.F8))
            {
                ExportContinuousData();
            }
            
            // 按F9键清空连续跟踪数据
            if (Input.GetKeyDown(KeyCode.F9))
            {
                ClearContinuousData();
            }
            
            // 按F键显示游戏采样频率信息
            if (Input.GetKeyDown(KeyCode.F))
            {
                string gameSamplingInfo = ContinuousTracker.GetGameSamplingInfo();
                Logger.LogInfo($"\n{gameSamplingInfo}");
            }
            
            // 初始化监控器
            InitializeMonitors();
            
            // 更新监控器
            UpdateMonitors();
        }
        
        /// <summary>
        /// 更新监控器
        /// </summary>
        private void UpdateMonitors()
        {
            // 更新State监控
            if (stateMonitorEnabled && stateMonitorInitialized)
            {
                currentState = StateMonitor.UpdateState();
            }
            
            // 更新Action监控
            if (actionMonitorEnabled && actionMonitorInitialized)
            {
                currentAction = ActionMonitor.UpdateAction();
            }
            
            // 更新连续跟踪
            if (continuousTrackerEnabled && continuousTrackerInitialized)
            {
                ContinuousTracker.UpdateTracking();
            }
        }
        
        /// <summary>
        /// 切换State监控
        /// </summary>
        private void ToggleStateMonitor()
        {
            stateMonitorEnabled = !stateMonitorEnabled;
            Logger.LogInfo($"StateMonitor已{(stateMonitorEnabled ? "启用" : "禁用")}");
            
            if (stateMonitorEnabled && !stateMonitorInitialized)
            {
                InitializeMonitors();
            }
        }
        
        /// <summary>
        /// 切换Action监控
        /// </summary>
        private void ToggleActionMonitor()
        {
            actionMonitorEnabled = !actionMonitorEnabled;
            Logger.LogInfo($"ActionMonitor已{(actionMonitorEnabled ? "启用" : "禁用")}");
            
            if (actionMonitorEnabled && !actionMonitorInitialized)
            {
                InitializeMonitors();
            }
        }
        
        /// <summary>
        /// 打印当前State
        /// </summary>
        private void PrintCurrentState()
        {
            if (!stateMonitorInitialized)
            {
                Logger.LogWarning("StateMonitor未初始化，请先进入游戏场景");
                return;
            }
            
            StateMonitor.PrintCurrentState();
        }
        
        /// <summary>
        /// 打印当前Action
        /// </summary>
        private void PrintCurrentAction()
        {
            if (!actionMonitorInitialized)
            {
                Logger.LogWarning("ActionMonitor未初始化，请先进入游戏场景");
                return;
            }
            
            ActionMonitor.PrintCurrentAction();
        }
        
        /// <summary>
        /// 导出State数据
        /// </summary>
        private void ExportStateData()
        {
            if (!stateMonitorInitialized)
            {
                Logger.LogWarning("StateMonitor未初始化，无法导出数据");
                return;
            }
            
            StateMonitor.ExportStateData(currentState);
        }
        
        /// <summary>
        /// 导出Action数据
        /// </summary>
        private void ExportActionData()
        {
            if (!actionMonitorInitialized)
            {
                Logger.LogWarning("ActionMonitor未初始化，无法导出数据");
                return;
            }
            
            ActionMonitor.ExportActionData(currentAction);
        }
        
        /// <summary>
        /// 切换连续跟踪
        /// </summary>
        private void ToggleContinuousTracking()
        {
            continuousTrackerEnabled = !continuousTrackerEnabled;
            Logger.LogInfo($"连续跟踪已{(continuousTrackerEnabled ? "启用" : "禁用")}");
            
            if (continuousTrackerEnabled && !continuousTrackerInitialized)
            {
                InitializeMonitors();
            }
            
            if (!continuousTrackerEnabled && isRecording)
            {
                // 如果禁用连续跟踪，同时停止记录
                ToggleRecording();
            }
        }
        
        /// <summary>
        /// 切换记录状态
        /// </summary>
        private void ToggleRecording()
        {
            if (!continuousTrackerInitialized)
            {
                Logger.LogWarning("连续跟踪器未初始化，无法开始记录");
                return;
            }
            
            if (!isRecording)
            {
                ContinuousTracker.StartTracking();
                isRecording = true;
                Logger.LogInfo("🔄 开始连续记录");
            }
            else
            {
                ContinuousTracker.StopTracking();
                isRecording = false;
                Logger.LogInfo("⏹️ 停止连续记录");
            }
        }
        
        /// <summary>
        /// 导出连续跟踪数据
        /// </summary>
        private void ExportContinuousData()
        {
            if (!continuousTrackerInitialized)
            {
                Logger.LogWarning("连续跟踪器未初始化，无法导出数据");
                return;
            }
            
            Logger.LogInfo("导出连续跟踪数据...");
            ContinuousTracker.ExportTrackingData();
        }
        
        /// <summary>
        /// 清空连续跟踪数据
        /// </summary>
        private void ClearContinuousData()
        {
            if (!continuousTrackerInitialized)
            {
                Logger.LogWarning("连续跟踪器未初始化");
                return;
            }
            
            ContinuousTracker.ClearData();
            Logger.LogInfo("🗑️ 连续跟踪数据已清空");
        }
        
        /// <summary>
        /// 检查自动化模式信号
        /// </summary>
        private void CheckAutoModeSignal()
        {
            string basePath = System.IO.Path.Combine(Application.dataPath, "..");
            string srcPath = System.IO.Path.Combine(basePath, "src");
            string dataPath = System.IO.Path.Combine(srcPath, "Data");
            string signalPath = System.IO.Path.Combine(dataPath, "auto_mode_signal.json");
            
            if (System.IO.File.Exists(signalPath))
            {
                try
                {
                    string jsonContent = System.IO.File.ReadAllText(signalPath);
                    var signalData = JsonUtility.FromJson<AutoModeSignal>(jsonContent);
                    
                    if (signalData.enable_auto_mode && !autoModeEnabled)
                    {
                        StartAutoMode(signalData);
                    }
                    else if (!signalData.enable_auto_mode && autoModeEnabled)
                    {
                        StopAutoMode();
                    }
                    
                    // 删除信号文件
                    System.IO.File.Delete(signalPath);
                }
                catch (System.Exception e)
                {
                    Logger.LogError($"❌ 读取自动化模式信号失败: {e.Message}");
                }
            }
        }
        
        /// <summary>
        /// 开始自动化模式
        /// </summary>
        private void StartAutoMode(AutoModeSignal signal)
        {
            autoModeEnabled = true;
            autoModeInitialized = false;
            hasStartedNewGame = false;
            waitingForMianScene = false;
            
            Logger.LogInfo($"🤖 自动化模式已启动 - 采集时长:{signal.duration}秒, 频率:{signal.frequency}Hz");
        }
        
        /// <summary>
        /// 停止自动化模式
        /// </summary>
        private void StopAutoMode()
        {
            autoModeEnabled = false;
            autoModeInitialized = false;
            hasStartedNewGame = false;
            waitingForMianScene = false;
            
            Logger.LogInfo("🤖 自动化模式已停止");
        }
        
        /// <summary>
        /// 处理自动化模式
        /// </summary>
        private void HandleAutoMode()
        {
            string currentScene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
            
            if (currentScene.ToLower().Contains("loader"))
            {
                HandleLoaderScene();
            }
            else if (currentScene.ToLower().Contains("mian"))
            {
                HandleMianScene();
            }
            else
            {
                Logger.LogInfo($"⏳ 等待场景切换... 当前场景: {currentScene}");
            }
        }
        
        /// <summary>
        /// 处理Loader场景
        /// </summary>
        private void HandleLoaderScene()
        {
            if (!hasStartedNewGame)
            {
                Logger.LogInfo("🎮 在Loader场景，尝试开始新游戏...");
                Logger.LogInfo("🎯 查找按钮路径: Canvas/Column/NewGame");
                
                // 尝试点击"开始新游戏"按钮
                if (TryClickStartNewGameButton())
                {
                    hasStartedNewGame = true;
                    waitingForMianScene = true;
                    mianSceneLoadStartTime = Time.time;
                    Logger.LogInfo("✅ 已点击开始新游戏按钮，等待场景切换...");
                }
                else
                {
                    Logger.LogWarning("⚠️ 未找到开始新游戏按钮，继续尝试...");
                    Logger.LogInfo("🔍 当前场景中的相关GameObject:");
                    GameObject[] allObjects = FindObjectsOfType<GameObject>();
                    foreach (var obj in allObjects)
                    {
                        if (obj.name.ToLower().Contains("canvas") || 
                            obj.name.ToLower().Contains("new") ||
                            obj.name.ToLower().Contains("game") ||
                            obj.name.ToLower().Contains("column"))
                        {
                            Logger.LogInfo($"   - {obj.name} (路径: {GetGameObjectPath(obj)})");
                        }
                    }
                }
            }
            else if (waitingForMianScene)
            {
                // 检查是否超时
                if (Time.time - mianSceneLoadStartTime > mianSceneLoadTimeout)
                {
                    Logger.LogError("❌ 等待Mian场景加载超时");
                    StopAutoMode();
                }
                else
                {
                    Logger.LogInfo("⏳ 等待Mian场景加载...");
                }
            }
        }
        
        /// <summary>
        /// 处理Mian场景
        /// </summary>
        private void HandleMianScene()
        {
            if (waitingForMianScene)
            {
                waitingForMianScene = false;
                Logger.LogInfo("✅ 已进入Mian场景，等待游戏完全加载...");
            }
            
            // 检查游戏是否完全加载
            if (IsGameFullyLoaded())
            {
                if (!autoModeInitialized)
                {
                    InitializeAutoMode();
                }
                else
                {
                    // 开始数据采集
                    StartAutoDataCollection();
                }
            }
            else
            {
                Logger.LogInfo("⏳ 等待游戏完全加载...");
            }
        }
        
        /// <summary>
        /// 尝试点击开始新游戏按钮
        /// </summary>
        private bool TryClickStartNewGameButton()
        {
            // 方法1: 通过完整路径查找按钮 (Canvas/Column/NewGame)
            GameObject newGameButton = GameObject.Find("Canvas/Column/NewGame");
            
            if (newGameButton != null)
            {
                Logger.LogInfo($"🎯 找到开始新游戏按钮: {newGameButton.name}");
                Logger.LogInfo($"🎯 按钮位置: {newGameButton.transform.position}");
                
                // 尝试通过UI按钮组件点击 (暂时禁用，因为Getting Over It可能使用旧版Unity)
                /*
                var button = newGameButton.GetComponent<UnityEngine.UI.Button>();
                if (button != null)
                {
                    Logger.LogInfo("🎯 通过UI按钮组件点击");
                    button.onClick.Invoke();
                    return true;
                }
                */
                
                // 备用方案：通过鼠标点击模拟
                Logger.LogInfo("🎯 尝试模拟鼠标点击按钮位置");
                // 这里可以添加鼠标点击逻辑
                return true;
            }
            
            // 方法2: 通过GameObject名称查找
            GameObject startButton = GameObject.Find("NewGame") ?? 
                                   GameObject.Find("StartButton") ?? 
                                   GameObject.Find("PlayButton");
            
            if (startButton != null)
            {
                Logger.LogInfo($"🎯 找到开始按钮: {startButton.name}");
                // 暂时禁用UI按钮点击，因为Getting Over It可能使用旧版Unity
                /*
                var button = startButton.GetComponent<UnityEngine.UI.Button>();
                if (button != null)
                {
                    button.onClick.Invoke();
                    return true;
                }
                */
            }
            
            Logger.LogWarning("⚠️ 未找到开始新游戏按钮");
            return false;
        }
        
        /// <summary>
        /// 获取GameObject的完整路径
        /// </summary>
        private string GetGameObjectPath(GameObject obj)
        {
            string path = obj.name;
            Transform parent = obj.transform.parent;
            while (parent != null)
            {
                path = parent.name + "/" + path;
                parent = parent.parent;
            }
            return path;
        }
        
        /// <summary>
        /// 检查游戏是否完全加载
        /// </summary>
        private bool IsGameFullyLoaded()
        {
            // 检查Player对象是否存在
            GameObject player = GameObject.Find("Player");
            if (player == null) return false;
            
            // 检查Mountain对象是否存在
            GameObject mountain = GameObject.Find("Mountain");
            if (mountain == null) return false;
            
            // 检查Player的Rigidbody2D是否已初始化
            var playerRb = player.GetComponent<Rigidbody2D>();
            if (playerRb == null) return false;
            
            // 检查是否在合理的位置（不是初始位置）
            if (player.transform.position.y < -10f) return false;
            
            return true;
        }
        
        /// <summary>
        /// 初始化自动化模式
        /// </summary>
        private void InitializeAutoMode()
        {
            Logger.LogInfo("🔧 初始化自动化模式...");
            
            // 初始化所有监控器
            InitializeMonitors();
            
            // 启用连续跟踪
            continuousTrackerEnabled = true;
            
            autoModeInitialized = true;
            Logger.LogInfo("✅ 自动化模式初始化完成");
        }
        
        /// <summary>
        /// 开始自动数据采集
        /// </summary>
        private void StartAutoDataCollection()
        {
            if (!isRecording)
            {
                Logger.LogInfo("🔄 开始自动数据采集...");
                
                // 开始记录
                ContinuousTracker.StartTracking();
                isRecording = true;
                
                // 发送响应信号
                SendAutoModeResponse("data_collection_started");
            }
        }
        
        /// <summary>
        /// 发送自动化模式响应
        /// </summary>
        private void SendAutoModeResponse(string response)
        {
            try
            {
                var responseData = new AutoModeResponse
                {
                    response = response,
                    scene_name = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name,
                    timestamp = Time.time
                };
                
                string jsonContent = JsonUtility.ToJson(responseData, true);
                string basePath = System.IO.Path.Combine(Application.dataPath, "..");
                string srcPath = System.IO.Path.Combine(basePath, "src");
                string dataPath = System.IO.Path.Combine(srcPath, "Data");
                string responsePath = System.IO.Path.Combine(dataPath, "auto_mode_response.json");
                
                System.IO.File.WriteAllText(responsePath, jsonContent);
                Logger.LogInfo($"📤 发送自动化响应: {response}");
            }
            catch (System.Exception e)
            {
                Logger.LogError($"❌ 发送自动化响应失败: {e.Message}");
            }
        }
        
        /// <summary>
        /// 检查是否在游戏场景中
        /// </summary>
        /// <returns>是否在游戏场景</returns>
        private bool IsInGameScene()
        {
            // 检查场景名称
            string sceneName = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
            if (sceneName.ToLower().Contains("main") || sceneName.ToLower().Contains("body"))
            {
                return true;
            }
            
            // 检查是否存在Mountain对象
            GameObject mountain = GameObject.Find("Mountain");
            return mountain != null;
        }
        
        /// <summary>
        /// 自动化模式信号数据结构
        /// </summary>
        [System.Serializable]
        public class AutoModeSignal
        {
            public bool enable_auto_mode;
            public float duration;
            public int frequency;
            public float timestamp;
        }
        
        /// <summary>
        /// 自动化模式响应数据结构
        /// </summary>
        [System.Serializable]
        public class AutoModeResponse
        {
            public string response;
            public string scene_name;
            public float timestamp;
        }
        
        /// <summary>
        /// 收集碰撞体数据
        /// </summary>
        private void CollectHitboxData()
        {
            Logger.LogInfo("开始收集Mountain碰撞体数据...");
            
            // 使用HitboxCollector收集所有碰撞体
            mountainColliders = HitboxCollector.GetAllMountainColliders();
            
            if (mountainColliders.Count > 0)
            {
                Logger.LogInfo($"成功收集到 {mountainColliders.Count} 个碰撞体");
                
                // 打印统计信息
                HitboxCollector.PrintColliderStatistics(mountainColliders);
                
                // 显示一些示例碰撞体
                ShowSampleColliders();
            }
            else
            {
                Logger.LogWarning("未找到任何碰撞体");
            }
        }
        
        /// <summary>
        /// 显示示例碰撞体信息
        /// </summary>
        private void ShowSampleColliders()
        {
            Logger.LogInfo("=== 示例碰撞体信息 ===");
            
            // 显示前5个碰撞体的基本信息
            int showCount = Mathf.Min(5, mountainColliders.Count);
            for (int i = 0; i < showCount; i++)
            {
                var info = mountainColliders[i];
                Logger.LogInfo($"[{i + 1}] {info.name} - 顶点数:{info.vertexCount} 位置:{info.position}");
            }
            
            if (mountainColliders.Count > 5)
            {
                Logger.LogInfo($"... 还有 {mountainColliders.Count - 5} 个碰撞体");
            }
        }
        
        /// <summary>
        /// 导出碰撞体数据
        /// </summary>
        private void ExportHitboxData()
        {
            if (mountainColliders == null || mountainColliders.Count == 0)
            {
                Logger.LogWarning("没有碰撞体数据可导出，请先按F8收集数据");
                return;
            }
            
            Logger.LogInfo("导出碰撞体数据到文件...");
            HitboxCollector.ExportCollidersToFile(mountainColliders);
        }
        
        /// <summary>
        /// 分析Player对象数据
        /// </summary>
        private void AnalyzePlayerData()
        {
            Logger.LogInfo("开始分析Player对象...");
            
            playerInfo = PlayerAnalyzer.AnalyzePlayer();
            
            if (playerInfo.name != "NOT_FOUND")
            {
                Logger.LogInfo($"成功分析Player对象: {playerInfo.name}");
                PlayerAnalyzer.PrintPlayerInfo(playerInfo);
            }
            else
            {
                Logger.LogWarning("未找到Player对象");
            }
        }
        
        /// <summary>
        /// 导出Player分析数据
        /// </summary>
        private void ExportPlayerAnalysis()
        {
            if (playerInfo.name == "NOT_FOUND" || string.IsNullOrEmpty(playerInfo.name))
            {
                Logger.LogWarning("没有Player分析数据可导出，请先按F10分析Player");
                return;
            }
            
            Logger.LogInfo("导出Player分析数据到文件...");
            PlayerAnalyzer.ExportPlayerAnalysis(playerInfo);
        }
        
        /// <summary>
        /// 获取收集到的碰撞体数据（供外部访问）
        /// </summary>
        /// <returns>碰撞体信息列表</returns>
        public List<HitboxCollector.ColliderInfo> GetCollectedColliders()
        {
            return mountainColliders ?? new List<HitboxCollector.ColliderInfo>();
        }
        
        /// <summary>
        /// 按名称查找碰撞体
        /// </summary>
        /// <param name="namePattern">名称模式</param>
        /// <returns>匹配的碰撞体列表</returns>
        public List<HitboxCollector.ColliderInfo> FindCollidersByName(string namePattern)
        {
            if (mountainColliders == null) return new List<HitboxCollector.ColliderInfo>();
            return HitboxCollector.FindCollidersByName(mountainColliders, namePattern);
        }
        
        void OnGUI()
        {
            // 只有在游戏界面才显示简单的状态信息
            if (IsInGameScene())
            {
                // 基础功能热键
                GUI.color = Color.cyan;
                GUI.Label(new Rect(10, 10, 800, 30), "F10: 分析Player | F11: 导出Player | F12: State监控 | F1: Action监控 | F2: 打印State | F3: 打印Action");
                
                // 连续跟踪功能热键
                GUI.color = Color.magenta;
                GUI.Label(new Rect(10, 35, 800, 30), "F6: 连续跟踪 | F7: 开始/停止记录 | F8: 导出连续数据 | F9: 清空数据 | F: 游戏采样频率");
                
                // 碰撞体数据状态
                if (mountainColliders != null && mountainColliders.Count > 0)
                {
                    GUI.color = Color.green;
                    GUI.Label(new Rect(10, 65, 300, 30), $"✅ 已收集 {mountainColliders.Count} 个碰撞体");
                }
                else
                {
                    GUI.color = Color.yellow;
                    GUI.Label(new Rect(10, 65, 300, 30), "⏳ 正在收集碰撞体数据...");
                }
                
                // Player分析状态
                if (!string.IsNullOrEmpty(playerInfo.name) && playerInfo.name != "NOT_FOUND")
                {
                    GUI.color = Color.green;
                    GUI.Label(new Rect(10, 95, 400, 30), $"✅ Player已分析: {playerInfo.name} ({playerInfo.components.Count}个组件)");
                }
                else if (playerAnalyzed)
                {
                    GUI.color = Color.red;
                    GUI.Label(new Rect(10, 95, 300, 30), "❌ 未找到Player对象");
                }
                else
                {
                    GUI.color = Color.yellow;
                    GUI.Label(new Rect(10, 95, 300, 30), "⏳ 正在搜索Player对象...");
                }
                
                // State监控状态
                if (stateMonitorInitialized)
                {
                    GUI.color = stateMonitorEnabled ? Color.green : Color.gray;
                    string stateStatus = stateMonitorEnabled ? "✅ State监控运行中" : "⏸️ State监控已暂停";
                    GUI.Label(new Rect(10, 125, 250, 30), stateStatus);
                    
                    if (stateMonitorEnabled)
                    {
                        GUI.color = Color.white;
                        GUI.Label(new Rect(270, 125, 400, 30), $"位置:({currentState.position.x:F1},{currentState.position.y:F1}) 速度:{currentState.movementSpeed:F1}");
                    }
                }
                else
                {
                    GUI.color = Color.yellow;
                    GUI.Label(new Rect(10, 125, 250, 30), "⏳ State监控未初始化");
                }
                
                // Action监控状态
                if (actionMonitorInitialized)
                {
                    GUI.color = actionMonitorEnabled ? Color.green : Color.gray;
                    string actionStatus = actionMonitorEnabled ? "✅ Action监控运行中" : "⏸️ Action监控已暂停";
                    GUI.Label(new Rect(10, 155, 250, 30), actionStatus);
                    
                    if (actionMonitorEnabled)
                    {
                        GUI.color = Color.white;
                        GUI.Label(new Rect(270, 155, 400, 30), $"控制器:{(currentAction.playerControlActive ? "激活" : "禁用")} 关节角度:{currentAction.mainHingeAngle:F1}°");
                    }
                }
                else
                {
                    GUI.color = Color.yellow;
                    GUI.Label(new Rect(10, 155, 250, 30), "⏳ Action监控未初始化");
                }
                
                // 连续跟踪状态
                if (continuousTrackerInitialized)
                {
                    GUI.color = continuousTrackerEnabled ? Color.green : Color.gray;
                    string trackerStatus = continuousTrackerEnabled ? "✅ 连续跟踪启用" : "⏸️ 连续跟踪禁用";
                    GUI.Label(new Rect(10, 185, 250, 30), trackerStatus);
                    
                    if (continuousTrackerEnabled)
                    {
                        GUI.color = isRecording ? Color.red : Color.yellow;
                        string recordStatus = isRecording ? "🔴 正在记录" : "⏸️ 已暂停";
                        GUI.Label(new Rect(270, 185, 150, 30), recordStatus);
                        
                        GUI.color = Color.white;
                        int dataCount = ContinuousTracker.GetDataPointCount();
                        GUI.Label(new Rect(430, 185, 200, 30), $"数据点: {dataCount}");
                    }
                }
                else
                {
                    GUI.color = Color.yellow;
                    GUI.Label(new Rect(10, 185, 250, 30), "⏳ 连续跟踪未初始化");
                }
                
                GUI.color = Color.white;
            }
        }
    }
}