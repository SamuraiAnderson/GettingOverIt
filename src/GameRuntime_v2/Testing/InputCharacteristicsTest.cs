using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using GoiRuntime.Core.Interfaces;
using GoiRuntime.Core.Utilities;
using GoiRuntime.PlayerControl;

namespace GoiRuntime.Testing
{
	/// <summary>
	/// 输入特性测试
	/// 验证 mouseInput 帧同步和输入范围 [10, 100]
	/// </summary>
	public class InputCharacteristicsTest
	{
		#region 依赖服务
		
		private IGameControlService gameControl;
		private PlayerInputService inputService;
		private PlayerStateService stateService;
		
		#endregion
		
		#region 测试状态
		
		private bool isInitialized = false;
		private bool isTestRunning = false;
		private TestType currentTestType = TestType.None;
		
		// 帧同步测试状态
		private int frameSyncPhase = 0;
		private PlayerState frameSyncS0;
		private PlayerState frameSyncS1;
		private float frameSyncInputValue = 50f;
		
		// 范围测试状态
		private int rangeTestIndex = 0;
		private int rangeTestPhase = 0;
		private PlayerState rangeTestS0;
		private List<RangeTestResult> rangeTestResults = new List<RangeTestResult>();
		
		// 测试输入值序列
		private float[] testInputValues = { 0f, 5f, 10f, 15f, 30f, 50f, 70f, 80f, 100f, 120f };
		
		// 完整范围测试状态
		private int fullRangeTestIndex = 0;
		private int fullRangeTestPhase = 0;
		private TestAxis fullRangeCurrentAxis = TestAxis.X;
		private PlayerState fullRangeTestS0;
		private List<FullRangeTestResult> fullRangeTestResults = new List<FullRangeTestResult>();
		
		// 完整范围测试值（包含正负值）
		private float[] fullRangeTestValues = { -120f, -100f, -70f, -50f, -30f, -10f, -5f, 0f, 5f, 10f, 30f, 50f, 70f, 100f, 120f };
		
		#endregion
		
		#region 数据结构
		
		public enum TestType
		{
			None,
			FrameSync,
			RangeValidation,
			FullRangeValidation  // 新增：完整双轴正负值测试
		}
		
		public enum TestAxis
		{
			X,
			Y
		}
		
		public struct RangeTestResult
		{
			public float inputValue;
			public float deltaTipX;
			public float deltaTipY;
			public float deltaHandleX;
			public float deltaHandleY;
			public float deltaHammerAngle;
			public float totalDelta;
			public bool isEffective;
		}
		
		public struct FullRangeTestResult
		{
			public TestAxis axis;
			public float inputValue;
			public float deltaTipX;
			public float deltaTipY;
			public float deltaHandleX;
			public float deltaHandleY;
			public float deltaHammerAngle;
			public float totalDelta;
			public bool isEffective;
		}
		
		#endregion
		
		#region 初始化
		
		public bool Initialize(
			IGameControlService control,
			PlayerInputService input,
			PlayerStateService state)
		{
			gameControl = control;
			inputService = input;
			stateService = state;
			
			isInitialized = gameControl != null && 
			                inputService != null && inputService.IsReady &&
			                stateService != null && stateService.IsReady;
			
			if (isInitialized)
			{
				Debug.Log("✅ InputCharacteristicsTest 初始化成功");
			}
			else
			{
				Debug.LogWarning("⚠️ InputCharacteristicsTest 初始化失败：服务不可用");
			}
			
			return isInitialized;
		}
		
		#endregion
		
		#region 帧同步测试
		
		/// <summary>
		/// 开始帧同步测试
		/// </summary>
		public void StartFrameSyncTest(float inputValue = 50f)
		{
			if (!isInitialized)
			{
				Debug.LogError("❌ 测试未初始化");
				return;
			}
			
			if (isTestRunning)
			{
				Debug.LogWarning("⚠️ 已有测试在运行");
				return;
			}
			
			Debug.Log("=== 开始帧同步测试 ===");
			Debug.Log($"测试输入值: {inputValue}");
			
			isTestRunning = true;
			currentTestType = TestType.FrameSync;
			frameSyncPhase = 0;
			frameSyncInputValue = inputValue;
			
			// Phase 0: 暂停游戏
			gameControl.Pause();
			Debug.Log("Phase 0: 游戏已暂停");
		}
		
		/// <summary>
		/// 帧同步测试更新（每帧调用）
		/// 使用 StepFrames 让游戏正常运行几帧
		/// </summary>
		private void UpdateFrameSyncTest()
		{
			switch (frameSyncPhase)
			{
				case 0:
					// 等待暂停稳定
					if (gameControl.IsPaused && !gameControl.IsStepping)
					{
						// Phase 1: 记录初始状态
						frameSyncS0 = stateService.GetCurrentState();
						Debug.Log($"Phase 1: 记录初始状态 S0");
						Debug.Log($"  位置: ({frameSyncS0.playerX:F3}, {frameSyncS0.playerY:F3})");
						Debug.Log($"  Tip: ({frameSyncS0.tipX:F3}, {frameSyncS0.tipY:F3})");
						Debug.Log($"  锤子角度: {frameSyncS0.hammerAngle:F2}");
						
						// Phase 2: 注入输入
						inputService.SetMouseInput(new Vector2(frameSyncInputValue, 0f));
						Debug.Log($"Phase 2: 注入输入 mouseInput = ({frameSyncInputValue}, 0)");
						
						// Phase 3: 步进帧（让游戏正常运行）
						Debug.Log("Phase 3: 执行 StepFrames(2)");
						gameControl.StepFrames(2);
						
						frameSyncPhase = 1;
					}
					break;
					
				case 1:
					// 等待步进完成
					if (gameControl.IsPaused && !gameControl.IsStepping)
					{
						// Phase 4: 记录新状态
						frameSyncS1 = stateService.GetCurrentState();
						Debug.Log($"Phase 4: 记录新状态 S1");
						Debug.Log($"  位置: ({frameSyncS1.playerX:F3}, {frameSyncS1.playerY:F3})");
						Debug.Log($"  Tip: ({frameSyncS1.tipX:F3}, {frameSyncS1.tipY:F3})");
						Debug.Log($"  锤子角度: {frameSyncS1.hammerAngle:F2}");
						
						// 清零输入
						inputService.SetMouseInput(Vector2.zero);
						
						// Phase 5: 分析结果
						AnalyzeFrameSyncResult();
						
						// 测试完成
						isTestRunning = false;
						currentTestType = TestType.None;
					}
					break;
			}
		}
		
		/// <summary>
		/// 分析帧同步测试结果
		/// </summary>
		private void AnalyzeFrameSyncResult()
		{
			Debug.Log("\n=== 帧同步测试结果 ===");
			
			float deltaTipX = Mathf.Abs(frameSyncS1.tipX - frameSyncS0.tipX);
			float deltaTipY = Mathf.Abs(frameSyncS1.tipY - frameSyncS0.tipY);
			float deltaHandleX = Mathf.Abs(frameSyncS1.handleX - frameSyncS0.handleX);
			float deltaHandleY = Mathf.Abs(frameSyncS1.handleY - frameSyncS0.handleY);
			float deltaHammerAngle = Mathf.Abs(frameSyncS1.hammerAngle - frameSyncS0.hammerAngle);
			float deltaPlayerX = Mathf.Abs(frameSyncS1.playerX - frameSyncS0.playerX);
			float deltaPlayerY = Mathf.Abs(frameSyncS1.playerY - frameSyncS0.playerY);
			
			float totalDelta = deltaTipX + deltaTipY + deltaHandleX + deltaHandleY + 
			                   deltaHammerAngle + deltaPlayerX + deltaPlayerY;
			
			Debug.Log($"状态变化:");
			Debug.Log($"  ΔTip: ({deltaTipX:F6}, {deltaTipY:F6})");
			Debug.Log($"  ΔHandle: ({deltaHandleX:F6}, {deltaHandleY:F6})");
			Debug.Log($"  ΔHammerAngle: {deltaHammerAngle:F6}");
			Debug.Log($"  ΔPlayer: ({deltaPlayerX:F6}, {deltaPlayerY:F6})");
			Debug.Log($"  总变化量: {totalDelta:F6}");
			
			float threshold = 0.001f;
			if (totalDelta > threshold)
			{
				Debug.Log($"\n✅ 帧同步验证通过！");
				Debug.Log($"   mouseInput 在单帧内生效");
				Debug.Log($"   变化量 {totalDelta:F6} > 阈值 {threshold}");
			}
			else
			{
				Debug.Log($"\n❌ 帧同步验证失败");
				Debug.Log($"   变化量 {totalDelta:F6} <= 阈值 {threshold}");
				Debug.Log($"   可能原因: 输入值在死区内，或需要更多帧才能生效");
			}
		}
		
		#endregion
		
		#region 范围验证测试
		
		/// <summary>
		/// 开始输入范围验证测试
		/// </summary>
		public void StartRangeValidationTest()
		{
			if (!isInitialized)
			{
				Debug.LogError("❌ 测试未初始化");
				return;
			}
			
			if (isTestRunning)
			{
				Debug.LogWarning("⚠️ 已有测试在运行");
				return;
			}
			
			Debug.Log("=== 开始输入范围验证测试 ===");
			Debug.Log($"测试序列: [0, 5, 10, 15, 30, 50, 70, 80, 100, 120]");
			
			isTestRunning = true;
			currentTestType = TestType.RangeValidation;
			rangeTestIndex = 0;
			rangeTestPhase = 0;
			rangeTestResults.Clear();
			
			// 暂停游戏
			gameControl.Pause();
			Debug.Log("游戏已暂停，开始测试序列...");
		}
		
		/// <summary>
		/// 范围验证测试更新
		/// 使用 StepFrames 让游戏正常运行
		/// </summary>
		private void UpdateRangeValidationTest()
		{
			if (rangeTestIndex >= testInputValues.Length)
			{
				// 所有测试完成
				AnalyzeRangeValidationResults();
				ExportRangeValidationResults();
				
				isTestRunning = false;
				currentTestType = TestType.None;
				return;
			}
			
			float currentValue = testInputValues[rangeTestIndex];
			
			switch (rangeTestPhase)
			{
				case 0:
					// 等待暂停稳定
					if (gameControl.IsPaused && !gameControl.IsStepping)
					{
						// 记录初始状态
						rangeTestS0 = stateService.GetCurrentState();
						
						// 注入输入
						inputService.SetMouseInput(new Vector2(currentValue, 0f));
						Debug.Log($"[{rangeTestIndex + 1}/{testInputValues.Length}] 测试输入值: {currentValue}");
						
						// 步进 3 帧（让游戏正常运行）
						gameControl.StepFrames(3);
						rangeTestPhase = 1;
					}
					break;
					
				case 1:
					// 等待步进完成
					if (gameControl.IsPaused && !gameControl.IsStepping)
					{
						// 记录新状态
						PlayerState s2 = stateService.GetCurrentState();
						
						// 清零输入
						inputService.SetMouseInput(Vector2.zero);
						
						// 计算变化量
						RangeTestResult result = new RangeTestResult();
						result.inputValue = currentValue;
						result.deltaTipX = Mathf.Abs(s2.tipX - rangeTestS0.tipX);
						result.deltaTipY = Mathf.Abs(s2.tipY - rangeTestS0.tipY);
						result.deltaHandleX = Mathf.Abs(s2.handleX - rangeTestS0.handleX);
						result.deltaHandleY = Mathf.Abs(s2.handleY - rangeTestS0.handleY);
						result.deltaHammerAngle = Mathf.Abs(s2.hammerAngle - rangeTestS0.hammerAngle);
						result.totalDelta = result.deltaTipX + result.deltaTipY + 
						                    result.deltaHandleX + result.deltaHandleY + 
						                    result.deltaHammerAngle;
						result.isEffective = result.totalDelta > 0.01f;
						
						rangeTestResults.Add(result);
						
						Debug.Log($"  总变化: {result.totalDelta:F4} | 有效: {result.isEffective}");
						
						// 步进几帧让物理稳定（清零输入状态下）
						gameControl.StepFrames(5);
						rangeTestPhase = 2;
					}
					break;
					
				case 2:
					// 等待稳定步进完成
					if (gameControl.IsPaused && !gameControl.IsStepping)
					{
						// 下一个测试值
						rangeTestIndex++;
						rangeTestPhase = 0;
					}
					break;
			}
		}
		
		/// <summary>
		/// 分析范围验证结果
		/// </summary>
		private void AnalyzeRangeValidationResults()
		{
			Debug.Log("\n=== 输入范围验证结果 ===\n");
			
			Debug.Log($"{"输入值",-8} {"ΔTipX",-10} {"ΔTipY",-10} {"ΔHandle",-10} {"Δ角度",-10} {"总变化",-10} {"有效"}");
			Debug.Log(new string('-', 70));
			
			foreach (var r in rangeTestResults)
			{
				Debug.Log($"{r.inputValue,-8:F0} {r.deltaTipX,-10:F4} {r.deltaTipY,-10:F4} " +
				          $"{r.deltaHandleX,-10:F4} {r.deltaHammerAngle,-10:F4} " +
				          $"{r.totalDelta,-10:F4} {(r.isEffective ? "✓" : "✗")}");
			}
			
			// 分析关键点
			Debug.Log("\n=== 关键发现 ===\n");
			
			// 找死区边界
			float minEffective = float.MaxValue;
			float maxIneffective = 0f;
			
			foreach (var r in rangeTestResults)
			{
				if (r.isEffective && r.inputValue < minEffective)
					minEffective = r.inputValue;
				if (!r.isEffective && r.inputValue > maxIneffective)
					maxIneffective = r.inputValue;
			}
			
			Debug.Log($"死区上界: {maxIneffective}");
			Debug.Log($"最小有效值: {minEffective}");
			
			// 检查饱和点
			float delta100 = 0f;
			float delta120 = 0f;
			
			foreach (var r in rangeTestResults)
			{
				if (Mathf.Approximately(r.inputValue, 100f)) delta100 = r.totalDelta;
				if (Mathf.Approximately(r.inputValue, 120f)) delta120 = r.totalDelta;
			}
			
			if (delta100 > 0 && delta120 > 0)
			{
				float saturationDiff = Mathf.Abs(delta120 - delta100);
				bool isSaturated = saturationDiff < delta100 * 0.1f; // 10%容差
				
				Debug.Log($"\n饱和点分析:");
				Debug.Log($"  输入100效果: {delta100:F4}");
				Debug.Log($"  输入120效果: {delta120:F4}");
				Debug.Log($"  差异: {saturationDiff:F4} ({saturationDiff / delta100 * 100:F1}%)");
				Debug.Log($"  饱和: {(isSaturated ? "是 (100是饱和点)" : "否")}");
			}
			
			// 总结
			Debug.Log("\n=== 结论 ===\n");
			
			if (minEffective <= 10f)
			{
				Debug.Log($"✅ 最小有效值验证通过: {minEffective} <= 10");
			}
			else
			{
				Debug.Log($"⚠️ 最小有效值可能大于10: {minEffective}");
			}
			
			Debug.Log($"推荐有效范围: [{minEffective}, 100]");
		}
		
		/// <summary>
		/// 导出范围验证结果到CSV
		/// </summary>
		private void ExportRangeValidationResults()
		{
			try
			{
				string fileName = $"range_validation_{DateTime.Now:yyyyMMdd_HHmmss}.csv";
				string testResultsPath = Path.Combine(PathManager.DataRootPath, "TestResults");
				string filePath = Path.Combine(testResultsPath, fileName);
				
				PathManager.EnsureDirectory(Path.GetDirectoryName(filePath));
				
				StringBuilder csv = new StringBuilder();
				csv.AppendLine("input_value,delta_tipX,delta_tipY,delta_handleX,delta_handleY,delta_hammerAngle,total_delta,is_effective");
				
				foreach (var r in rangeTestResults)
				{
					csv.AppendLine($"{r.inputValue:F1},{r.deltaTipX:F6},{r.deltaTipY:F6}," +
					               $"{r.deltaHandleX:F6},{r.deltaHandleY:F6},{r.deltaHammerAngle:F6}," +
					               $"{r.totalDelta:F6},{r.isEffective}");
				}
				
				File.WriteAllText(filePath, csv.ToString());
				Debug.Log($"\n📁 结果已导出: {filePath}");
			}
			catch (Exception e)
			{
				Debug.LogError($"导出失败: {e.Message}");
			}
		}
		
		#endregion
		
		#region 完整范围验证测试（双轴正负值）
		
		/// <summary>
		/// 开始完整范围验证测试（X/Y轴，正负值）
		/// </summary>
		public void StartFullRangeValidationTest()
		{
			if (!isInitialized)
			{
				Debug.LogError("❌ 测试未初始化");
				return;
			}
			
			if (isTestRunning)
			{
				Debug.LogWarning("⚠️ 已有测试在运行");
				return;
			}
			
			Debug.Log("=== 开始完整范围验证测试 (X/Y轴，正负值) ===");
			Debug.Log($"测试值: [{string.Join(", ", Array.ConvertAll(fullRangeTestValues, v => v.ToString("F0")))}]");
			Debug.Log($"总测试数: {fullRangeTestValues.Length * 2} (X轴 + Y轴)");
			
			isTestRunning = true;
			currentTestType = TestType.FullRangeValidation;
			fullRangeTestIndex = 0;
			fullRangeTestPhase = 0;
			fullRangeCurrentAxis = TestAxis.X;
			fullRangeTestResults.Clear();
			
			// 暂停游戏
			gameControl.Pause();
			Debug.Log("游戏已暂停，开始测试序列...");
			Debug.Log("\n--- X 轴测试 ---");
		}
		
		/// <summary>
		/// 完整范围验证测试更新
		/// </summary>
		private void UpdateFullRangeValidationTest()
		{
			// 检查是否完成所有测试
			if (fullRangeCurrentAxis == TestAxis.Y && fullRangeTestIndex >= fullRangeTestValues.Length)
			{
				// 所有测试完成
				AnalyzeFullRangeResults();
				ExportFullRangeResults();
				
				isTestRunning = false;
				currentTestType = TestType.None;
				return;
			}
			
			// 切换到 Y 轴测试
			if (fullRangeCurrentAxis == TestAxis.X && fullRangeTestIndex >= fullRangeTestValues.Length)
			{
				Debug.Log("\n--- Y 轴测试 ---");
				fullRangeCurrentAxis = TestAxis.Y;
				fullRangeTestIndex = 0;
				fullRangeTestPhase = 0;
				
				// 等待几帧让物理稳定
				gameControl.StepFrames(10);
				return;
			}
			
			float currentValue = fullRangeTestValues[fullRangeTestIndex];
			
			switch (fullRangeTestPhase)
			{
				case 0:
					// 等待暂停稳定
					if (gameControl.IsPaused && !gameControl.IsStepping)
					{
						// 记录初始状态
						fullRangeTestS0 = stateService.GetCurrentState();
						
						// 根据轴注入输入
						Vector2 input = fullRangeCurrentAxis == TestAxis.X 
							? new Vector2(currentValue, 0f) 
							: new Vector2(0f, currentValue);
						
						inputService.SetMouseInput(input);
						
						string axisLabel = fullRangeCurrentAxis == TestAxis.X ? "X" : "Y";
						int totalIndex = fullRangeCurrentAxis == TestAxis.X 
							? fullRangeTestIndex + 1 
							: fullRangeTestValues.Length + fullRangeTestIndex + 1;
						int totalCount = fullRangeTestValues.Length * 2;
						
						Debug.Log($"[{totalIndex}/{totalCount}] {axisLabel}轴 = {currentValue:F0}");
						
						// 步进 3 帧
						gameControl.StepFrames(3);
						fullRangeTestPhase = 1;
					}
					break;
					
				case 1:
					// 等待步进完成
					if (gameControl.IsPaused && !gameControl.IsStepping)
					{
						// 记录新状态
						PlayerState s2 = stateService.GetCurrentState();
						
						// 清零输入
						inputService.SetMouseInput(Vector2.zero);
						
						// 计算变化量
						FullRangeTestResult result = new FullRangeTestResult();
						result.axis = fullRangeCurrentAxis;
						result.inputValue = currentValue;
						result.deltaTipX = Mathf.Abs(s2.tipX - fullRangeTestS0.tipX);
						result.deltaTipY = Mathf.Abs(s2.tipY - fullRangeTestS0.tipY);
						result.deltaHandleX = Mathf.Abs(s2.handleX - fullRangeTestS0.handleX);
						result.deltaHandleY = Mathf.Abs(s2.handleY - fullRangeTestS0.handleY);
						result.deltaHammerAngle = Mathf.Abs(s2.hammerAngle - fullRangeTestS0.hammerAngle);
						result.totalDelta = result.deltaTipX + result.deltaTipY + 
						                    result.deltaHandleX + result.deltaHandleY + 
						                    result.deltaHammerAngle;
						result.isEffective = result.totalDelta > 0.01f;
						
						fullRangeTestResults.Add(result);
						
						Debug.Log($"  总变化: {result.totalDelta:F4} | 有效: {(result.isEffective ? "✓" : "✗")}");
						
						// 步进让物理稳定
						gameControl.StepFrames(5);
						fullRangeTestPhase = 2;
					}
					break;
					
				case 2:
					// 等待稳定步进完成
					if (gameControl.IsPaused && !gameControl.IsStepping)
					{
						// 下一个测试值
						fullRangeTestIndex++;
						fullRangeTestPhase = 0;
					}
					break;
			}
		}
		
		/// <summary>
		/// 分析完整范围测试结果
		/// </summary>
		private void AnalyzeFullRangeResults()
		{
			Debug.Log("\n=== 完整范围验证结果 ===\n");
			
			// X 轴结果
			Debug.Log("【X 轴】");
			Debug.Log($"{"输入值",-8} {"ΔTip",-12} {"ΔHandle",-12} {"Δ角度",-10} {"总变化",-10} {"有效"}");
			Debug.Log(new string('-', 60));
			
			var xResults = fullRangeTestResults.FindAll(r => r.axis == TestAxis.X);
			foreach (var r in xResults)
			{
				float deltaTip = r.deltaTipX + r.deltaTipY;
				float deltaHandle = r.deltaHandleX + r.deltaHandleY;
				Debug.Log($"{r.inputValue,-8:F0} {deltaTip,-12:F4} {deltaHandle,-12:F4} " +
				          $"{r.deltaHammerAngle,-10:F4} {r.totalDelta,-10:F4} {(r.isEffective ? "✓" : "✗")}");
			}
			
			// Y 轴结果
			Debug.Log("\n【Y 轴】");
			Debug.Log($"{"输入值",-8} {"ΔTip",-12} {"ΔHandle",-12} {"Δ角度",-10} {"总变化",-10} {"有效"}");
			Debug.Log(new string('-', 60));
			
			var yResults = fullRangeTestResults.FindAll(r => r.axis == TestAxis.Y);
			foreach (var r in yResults)
			{
				float deltaTip = r.deltaTipX + r.deltaTipY;
				float deltaHandle = r.deltaHandleX + r.deltaHandleY;
				Debug.Log($"{r.inputValue,-8:F0} {deltaTip,-12:F4} {deltaHandle,-12:F4} " +
				          $"{r.deltaHammerAngle,-10:F4} {r.totalDelta,-10:F4} {(r.isEffective ? "✓" : "✗")}");
			}
			
			// 分析结论
			Debug.Log("\n=== 关键发现 ===\n");
			
			// X 轴分析
			AnalyzeAxisResults(xResults, "X");
			
			// Y 轴分析
			AnalyzeAxisResults(yResults, "Y");
			
			// 总结
			Debug.Log("\n=== 结论 ===\n");
			
			float xMinEffective = GetMinEffective(xResults);
			float xMaxEffective = GetMaxEffective(xResults);
			float yMinEffective = GetMinEffective(yResults);
			float yMaxEffective = GetMaxEffective(yResults);
			
			Debug.Log($"X 轴有效范围: [{-xMaxEffective}, {-xMinEffective}] ∪ [{xMinEffective}, {xMaxEffective}]");
			Debug.Log($"Y 轴有效范围: [{-yMaxEffective}, {-yMinEffective}] ∪ [{yMinEffective}, {yMaxEffective}]");
			Debug.Log($"\n推荐 RL 动作空间:");
			Debug.Log($"  mouseInput.x ∈ [-{xMaxEffective}, {xMaxEffective}]");
			Debug.Log($"  mouseInput.y ∈ [-{yMaxEffective}, {yMaxEffective}]");
		}
		
		private void AnalyzeAxisResults(List<FullRangeTestResult> results, string axisName)
		{
			// 正值分析
			var positiveResults = results.FindAll(r => r.inputValue > 0);
			var negativeResults = results.FindAll(r => r.inputValue < 0);
			
			float posMinEffective = float.MaxValue;
			float posMaxEffective = 0f;
			float negMinEffective = float.MaxValue;
			float negMaxEffective = 0f;
			
			foreach (var r in positiveResults)
			{
				if (r.isEffective)
				{
					if (r.inputValue < posMinEffective) posMinEffective = r.inputValue;
					if (r.inputValue > posMaxEffective) posMaxEffective = r.inputValue;
				}
			}
			
			foreach (var r in negativeResults)
			{
				float absValue = Mathf.Abs(r.inputValue);
				if (r.isEffective)
				{
					if (absValue < negMinEffective) negMinEffective = absValue;
					if (absValue > negMaxEffective) negMaxEffective = absValue;
				}
			}
			
			Debug.Log($"【{axisName} 轴分析】");
			Debug.Log($"  正值: 最小有效 = {posMinEffective}, 最大有效 = {posMaxEffective}");
			Debug.Log($"  负值: 最小有效 = -{negMinEffective}, 最大有效 = -{negMaxEffective}");
		}
		
		private float GetMinEffective(List<FullRangeTestResult> results)
		{
			float min = float.MaxValue;
			foreach (var r in results)
			{
				if (r.isEffective && Mathf.Abs(r.inputValue) < min && r.inputValue != 0)
					min = Mathf.Abs(r.inputValue);
			}
			return min == float.MaxValue ? 5f : min;
		}
		
		private float GetMaxEffective(List<FullRangeTestResult> results)
		{
			float max = 0f;
			foreach (var r in results)
			{
				if (r.isEffective && Mathf.Abs(r.inputValue) > max)
					max = Mathf.Abs(r.inputValue);
			}
			return max == 0f ? 100f : max;
		}
		
		/// <summary>
		/// 导出完整范围测试结果
		/// </summary>
		private void ExportFullRangeResults()
		{
			try
			{
				string fileName = $"full_range_validation_{DateTime.Now:yyyyMMdd_HHmmss}.csv";
				string testResultsPath = Path.Combine(PathManager.DataRootPath, "TestResults");
				string filePath = Path.Combine(testResultsPath, fileName);
				
				PathManager.EnsureDirectory(Path.GetDirectoryName(filePath));
				
				StringBuilder csv = new StringBuilder();
				csv.AppendLine("axis,input_value,delta_tipX,delta_tipY,delta_handleX,delta_handleY,delta_hammerAngle,total_delta,is_effective");
				
				foreach (var r in fullRangeTestResults)
				{
					csv.AppendLine($"{r.axis},{r.inputValue:F1},{r.deltaTipX:F6},{r.deltaTipY:F6}," +
					               $"{r.deltaHandleX:F6},{r.deltaHandleY:F6},{r.deltaHammerAngle:F6}," +
					               $"{r.totalDelta:F6},{r.isEffective}");
				}
				
				File.WriteAllText(filePath, csv.ToString());
				Debug.Log($"\n📁 结果已导出: {filePath}");
			}
			catch (Exception e)
			{
				Debug.LogError($"导出失败: {e.Message}");
			}
		}
		
		#endregion
		
		#region 更新循环
		
		/// <summary>
		/// 更新（每帧调用）
		/// </summary>
		public void Update()
		{
			if (!isTestRunning) return;
			
			switch (currentTestType)
			{
				case TestType.FrameSync:
					UpdateFrameSyncTest();
					break;
				case TestType.RangeValidation:
					UpdateRangeValidationTest();
					break;
				case TestType.FullRangeValidation:
					UpdateFullRangeValidationTest();
					break;
			}
		}
		
		/// <summary>
		/// 测试是否正在运行
		/// </summary>
		public bool IsTestRunning => isTestRunning;
		
		/// <summary>
		/// 停止当前测试
		/// </summary>
		public void StopTest()
		{
			if (isTestRunning)
			{
				Debug.Log("⏹️ 测试已停止");
				isTestRunning = false;
				currentTestType = TestType.None;
				inputService?.SetMouseInput(Vector2.zero);
			}
		}
		
		#endregion
	}
}

