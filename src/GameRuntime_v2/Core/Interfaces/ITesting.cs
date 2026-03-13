using System;

namespace GoiRuntime.Core.Interfaces
{
	/// <summary>
	/// 测试报告
	/// </summary>
	public struct TestReport
	{
		public int totalTests;
		public int passedTests;
		public int failedTests;
		public string[] failedTestNames;
		public string summary;
		public float executionTime;
	}

	/// <summary>
	/// 单元测试接口
	/// </summary>
	public interface ITestCase
	{
		/// <summary>
		/// 测试名称
		/// </summary>
		string TestName { get; }

		/// <summary>
		/// 测试描述
		/// </summary>
		string Description { get; }

		/// <summary>
		/// 设置测试环境
		/// </summary>
		void Setup();

		/// <summary>
		/// 运行测试
		/// </summary>
		void Run();

		/// <summary>
		/// 清理测试环境
		/// </summary>
		void Teardown();

		/// <summary>
		/// 测试是否通过
		/// </summary>
		bool IsPassed { get; }

		/// <summary>
		/// 获取测试结果
		/// </summary>
		string GetResult();

		/// <summary>
		/// 获取错误消息（如果失败）
		/// </summary>
		string GetErrorMessage();
	}

	/// <summary>
	/// 测试运行器接口
	/// </summary>
	public interface ITestRunner
	{
		/// <summary>
		/// 注册测试用例
		/// </summary>
		void RegisterTest(ITestCase test);

		/// <summary>
		/// 运行所有测试
		/// </summary>
		void RunAll();

		/// <summary>
		/// 运行单个测试
		/// </summary>
		void RunSingle(string testName);

		/// <summary>
		/// 获取测试报告
		/// </summary>
		TestReport GetReport();

		/// <summary>
		/// 清除所有测试
		/// </summary>
		void ClearTests();
	}

	/// <summary>
	/// 调试工具接口
	/// </summary>
	public interface IDebugTool
	{
		/// <summary>
		/// 工具名称
		/// </summary>
		string ToolName { get; }

		/// <summary>
		/// 启用调试工具
		/// </summary>
		void Enable();

		/// <summary>
		/// 禁用调试工具
		/// </summary>
		void Disable();

		/// <summary>
		/// 是否启用
		/// </summary>
		bool IsEnabled { get; }

		/// <summary>
		/// GUI 显示（在 OnGUI 中调用）
		/// </summary>
		void OnGUI();

		/// <summary>
		/// 更新（在 Update 中调用）
		/// </summary>
		void Update();
	}
}

