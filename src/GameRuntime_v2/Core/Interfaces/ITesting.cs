namespace GoiRuntime.Core.Interfaces
{
	public interface IDebugTool
	{
		string ToolName { get; }
		void Enable();
		void Disable();
		bool IsEnabled { get; }
		void OnGUI();
		void Update();
	}
}
