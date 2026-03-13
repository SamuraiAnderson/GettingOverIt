using UnityEngine;

namespace GoiRuntime.Core.Interfaces
{
	public interface IGameControlService
	{
		bool IsPaused { get; }
		float TimeScale { get; }
		bool IsStepping { get; }
		void Pause();
		void Resume();
		void TogglePause();
		void SetTimeScale(float scale);
		void StepFrames(int frameCount = 1);
		void OnLateUpdate();
		string GetStatusInfo();
	}
}
