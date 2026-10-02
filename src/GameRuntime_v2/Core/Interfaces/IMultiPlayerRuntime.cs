using System.Collections;
using UnityEngine;

namespace GoiRuntime.Core.Interfaces
{
    /// <summary>One batch of players. Backend owns physics, initialization and reset.</summary>
    public interface IMultiPlayerRuntime
    {
        int NumAgents { get; }
        int LastCollectSequence { get; }
        bool LastAnchorCollectOk { get; }
        IEnumerator Initialize();
        IEnumerator Reset();
        float[] ExecuteStep(float[] actions);
        float[] CollectAllStates();
        float[] Teleport(int agent, Vector2 position);
        void TakeNewSnapshot();
        void ConfigureInput(bool active, int mouseX, int mouseY);
        string CollectResetDiagnostics();
        string CollectObservationDiagnostics();
        string CollectCalibrationCapture();
        GameObject GetPlayer(int agent);
        GameObject[] GetVisualizationRoots();
        void SetCameraFree(bool enabled);
        void SetVisualization(bool enabled);
        void ExportColliders(string directory);
    }
}
