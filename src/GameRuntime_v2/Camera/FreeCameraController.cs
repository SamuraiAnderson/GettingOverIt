using UnityEngine;
using System.Collections.Generic;

namespace GoiRuntime.CameraControl
{
	/// <summary>
	/// 自由相机控制器：解除原相机跟随锁定，允许滚轮缩放和拖动平移。
	/// 挂在 Camera.main 上，通过 SetFreeMode 开关切换。
	/// </summary>
	public class FreeCameraController : MonoBehaviour
	{
		public float zoomSpeed = 5f;
		public float dragSpeed = 1f;
		public float minZoom   = 1f;
		public float maxZoom   = 200f;
		public float initialZoom = 80f;

		private bool _freeMode;
		private List<MonoBehaviour> _disabledScripts = new List<MonoBehaviour>();

		private UnityEngine.Camera _cam;
		private bool _dragging;
		private Vector3 _dragOrigin;

		void Awake()
		{
			_cam = GetComponent<UnityEngine.Camera>();
			enabled = false;
		}

		/// <summary>
		/// 启用/禁用自由相机模式。
		/// 启用时禁用 Camera.main 上的所有其他脚本（排除自身和 ColliderVisualizer），
		/// 接管相机位置和缩放；禁用时恢复原有脚本。
		/// </summary>
		public void SetFreeMode(bool enable)
		{
			if (enable == _freeMode) return;
			_freeMode = enable;
			enabled = enable;

			if (enable)
			{
				_disabledScripts.Clear();
				var allScripts = GetComponents<MonoBehaviour>();
				foreach (var script in allScripts)
				{
					if (script == this) continue;
					if (script.GetType().Name == "ColliderVisualizer") continue;
					if (script.enabled)
					{
						script.enabled = false;
						_disabledScripts.Add(script);
					}
				}
				if (_cam.orthographic)
					_cam.orthographicSize = initialZoom;
				else
					transform.position = new Vector3(transform.position.x, transform.position.y, -initialZoom);

				Debug.Log($"[FreeCameraController] 自由相机已启用，禁用了 {_disabledScripts.Count} 个原相机脚本，zoom={initialZoom}");
			}
			else
			{
				foreach (var script in _disabledScripts)
				{
					if (script != null) script.enabled = true;
				}
				Debug.Log($"[FreeCameraController] 自由相机已禁用，恢复了 {_disabledScripts.Count} 个原相机脚本");
				_disabledScripts.Clear();
				_dragging = false;
			}
		}

		void LateUpdate()
		{
			if (!_freeMode || _cam == null) return;

			HandleZoom();
			HandleDrag();
		}

		private void HandleZoom()
		{
			float scroll = Input.mouseScrollDelta.y;
			if (Mathf.Abs(scroll) < 0.01f) return;

			if (_cam.orthographic)
			{
				_cam.orthographicSize -= scroll * zoomSpeed;
				_cam.orthographicSize = Mathf.Clamp(_cam.orthographicSize, minZoom, maxZoom);
			}
			else
			{
				Vector3 pos = transform.position;
				pos.z += scroll * zoomSpeed;
				transform.position = pos;
			}
		}

		private void HandleDrag()
		{
			if (Input.GetMouseButtonDown(0))
			{
				_dragging = true;
				_dragOrigin = _cam.ScreenToWorldPoint(Input.mousePosition);
			}

			if (Input.GetMouseButtonUp(0))
			{
				_dragging = false;
			}

			if (_dragging && Input.GetMouseButton(0))
			{
				Vector3 current = _cam.ScreenToWorldPoint(Input.mousePosition);
				Vector3 delta = _dragOrigin - current;
				transform.position += new Vector3(delta.x, delta.y, 0f);
			}
		}
	}
}
