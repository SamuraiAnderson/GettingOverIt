"""短启动：游戏窗口只应出现在 GoiRuntime，不应出现在当前桌面（界面 1 所在的 Win32 桌面）。"""
import ctypes
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from start.game_launcher import GameLauncher

_ENUM = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_ssize_t)
_user32 = ctypes.windll.user32
_user32.EnumWindows.argtypes = [_ENUM, ctypes.c_ssize_t]
_user32.EnumDesktopWindows.argtypes = [ctypes.c_void_p, _ENUM, ctypes.c_ssize_t]
_user32.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
_user32.GetWindowThreadProcessId.restype = ctypes.c_uint32
_user32.GetForegroundWindow.restype = ctypes.c_void_p
_user32.IsWindowVisible.argtypes = [ctypes.c_void_p]
_user32.IsWindowVisible.restype = ctypes.c_int


def _pids_on(enum_call) -> set[int]:
    found: set[int] = set()
    owner = ctypes.c_uint32()

    @_ENUM
    def callback(hwnd, _lparam):
        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value and _user32.IsWindowVisible(hwnd):
            found.add(int(owner.value))
        return True

    enum_call(callback)
    return found


def _current_desktop_pids() -> set[int]:
    return _pids_on(lambda callback: _user32.EnumWindows(callback, 0))


def _desktop_pids(desktop_handle) -> set[int]:
    return _pids_on(lambda callback: _user32.EnumDesktopWindows(desktop_handle, callback, 0))


def _foreground_pid() -> int:
    hwnd = _user32.GetForegroundWindow()
    owner = ctypes.c_uint32()
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
    return int(owner.value)


def main() -> int:
    import subprocess
    listed = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq GettingOverIt.exe", "/NH"],
        capture_output=True, text=True, check=False,
    )
    if "GettingOverIt.exe" in listed.stdout:
        print("REFUSE an existing game is running", flush=True)
        return 2
    before = _foreground_pid()
    process = GameLauncher().launch()
    print(f"started pid={process.pid} desktop={process.desktop_name} foreground_before={before}", flush=True)
    deadline = time.monotonic() + 40
    stole = False
    on_isolated = False
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                print(f"FAIL process exited early code={process.returncode}", flush=True)
                return 1
            if process.pid in _current_desktop_pids():
                print("FAIL game window is on the current desktop", flush=True)
                return 1
            if _foreground_pid() == process.pid:
                stole = True
            if process.pid in _desktop_pids(process.desktop_handle):
                on_isolated = True
                break
            time.sleep(0.25)
        if stole:
            print("FAIL game became the foreground process", flush=True)
            return 1
        if not on_isolated:
            print("FAIL no visible game window on GoiRuntime within 40s", flush=True)
            return 1
        print(
            f"PASS pid={process.pid} window_on_GoiRuntime=1 "
            f"window_on_current_desktop=0 foreground_pid={_foreground_pid()}",
            flush=True,
        )
        return 0
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except Exception:
                process.kill()
                process.wait(timeout=15)
        print(f"stopped code={process.poll()}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
