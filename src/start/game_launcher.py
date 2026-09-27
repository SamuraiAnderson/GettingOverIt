"""
游戏启动器
用于启动 Getting Over It 游戏
"""
from pathlib import Path
import ctypes
import logging
import subprocess
import sys

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "project.json"

# 任务视图里的「界面 1」和当前桌面共用同一个 Win32 桌面。
# 游戏要放到另一个 Win32 桌面，窗口才不会出现在界面 1。
_ISOLATED_DESKTOP = "GoiRuntime"
_DESKTOP_ACCESS = 0x01FF  # 当前用户创建窗口、枚举窗口所需的桌面权限


def _load_exe_path() -> Path:
    """从 project.json 读取游戏可执行文件路径"""
    try:
        import json
        with open(_CONFIG_PATH) as f:
            cfg = json.load(f)
        return Path(cfg["game"]["executable_path"])
    except Exception:
        return None


def desktop_path(name: str) -> str:
    """CreateProcess 使用的「窗口站\\桌面」名。不切换到该桌面。"""
    return f"WinSta0\\{name}"


class GameLauncher:
    """游戏启动器"""

    def __init__(self, game_exe: str = None):
        if game_exe is not None:
            self.game_exe = Path(game_exe)
        else:
            self.game_exe = _load_exe_path()

        if self.game_exe is None or not self.game_exe.exists():
            raise FileNotFoundError(f"游戏执行文件不存在: {self.game_exe}")

        self.game_root = self.game_exe.parent
        logger.info(f"游戏路径: {self.game_exe}")

    def launch(self, wait: bool = False, *, desktop: str | None = _ISOLATED_DESKTOP):
        """
        启动游戏。

        desktop 默认是 GoiRuntime：在当前窗口站新建该桌面，只把游戏进程放上去，
        不调用 SwitchDesktop。传 None 则在当前桌面启动。

        Returns:
            wait=False: 带 pid / wait / terminate / kill 的进程对象
            wait=True:  返回码 int
        """
        logger.info("启动游戏: %s", self.game_exe.name)

        try:
            if desktop and sys.platform == "win32":
                process = _start_on_desktop(self.game_exe, self.game_root, desktop)
                logger.info(
                    "游戏已在桌面 %s 启动 (PID: %d)，未切换当前桌面",
                    desktop_path(desktop), process.pid,
                )
            else:
                process = subprocess.Popen(
                    [str(self.game_exe)],
                    cwd=str(self.game_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                logger.info("游戏已启动 (PID: %d)", process.pid)

            if wait:
                returncode = process.wait()
                logger.info("游戏已退出 (返回码: %d)", returncode)
                return returncode
            return process

        except Exception as e:
            logger.error("启动游戏失败: %s", e)
            raise


class _STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("lpReserved", ctypes.c_wchar_p),
        ("lpDesktop", ctypes.c_wchar_p),
        ("lpTitle", ctypes.c_wchar_p),
        ("dwX", ctypes.c_uint32),
        ("dwY", ctypes.c_uint32),
        ("dwXSize", ctypes.c_uint32),
        ("dwYSize", ctypes.c_uint32),
        ("dwXCountChars", ctypes.c_uint32),
        ("dwYCountChars", ctypes.c_uint32),
        ("dwFillAttribute", ctypes.c_uint32),
        ("dwFlags", ctypes.c_uint32),
        ("wShowWindow", ctypes.c_uint16),
        ("cbReserved2", ctypes.c_uint16),
        ("lpReserved2", ctypes.c_void_p),
        ("hStdInput", ctypes.c_void_p),
        ("hStdOutput", ctypes.c_void_p),
        ("hStdError", ctypes.c_void_p),
    ]


class _PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", ctypes.c_void_p),
        ("hThread", ctypes.c_void_p),
        ("dwProcessId", ctypes.c_uint32),
        ("dwThreadId", ctypes.c_uint32),
    ]


class DesktopGameProcess:
    """CreateProcess 得到的游戏进程。持有桌面句柄，直到进程退出。"""

    def __init__(self, process_handle, pid: int, desktop_handle, desktop_name: str):
        self.pid = int(pid)
        self.desktop_handle = desktop_handle
        self.desktop_name = desktop_name
        self.returncode = None
        self._handle = process_handle
        self._args = desktop_name

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        if _kernel32().WaitForSingleObject(self._handle, 0) != 0:
            return None
        return self.wait(timeout=0)

    def wait(self, timeout=None):
        if self.returncode is not None:
            return self.returncode
        millis = 0xFFFFFFFF if timeout is None else max(0, int(timeout * 1000))
        result = _kernel32().WaitForSingleObject(self._handle, millis)
        if result == 0x102:
            raise subprocess.TimeoutExpired(self._args, timeout)
        if result == 0xFFFFFFFF:
            raise OSError(ctypes.get_last_error(), "WaitForSingleObject failed")
        code = ctypes.c_uint32()
        if not _kernel32().GetExitCodeProcess(self._handle, ctypes.byref(code)):
            raise OSError(ctypes.get_last_error(), "GetExitCodeProcess failed")
        self.returncode = int(code.value)
        self._close_desktop()
        return self.returncode

    def terminate(self):
        if self.poll() is None and not _kernel32().TerminateProcess(self._handle, 1):
            raise OSError(ctypes.get_last_error(), "TerminateProcess failed")

    def kill(self):
        self.terminate()

    def _close_desktop(self):
        handle, self.desktop_handle = self.desktop_handle, None
        if handle:
            _user32().CloseDesktop(handle)
        if self._handle:
            _kernel32().CloseHandle(self._handle)
            self._handle = None


def _user32():
    user32 = ctypes.windll.user32
    user32.CreateDesktopW.argtypes = [
        ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_void_p,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
    ]
    user32.CreateDesktopW.restype = ctypes.c_void_p
    user32.OpenDesktopW.argtypes = [
        ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32,
    ]
    user32.OpenDesktopW.restype = ctypes.c_void_p
    user32.CloseDesktop.argtypes = [ctypes.c_void_p]
    user32.CloseDesktop.restype = ctypes.c_int
    return user32


def _kernel32():
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateProcessW.argtypes = [
        ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_wchar_p,
        ctypes.POINTER(_STARTUPINFOW), ctypes.POINTER(_PROCESS_INFORMATION),
    ]
    kernel32.CreateProcessW.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.WaitForSingleObject.restype = ctypes.c_uint32
    kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel32.TerminateProcess.restype = ctypes.c_int
    kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    return kernel32


def _open_desktop(name: str):
    user32 = _user32()
    handle = user32.OpenDesktopW(name, 0, False, _DESKTOP_ACCESS)
    if handle:
        return handle
    handle = user32.CreateDesktopW(name, None, None, 0, _DESKTOP_ACCESS, None)
    if not handle:
        raise OSError(ctypes.get_last_error(), f"CreateDesktop failed: {name}")
    return handle


def _start_on_desktop(exe: Path, cwd: Path, desktop_name: str) -> DesktopGameProcess:
    if ctypes.sizeof(_STARTUPINFOW) != 104:
        raise RuntimeError(f"STARTUPINFOW size {ctypes.sizeof(_STARTUPINFOW)} != 104")
    desktop_handle = _open_desktop(desktop_name)
    startup = _STARTUPINFOW()
    startup.cb = ctypes.sizeof(startup)
    startup.lpDesktop = desktop_path(desktop_name)
    process_info = _PROCESS_INFORMATION()
    command = ctypes.create_unicode_buffer(f'"{exe}"')
    created = _kernel32().CreateProcessW(
        str(exe), command, None, None, False, 0, None, str(cwd),
        ctypes.byref(startup), ctypes.byref(process_info),
    )
    if not created:
        error = ctypes.get_last_error()
        _user32().CloseDesktop(desktop_handle)
        raise OSError(error, f"CreateProcess failed on {desktop_path(desktop_name)}")
    _kernel32().CloseHandle(process_info.hThread)
    return DesktopGameProcess(
        process_info.hProcess, process_info.dwProcessId, desktop_handle, desktop_name,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        launcher = GameLauncher()
        wait = "--wait" in sys.argv or "-w" in sys.argv
        current = "--current-desktop" in sys.argv
        launcher.launch(wait=wait, desktop=None if current else _ISOLATED_DESKTOP)
        if not wait:
            logger.info("游戏已在独立桌面运行（使用 --wait 等待退出，--current-desktop 回到当前桌面）")
    except Exception as e:
        logger.error("错误: %s", e)
        sys.exit(1)
