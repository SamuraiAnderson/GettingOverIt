"""构造一个符合 libTAS 格式的合成 .ltm，用于验证 ltm_inspect.py（无法下载真实 GOI 影片时的替身）。"""
import io
import math
import tarfile
import time
from pathlib import Path

N = 12
config = (
    "[General]\n"
    f"frame_count={N}\n"
    "framerate_num=60\n"
    "framerate_den=1\n"
    "rerecord_count=120\n"
    "mouse_support=1\n"
    "nb_controllers=0\n"
    "game_name=GettingOverIt\n"
    "md5_movie=deadbeefdeadbeefdeadbeefdeadbeef\n"
)

lines = []
for i in range(N):
    x = 480 + int(60 * math.sin(i / 2.0))       # 绝对屏幕坐标(模拟画圈)
    y = 270 + int(40 * math.cos(i / 2.0))
    buttons = "1...." if i in (4, 5) else "....."  # 第4/5帧按下左键，测按键统计
    lines.append(f"|K|M{x}:{y}:A:{buttons}")
inputs_text = "\n".join(lines) + "\n"

annotations = "Synthetic sample for ltm_inspect validation (not a real run).\n"
editor = "[input_names]\n[markers]\n[nondraw_frames]\n"

members = {
    "config.ini": config,
    "inputs": inputs_text,
    "annotations.txt": annotations,
    "editor.ini": editor,
}

out = Path(__file__).parent / "sample_synthetic.ltm"
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    for name, text in members.items():
        data = text.encode("utf-8")
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        info.mtime = int(time.time())
        tar.addfile(info, io.BytesIO(data))
out.write_bytes(buf.getvalue())
print(f"wrote {out} ({out.stat().st_size} bytes)")
