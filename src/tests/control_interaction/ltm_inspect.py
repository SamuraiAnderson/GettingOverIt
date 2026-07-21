"""
libTAS 影片(.ltm)解包 + 输入表结构分析工具。

.ltm 本质是 tar.gz，内部是几个文本文件（libTAS 官方格式）：
  - config.ini      影片元数据（[General]：framerate、frame_count、mouse_support、nb_controllers…）
  - inputs          输入表：每行以 '|' 开头为一帧，按设备分段
  - annotations.txt 注释
  - editor.ini      输入编辑器信息（single input 名称/顺序、markers…）

inputs 每帧行按 '|' 分段，每段首字符标识设备：
  K  键盘：'|K' 后接若干十六进制 keysym，':' 分隔
  M  鼠标：'|M' 后接 'xpos:ypos:REF:BUTTONS'
           REF = 'A'(绝对) 或 'R'(相对)；BUTTONS = 5 个字符，按下为数字否则 '.'
  C<n> 手柄 n；T 变帧率(num:den)；F 帧标志…（本工具聚焦鼠标）

用途：GOI 的 TAS 用 libTAS 录制，鼠标存的是「操作系统光标屏幕坐标」，不是我们注入的
mouseInput(dx,dy)。本工具把鼠标轨迹拆出来、算逐帧位移，便于后续换算成我们的动作空间。

用法:
  python src/tests/control_interaction/ltm_inspect.py <path/to/movie.ltm>
  python src/tests/control_interaction/ltm_inspect.py <movie.ltm> --head 20 --dump-mouse logs/tas_mouse.csv
"""

from __future__ import annotations

import argparse
import gzip
import io
import sys
import tarfile
from pathlib import Path

# Windows 控制台默认 cp936，UTF-8 中文会乱码；强制 stdout 为 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


MOVIE_TEXT_FILES = ("config.ini", "inputs", "annotations.txt", "editor.ini")


def read_ltm(path: Path) -> dict[str, str]:
    """把 .ltm(tar.gz) 内的文本文件全部读出为 {成员名: 文本}。

    兼容两种封装：标准 tar.gz，以及个别导出的「裸 gzip 单文件」兜底。
    """
    raw = path.read_bytes()
    # 优先按 tar.gz 解
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tar:
            out: dict[str, str] = {}
            for m in tar.getmembers():
                if not m.isfile():
                    continue
                f = tar.extractfile(m)
                if f is None:
                    continue
                name = Path(m.name).name  # 去掉可能的目录前缀
                out[name] = f.read().decode("utf-8", errors="replace")
            if out:
                return out
    except (tarfile.TarError, OSError):
        pass
    # 兜底：可能是裸 gzip
    try:
        text = gzip.decompress(raw).decode("utf-8", errors="replace")
        return {"inputs": text}
    except OSError as e:
        raise ValueError(f"无法解析 .ltm（既非 tar.gz 也非 gzip）: {e}") from e


def parse_config(text: str) -> dict[str, str]:
    """极简 ini 解析：返回扁平 {key: value}（section 名作为前缀省略，key 唯一即可）。"""
    cfg: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith((";", "#", "[")):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def parse_mouse_line(section: str) -> dict | None:
    """解析一段 'Mxpos:ypos:REF:BUTTONS'（section 首字符 'M' 已含）。"""
    if not section.startswith("M"):
        return None
    body = section[1:]
    parts = body.split(":")
    if len(parts) < 3:
        return None
    try:
        x = int(parts[0])
        y = int(parts[1])
    except ValueError:
        return None
    ref = parts[2] if len(parts) > 2 else "A"
    buttons = parts[3] if len(parts) > 3 else "....."
    pressed = [i for i, c in enumerate(buttons) if c != "."]
    return {"x": x, "y": y, "ref": ref, "buttons": buttons, "pressed": pressed}


def parse_inputs(text: str) -> dict:
    """遍历 inputs 文本，抽取每帧的鼠标/键盘段，返回统计与逐帧鼠标轨迹。"""
    frames: list[dict] = []
    kb_frames = 0
    variable_fr = 0
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        sections = [s for s in line.split("|")[1:]]  # 去掉行首空串
        frame: dict = {"mouse": None}
        for sec in sections:
            if not sec:
                continue
            tag = sec[0]
            if tag == "M":
                frame["mouse"] = parse_mouse_line(sec)
            elif tag == "K":
                if sec[1:].strip():
                    kb_frames += 1
            elif tag == "T":
                variable_fr += 1
        frames.append(frame)
    return {"frames": frames, "kb_frames": kb_frames, "variable_fr_frames": variable_fr}


def summarize(cfg: dict, parsed: dict) -> list[str]:
    frames = parsed["frames"]
    mice = [f["mouse"] for f in frames if f["mouse"] is not None]
    lines: list[str] = []
    lines.append(f"总输入帧: {len(frames)}   含鼠标段的帧: {len(mice)}   含键盘输入的帧: {parsed['kb_frames']}")

    for key in ("frame_count", "framerate_num", "framerate_den", "mouse_support",
                "nb_controllers", "rerecord_count", "game_name", "md5_movie", "length_sec"):
        if key in cfg:
            lines.append(f"  config.{key} = {cfg[key]}")

    if not mice:
        lines.append("未发现鼠标输入段（可能 mouse_support=0）。")
        return lines

    refs = {m["ref"] for m in mice}
    xs = [m["x"] for m in mice]
    ys = [m["y"] for m in mice]
    lines.append(f"鼠标参考系: {sorted(refs)}  (A=绝对屏幕坐标, R=相对位移)")
    lines.append(f"  x 范围: [{min(xs)}, {max(xs)}]   y 范围: [{min(ys)}, {max(ys)}]")

    # 逐帧位移：绝对坐标取差分；相对坐标本身即位移
    if refs == {"R"}:
        dxs, dys = xs, ys
    else:
        dxs = [xs[i] - xs[i - 1] for i in range(1, len(xs))]
        dys = [ys[i] - ys[i - 1] for i in range(1, len(ys))]
    if dxs:
        adx = [abs(v) for v in dxs]
        ady = [abs(v) for v in dys]
        lines.append(f"  逐帧位移 |dx|: 最大 {max(adx)}  均值 {sum(adx)/len(adx):.2f}   "
                     f"|dy|: 最大 {max(ady)}  均值 {sum(ady)/len(ady):.2f}")

    # 按键统计（鼠标 5 键：0=左 1=中 2=右 3=滚上 4=滚下，按 libTAS 约定）
    btn_count = {}
    for m in mice:
        for b in m["pressed"]:
            btn_count[b] = btn_count.get(b, 0) + 1
    if btn_count:
        lines.append(f"  鼠标按键按下帧数: {dict(sorted(btn_count.items()))}")
    else:
        lines.append("  鼠标按键：全程未按下（GOI 只用移动，符合预期）")

    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="解包并分析 libTAS .ltm 影片的输入表")
    ap.add_argument("ltm", type=str, help=".ltm 文件路径")
    ap.add_argument("--head", type=int, default=10, help="打印前 N 行原始 inputs")
    ap.add_argument("--dump-mouse", type=str, default=None,
                    help="把逐帧鼠标 (frame,x,y,ref,dx,dy,buttons) 导出为 CSV")
    args = ap.parse_args()

    path = Path(args.ltm)
    if not path.exists():
        raise SystemExit(f"文件不存在: {path}")

    members = read_ltm(path)
    print("=" * 64)
    print(f".ltm: {path}  ({path.stat().st_size} bytes)")
    print(f"内部成员: {sorted(members.keys())}")
    print("=" * 64)

    cfg = parse_config(members.get("config.ini", ""))
    inputs_text = members.get("inputs", "")
    if not inputs_text:
        raise SystemExit("未找到 inputs 文本，无法分析。")

    parsed = parse_inputs(inputs_text)

    print("[结构摘要]")
    for ln in summarize(cfg, parsed):
        print(ln)

    print("\n[原始 inputs 前 %d 行]" % args.head)
    for ln in inputs_text.splitlines()[: args.head]:
        print("  " + ln)

    ann = members.get("annotations.txt", "").strip()
    if ann:
        print("\n[annotations.txt]")
        print("  " + ann.replace("\n", "\n  "))

    if args.dump_mouse:
        out = Path(args.dump_mouse)
        out.parent.mkdir(parents=True, exist_ok=True)
        mice_frames = [(i, f["mouse"]) for i, f in enumerate(parsed["frames"]) if f["mouse"]]
        with open(out, "w", encoding="utf-8") as fp:
            fp.write("frame,x,y,ref,dx,dy,buttons\n")
            prev = None
            for i, m in mice_frames:
                if m["ref"] == "R":
                    dx, dy = m["x"], m["y"]
                else:
                    dx = 0 if prev is None else m["x"] - prev["x"]
                    dy = 0 if prev is None else m["y"] - prev["y"]
                fp.write(f"{i},{m['x']},{m['y']},{m['ref']},{dx},{dy},{m['buttons']}\n")
                prev = m
        print(f"\n逐帧鼠标已导出: {out} ({len(mice_frames)} 行)")


if __name__ == "__main__":
    main()
