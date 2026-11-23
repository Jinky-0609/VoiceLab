#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VoiceLab 启动器（双击 .cmd 会调用它）。

解决的问题：8765 被占用时（上次没关干净、或启动了两次），
uvicorn 只会在终端刷一行 WinError 10048，窗口一闪而过，用户不知道发生了什么。

行为：
  1. 端口空闲 -> 正常启动并打开浏览器
  2. 端口已被**另一个 VoiceLab 实例**占用 -> 提示并直接打开那个实例，不再重复启动
  3. 端口被其他程序占用 -> 自动往后找空闲端口（8766、8767…），照常用
"""
from __future__ import annotations

import json
import socket
import sys
import urllib.request
import webbrowser
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
DEFAULT_PORT = 8765
MARKER = "/api/config"


def _listening(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.35)
        return s.connect_ex((host, port)) == 0


def _is_voicelab(port: int) -> bool:
    """检查该端口上跑的是不是 VoiceLab（而不是别的程序）。"""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{MARKER}", timeout=1.2) as r:
            data = json.loads(r.read(4096).decode("utf-8", "replace"))
            return isinstance(data, dict) and "profiles" in data
    except Exception:
        return False


def _free_port(start: int, tries: int = 12) -> int:
    for p in range(start, start + tries):
        if not _listening(p):
            return p
    return start


def main() -> int:
    port = DEFAULT_PORT
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--port" and i + 1 < len(sys.argv) - 1:
            try:
                port = int(sys.argv[i + 2])
            except ValueError:
                pass
        elif arg.isdigit():
            port = int(arg)

    if _listening(port):
        if _is_voicelab(port):
            url = f"http://127.0.0.1:{port}"
            print(f"VoiceLab 已经在运行：{url}")
            print("已为你打开那个窗口。要重启请先关掉原来的黑窗口。")
            webbrowser.open(url)
            return 0
        new_port = _free_port(port + 1)
        print(f"端口 {port} 被其他程序占用，改用 {new_port}")
        port = new_port

    # 正常启动（webui.py 自己会打开浏览器）
    sys.argv = [str(LAB / "tools" / "webui.py"), "--port", str(port)]
    sys.path.insert(0, str(LAB / "tools"))
    import webui
    return webui.main()


if __name__ == "__main__":
    sys.exit(main())
