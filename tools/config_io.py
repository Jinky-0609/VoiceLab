#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""读取 / 引导 VoiceLab 的 config.json。

config.json 含各人音色档案（声音身份数据），**不入库**；
仓库只带 config.example.json 模板。任何工具都通过本模块拿配置，
第一次运行时若缺失就自动从模板复制一份，避免各脚本各自 sys.exit。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT / "config.json"
EXAMPLE_PATH = PROJECT / "config.example.json"


def load(verbose: bool = True) -> dict:
    if not CONFIG_PATH.is_file():
        if not EXAMPLE_PATH.is_file():
            sys.exit(f"[FATAL] 找不到 {CONFIG_PATH}，也没有 {EXAMPLE_PATH.name}")
        CONFIG_PATH.write_text(EXAMPLE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        if verbose:
            print(f"[引导] 已生成 {CONFIG_PATH.name}（来自 config.example.json）\n"
                  f"       参考音频放 data/refs/，音色档案写进 profiles", flush=True)
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
