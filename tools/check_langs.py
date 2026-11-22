#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""语言可用性自检：逐语言告诉你能不能合成、缺什么、怎么装。

    .venv-tts/Scripts/python.exe tools/check_langs.py

背景（2026-11-23 实测三处故障）：
  * 英文：nltk 数据缺失时**不报错，直接产出静音音频**（最隐蔽）
  * 粤语：缺 ToJyutping
  * 韩文：缺 jamo / ko_pron / g2pk2 / eunjeon（Windows 上 mecab 的替代）
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "tools"))

import runtime_paths  # noqa: E402

RUNTIME_DEPS = {
    "粤语": [("ToJyutping", "ToJyutping", "粤语注音（Jyutping）")],
    "韩文": [("jamo", "jamo", "韩文字母分解"),
             ("ko_pron", "ko_pron", "韩文发音规则"),
             ("g2pk2", "g2pk2", "韩文 G2P"),
             ("eunjeon", "eunjeon", "Windows 上的 mecab 绑定（韩文形态素）")],
    "日文": [("pyopenjtalk", "pyopenjtalk", "日文前端")],
    "中文": [("jieba", "jieba", "中文分词"),
             ("pypinyin", "pypinyin", "中文注音")],
}


def ok(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


def main() -> int:
    runtime_paths.ensure_env()
    print("=" * 58)
    print("VoiceLab 语言可用性自检")
    print("=" * 58)

    # 1) nltk（英文）
    nltk_ok, nltk_msg = True, ""
    try:
        import nltk_fix
        missing = nltk_fix.verify(runtime_paths.NLTK_DIR)
        if missing:
            nltk_ok = False
            nltk_msg = f"缺 {missing}"
    except Exception as e:
        nltk_ok = False
        nltk_msg = f"{type(e).__name__}: {e}"

    print(f"\n[英文] {'可用' if nltk_ok else '不可用 —— ' + nltk_msg}")
    if not nltk_ok:
        print("       修复：python tools/nltk_fix.py")
        print("       注意：缺 nltk 时英文**不会报错**，只会生成静音音频。")

    # 2) 各语言 python 依赖
    all_ok = nltk_ok
    for lang, deps in RUNTIME_DEPS.items():
        bad = [(m, d, why) for m, d, why in deps if not ok(m)]
        if not bad:
            print(f"\n[{lang}] 可用")
        else:
            all_ok = False
            print(f"\n[{lang}] 不可用 —— 缺 {len(bad)} 个依赖")
            for mod, pkg, why in bad:
                print(f"       · {pkg:12s} ({why})")
            pkgs = " ".join(p for _, p, _ in bad)
            print(f"       修复：uv pip install -p .venv-tts/Scripts/python.exe {pkgs}")

    print("\n" + "=" * 58)
    print("全部语言可用" if all_ok else "部分语言不可用（见上）")
    print("提示：v3 模型跨语言合成会概率性语义坍缩，建议用目标语言的参考音频。")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
