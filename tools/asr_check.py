#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ASR 回环校验：把合成音频再识别成文字，与原稿比对（本项目自带耳朵）。

判定用**拼音相似度**为主：语音合成的目标是"听得懂"，不是"字形一致"。
实测「解耦」被识别成「截藕」——字形编辑距离 0.727（会误判为失败），
而拼音 jiě ǒu 完全相同，这才是正确结论。

用法：
    .venv-asr/Scripts/python.exe tools/asr_check.py out/x.wav "要合成的文本"
    .venv-asr/Scripts/python.exe tools/asr_check.py out/x.wav "文本" --json   # 机器可读
退出码：0 = PASS，1 = FAIL（可用于流水线卡关）。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "tools"))
try:
    import runtime_paths
    runtime_paths.ensure_env()
except Exception:
    pass

THRESHOLD = 0.85
_PUNCT = re.compile(r"[\s，。、！？；：,.!?;:\"'“”‘’（）()【】\[\]{}~～…·—-]+")


def _norm(s: str) -> str:
    return _PUNCT.sub("", s or "")


def edit_sim(a: str, b: str) -> float:
    """字符级相似度（1 - 归一化编辑距离）。"""
    if not a:
        return 0.0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return 1.0 - prev[lb] / la


def pinyin_seq(s: str) -> str:
    """汉字转无声调拼音后比较：同音字误识别不该算合成失败。"""
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError:      # 环境缺包时退回字符级
        return _norm(s)
    return " ".join(lazy_pinyin(_norm(s), style=Style.NORMAL, errors="default"))


def score(expect: str, got: str) -> dict:
    e, g = _norm(expect), _norm(got)
    char_sim = edit_sim(e, g)
    pin_sim = edit_sim(pinyin_seq(expect), pinyin_seq(got))
    return {
        "char_sim": round(char_sim, 3),
        "pinyin_sim": round(pin_sim, 3),
        # 拼音为主、字形为辅：两者取高，避免纯英文文本被拼音逻辑拖累
        "verdict": "PASS" if max(pin_sim, char_sim) >= THRESHOLD else "FAIL",
    }


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--json"]
    as_json = "--json" in sys.argv[1:]
    if len(args) < 2:
        sys.exit("用法: asr_check.py <wav> <原稿文本> [--json]")
    wav, expect = args[0], args[1]

    import numpy as np
    import soundfile as sf
    import torch
    from qwen_asr import Qwen3ASRModel

    model_dir = LAB / "models" / "asr" / "Qwen3-ASR-0.6B"
    if not (model_dir / "model.safetensors").is_file():
        sys.exit(f"[FATAL] 缺 Qwen3-ASR 权重: {model_dir}\n"
                 f"        跑 tools/fetch_qwen_asr.py 下载。")
    m = Qwen3ASRModel.from_pretrained(str(model_dir), local_files_only=True,
                                      dtype=torch.bfloat16, device_map="cuda",
                                      max_inference_batch_size=1, max_new_tokens=256,
                                      attn_implementation="sdpa")
    audio, sr = sf.read(wav, dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    res = m.transcribe(audio=(np.ascontiguousarray(audio, dtype=np.float32), sr),
                       language=None, context="")     # 语种交给模型自己判
    got = (getattr(res[0], "text", "") or "").strip() if res else ""
    r = score(expect, got)

    if as_json:
        print(json.dumps({"file": wav, "expect": expect, "asr": got, **r}, ensure_ascii=False))
    else:
        print(f"原稿: {expect}")
        print(f"ASR : {got or '(空)'}")
        print(f"拼音相似度 {r['pinyin_sim']:.3f} / 字形相似度 {r['char_sim']:.3f}"
              f"（阈值 {THRESHOLD}） -> {r['verdict']}")
    return 0 if r["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
