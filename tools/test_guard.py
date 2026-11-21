#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""semantic_guard 回归测试：真实乱音样本必须拦、正常样本不能误伤。

    .venv-tts/Scripts/python.exe tools/test_guard.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "tools"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import semantic_guard as sg  # noqa: E402

fails = 0


def expect(name, cond, detail=""):
    global fails
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        fails += 1


def load(p):
    d, sr = sf.read(p, dtype="float32", always_2d=True)
    return d[:, 0], sr


print("[1] 历史乱音样本（日文参考→中文目标，28字生成52.9s）必须被拦")
p = LAB / "out" / "zs_zh_noguard.wav"
if p.is_file():
    a, sr = load(p)
    r = sg.analyze(a, sr, nchars=28)
    expect("length_explosion 命中", "length_explosion" in r, str(r))
    expect("periodic_repetition 命中", "periodic_repetition" in r, str(r))
else:
    print("  (样本已不存在，跳过)")

print("[2] 今天的正常合成必须放行")
p = LAB / "out" / "guard_pass.wav"
if p.is_file():
    a, sr = load(p)
    r = sg.analyze(a, sr, nchars=17)
    expect("干净音频 0 违规", r == [], str(r))

print("[3] 历史参考音频 80 个不能误伤（只测规则2/3：给足长度预算）")
n_bad = 0
worst = ""
for f in sorted(glob.glob(str(LAB / "data" / "refs" / "*.wav")) +
                glob.glob(str(LAB / "out" / "*.wav"))):
    if "noguard" in f or "guard_pass" in f:
        continue
    try:
        a, sr = load(f)
    except Exception:
        continue
    r = sg.analyze(a, sr, nchars=max(4, int(len(a) / sr)))  # 预算 0.8s/字，宽松
    if r:
        n_bad += 1
        worst = f"{Path(f).name}: {r}"
expect("误伤数 == 0", n_bad == 0, worst)

print("[4] 合成用例：错误量纲 / 饱和 / 完美循环")
x = (np.random.RandomState(0).randn(48000) * 8000).astype(np.int16)  # 忘转 float
expect("int16 量纲 -> bad_scale_audio", "bad_scale_audio" in sg.analyze(x.astype(np.float32), 24000, 5))
sat = np.ones(24000 * 3, dtype=np.float32) * 0.95                      # 满量程
expect("饱和方波 -> saturated_audio", "saturated_audio" in sg.analyze(sat, 24000, 6))
tone = np.sin(2 * np.pi * 220 * np.arange(24000 * 6) / 24000).astype(np.float32)
expect("纯音循环 -> periodic_repetition", "periodic_repetition" in sg.analyze(tone, 24000, 100))
nan = np.full(24000 * 3, np.nan, dtype=np.float32)
expect("NaN -> non_finite_audio", "non_finite_audio" in sg.analyze(nan, 24000, 5))

print(f"\n{'== 全部通过 ==' if fails == 0 else f'== {fails} 项失败 =='}")
sys.exit(1 if fails else 0)
