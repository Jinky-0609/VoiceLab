#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""参数化情绪：不依赖任何情绪素材，用语速 + 音高 + 力度表达情绪倾向。

    .venv-tts/Scripts/python.exe tools/prosody.py -f out/x.wav -e 开心

## 为什么要有这个

原有的"情绪参考"要求每种情绪都提供一段音频素材，门槛太高。
本模块走另一条路：**在合成后的音频上做韵律变换**，因此：
  * 不需要任何素材
  * **对所有音色都有效**——零样本、微调音色一视同仁
    （因为它作用在后处理，不知道也不关心音色是怎么来的）
  * 可叠加在素材情绪之上：先融合素材的情绪，再用参数微调

## 原理与边界（必须清楚）

| 维度 | 手段 | 听感 |
|---|---|---|
| 语速 | 时间伸缩（保音高） | 快=急/兴奋，慢=沉/低落 |
| 音高 | 变调（保时长） | 高=激动/年轻，低=压抑/严肃 |
| 力度 | 增益 + 轻微压缩 | 强=有冲击，弱=虚弱/耳语 |

**能力上限**：这改变的是"怎么说"，不是"说什么情绪"。
它能做出可听辨的激动/低落，但做不出"哭笑不得"这种复杂情绪。
真正的语义级情绪仍需素材（或未来的情感条件模型）。
"""
from __future__ import annotations

import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "tools"))

# 预设：语义名 -> (语速倍率, 半音, 增益dB, 说明)
# 数值都是相对中性（1.0 / 0 / 0dB）的偏移，经试听调整过保守区间。
PRESETS = {
    "中性":    (1.00,  0.0,  0.0, "不做处理"),
    "开心":    (1.08, +1.2, +1.5, "略快、略高、稍亮"),
    "激动":    (1.15, +2.0, +2.5, "明显加快、升高、加大"),
    "生气":    (1.12, +1.5, +3.0, "加快、略高、力度强"),
    "悲伤":    (0.88, -1.8, -2.0, "放慢、降低、变轻"),
    "温柔":    (0.92, -0.8, -1.0, "稍慢、稍低、柔和"),
    "严肃":    (0.95, -2.5, +1.0, "稍慢、明显降低、保持力度"),
    "耳语":    (0.96, -1.0, -7.0, "接近耳语（大幅降增益）"),
    "疲惫":    (0.85, -2.2, -3.0, "很慢、低、虚弱"),
}


def available() -> list[str]:
    return list(PRESETS)


def describe(name: str) -> str:
    p = PRESETS.get(name)
    return p[3] if p else ""


def apply(audio, sr: int, emotion: str, strength: float = 1.0):
    """对音频应用参数化情绪。返回处理后的 float32 波形。

    strength: 0~1.5，控制变换幅度（0=不处理，1=预设原值）。
    """
    import numpy as np

    p = PRESETS.get(emotion)
    if p is None or emotion == "中性":
        return np.asarray(audio, dtype=np.float32).ravel()

    speed, semitone, gain_db, _ = p
    s = max(0.0, min(1.5, float(strength)))
    if s == 0:
        return np.asarray(audio, dtype=np.float32).ravel()

    # 从中性插值到预设值
    speed = 1.0 + (speed - 1.0) * s
    semitone = semitone * s
    gain_db = gain_db * s

    x = np.asarray(audio, dtype=np.float32).ravel()
    if len(x) < 256:
        return x

    # 1) 变调（保时长）
    if abs(semitone) > 0.05:
        try:
            import librosa
            x = librosa.effects.pitch_shift(y=x, sr=sr, n_steps=semitone)
        except Exception as e:
            print(f"[韵律] 变调失败，跳过: {type(e).__name__}", flush=True)

    # 2) 变速（保音高）
    if abs(speed - 1.0) > 0.005:
        try:
            import librosa
            x = librosa.effects.time_stretch(y=x, rate=speed)
        except Exception as e:
            print(f"[韵律] 变速失败，跳过: {type(e).__name__}", flush=True)

    # 3) 增益。用"乘增益 + 归一化"而不是 tanh 软限幅——
    #    后者会把所有样本压扁，实测峰值从 0.98 掉到 0.56，音量明显变小。
    if abs(gain_db) > 0.05:
        x = x * (10 ** (gain_db / 20))

    # 4) 统一归一化到 -1.5 dBFS 附近，既保留情绪带来的力度差，
    #    又不削顶。之前的实现直接归一化到 0.99 会抹平"生气更响"的差异，
    #    所以这里按情绪类型给不同目标电平。
    # librosa 的 pitch_shift/time_stretch 都会改变峰值（实测能掉到 0.4），
    # 所以必须无条件归一化到目标电平，否则"情绪越强音量越小"——完全反直觉。
    target = 0.90 if gain_db > 0.5 else (0.72 if gain_db < -2 else 0.85)
    peak = float(np.abs(x).max())
    if peak > 1e-6:
        x = x * (target / peak)
    return x.astype(np.float32)


def apply_file(src: str | Path, dst: str | Path, emotion: str,
               strength: float = 1.0) -> Path:
    import soundfile as sf
    a, sr = sf.read(str(src), dtype="float32", always_2d=True)
    mono = a.mean(axis=1)
    out = apply(mono, sr, emotion, strength)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), out, sr, subtype="PCM_16")
    return Path(dst)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="参数化情绪（不需要素材）")
    ap.add_argument("--list", action="store_true", help="列出预设")
    ap.add_argument("-f", "--file", help="输入音频")
    ap.add_argument("-o", "--out", help="输出（默认 _<emotion>.wav）")
    ap.add_argument("-e", "--emotion", help="情绪预设名")
    ap.add_argument("--strength", type=float, default=1.0, help="0~1.5 强度")
    args = ap.parse_args()

    if args.list or not args.file:
        print("可用预设（强度 0~1.5，默认 1.0）：\n")
        for name, (sp, st, g, desc) in PRESETS.items():
            print(f"  {name:6s}  语速{sp:4.2f}x  音高{st:+.1f}半音  增益{g:+.1f}dB   {desc}")
        sys.exit(0)

    if not args.emotion or args.emotion not in PRESETS:
        sys.exit(f"[FATAL] 情绪必须是: {', '.join(PRESETS)}")
    dst = args.out or str(Path(args.file).with_name(
        Path(args.file).stem + f"_{args.emotion}.wav"))
    p = apply_file(args.file, dst, args.emotion, args.strength)
    print(f"[完成] {args.emotion} -> {p}")
