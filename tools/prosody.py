#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""参数化情绪 v2：不用变调，靠"时长 + 力度 + 停顿"表达情绪。

    .venv-tts/Scripts/python.exe tools/prosody.py -f out/x.wav -e 开心

## 为什么 v1（变调）是错的

v1 用 pitch_shift 升/降半音来表达情绪。实测代价很大：

    原始   F0 中位 146.3Hz
    开心   F0 中位 157.9Hz   (+11.6Hz)
    激动   F0 中位 161.1Hz   (+14.8Hz)

**变调整段波形等于改变了说话人的性别感和年龄感**，不是表达情绪。
用户听到的"声音变尖"就是这个 —— 那是音色被改坏了，不是情绪。

## v2 的做法（全部不碰音高）

| 手段 | 作用 | 为什么这样是对的 |
|---|---|---|
| 时长伸缩 | 语速 | 只改时间轴，基频原样保留（用保音高的 time_stretch） |
| 增益包络 | 力度 | 情绪强弱最直接的听觉线索 |
| 停顿压缩/拉伸 | 节奏 | 激动时停顿短，悲伤时停顿长——这是真人说话的真实差异 |
| 轻微低通/高通 | 明暗 | 激动略提高频（更"亮"），悲伤略削高频（更"闷"），幅度极小（<1.5dB） |

**不做的事**：不变调、不加重叠、不加混响。这些都会让声音失真或"分层"。
"""
from __future__ import annotations

import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "tools"))

# 预设：语义名 -> (语速, 增益dB, 停顿倍率, 高频倾斜dB, 说明)
# 语速：>1 更快；停顿：<1 停顿更短（更急）；倾斜：>0 更亮，<0 更闷
PRESETS = {
    "中性": (1.00, 0.0, 1.00, 0.0, "不做处理"),
    "开心": (1.06, 1.5, 0.90, 1.0, "略快、稍亮、停顿略短"),
    "激动": (1.12, 2.5, 0.80, 1.5, "更快、更响、停顿明显变短"),
    "生气": (1.08, 3.0, 0.85, 1.0, "偏快、力度强、停顿短"),
    "悲伤": (0.90, -2.0, 1.25, -1.5, "放慢、变轻、停顿拉长"),
    "温柔": (0.94, -1.0, 1.10, -0.8, "稍慢、柔和、停顿稍长"),
    "严肃": (0.97, 0.5, 1.05, -0.5, "略慢、保持力度、少起伏"),
    "耳语": (0.98, -7.0, 1.10, -1.0, "音量很低"),
    "疲惫": (0.88, -3.0, 1.35, -1.5, "很慢、很轻、停顿很长"),
}

# 实测：情绪强度靠语速+力度+停顿已经足够可辨，不需要动音高。
# 保留这个常量是为了让"不做变调"这件事在代码里显式可见。
PITCH_SHIFT_DISABLED = True


def available() -> list[str]:
    return list(PRESETS)


def describe(name: str) -> str:
    p = PRESETS.get(name)
    return p[4] if p else ""


def _detect_silences(x, sr: int, thr_db: float = -38.0):
    """找出静音区间 [(start, end), ...]，用于压缩/拉伸停顿。"""
    import numpy as np
    win = max(2, int(0.02 * sr))
    n = len(x) // win
    if n < 2:
        return []
    e = np.sqrt((x[: n * win].reshape(n, win) ** 2).mean(axis=1)) + 1e-9
    db = 20 * np.log10(e / (e.max() + 1e-9))
    quiet = db < thr_db
    spans, start = [], None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if (i - start) * win >= int(0.08 * sr):     # 只处理 ≥80ms 的停顿
                spans.append((start * win, i * win))
            start = None
    if start is not None and (n - start) * win >= int(0.08 * sr):
        spans.append((start * win, n * win))
    return spans


def _adjust_pauses(x, sr: int, factor: float):
    """按倍率调整停顿长度（>1 拉长，<1 压缩）。不改动有声部分。"""
    import numpy as np
    if abs(factor - 1.0) < 0.02:
        return x
    spans = _detect_silences(x, sr)
    if not spans:
        return x
    out, cursor = [], 0
    for s, e in spans:
        out.append(x[cursor:s])
        seg = x[s:e]
        target = max(1, int(len(seg) * factor))
        if target == len(seg):
            out.append(seg)
        else:
            # 静音段用线性重采样即可，不会引入可听失真
            idx = np.linspace(0, len(seg) - 1, target)
            out.append(np.interp(idx, np.arange(len(seg)), seg).astype(np.float32))
        cursor = e
    out.append(x[cursor:])
    return np.concatenate(out) if out else x


def _tilt(x, sr: int, gain_db: float):
    """极轻的高频倾斜（不用完整 EQ，避免相位问题）。

    用一阶差分做高通分量，按 dB 混回去：
        y = x + k * (x - lowpass(x))
    k 很小，听感上是"亮/闷"的微调，不是音色改变。
    """
    import numpy as np
    if abs(gain_db) < 0.3:
        return x
    k = 10 ** (gain_db / 20) - 1.0
    # 一阶低通
    a = 0.85
    lp = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc = a * acc + (1 - a) * x[i]
        lp[i] = acc
    hp = x - lp
    y = x + k * hp
    return y.astype(np.float32)


def apply(audio, sr: int, emotion: str, strength: float = 1.0):
    """对音频应用参数化情绪（不改音高）。"""
    import numpy as np

    p = PRESETS.get(emotion)
    if p is None or emotion == "中性":
        return np.asarray(audio, dtype=np.float32).ravel()

    speed, gain_db, pause_f, tilt_db, _ = p
    s = max(0.0, min(1.5, float(strength)))
    if s == 0:
        return np.asarray(audio, dtype=np.float32).ravel()

    speed = 1.0 + (speed - 1.0) * s
    gain_db = gain_db * s
    pause_f = 1.0 + (pause_f - 1.0) * s
    tilt_db = tilt_db * s

    x = np.asarray(audio, dtype=np.float32).ravel()
    if len(x) < 256:
        return x

    # 1) 停顿调整（先做，避免后面变速干扰静音检测）
    x = _adjust_pauses(x, sr, pause_f)

    # 2) 变频（保音高）。librosa 的 time_stretch 用相位声码器，不改变基频。
    if abs(speed - 1.0) > 0.005:
        try:
            import librosa
            x = librosa.effects.time_stretch(y=x, rate=speed).astype(np.float32)
        except Exception as e:
            print(f"[韵律] 变速失败，跳过: {type(e).__name__}", flush=True)

    # 3) 高频倾斜（明暗微调）
    x = _tilt(x, sr, tilt_db)

    # 4) 增益 + 归一化（保留情绪间的力度差，但不削顶）
    if abs(gain_db) > 0.05:
        x = x * (10 ** (gain_db / 20))
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
    ap = argparse.ArgumentParser(description="参数化情绪（不改音高）")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("-f", "--file")
    ap.add_argument("-o", "--out")
    ap.add_argument("-e", "--emotion")
    ap.add_argument("--strength", type=float, default=1.0)
    args = ap.parse_args()

    if args.list or not args.file:
        print("参数化情绪预设（全部不改音高）：\n")
        for name, (sp, g, pf, tl, desc) in PRESETS.items():
            print(f"  {name:6s} 语速{sp:4.2f}x 增益{g:+.1f}dB "
                  f"停顿×{pf:.2f} 明暗{tl:+.1f}dB   {desc}")
        sys.exit(0)

    if not args.emotion or args.emotion not in PRESETS:
        sys.exit(f"[FATAL] 情绪必须是: {', '.join(PRESETS)}")
    dst = args.out or str(Path(args.file).with_name(
        Path(args.file).stem + f"_{args.emotion}.wav"))
    p = apply_file(args.file, dst, args.emotion, args.strength)
    print(f"[完成] {args.emotion} -> {p}")
