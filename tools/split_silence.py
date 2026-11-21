#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按静音把长音频切成训练片段（GPT-SoVITS 数据集第一步）。

用法：
  python tools/split_silence.py -i 长音频.wav -o data/raw/myspeaker
  python split_silence.py -i 长音频.wav -o ... --min-len 3 --max-len 12 --silence-db -38
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa


def main() -> int:
    ap = argparse.ArgumentParser(description="按静音切分长音频")
    ap.add_argument("-i", "--input", required=True, help="输入音频")
    ap.add_argument("-o", "--out-dir", required=True, help="片段输出目录")
    ap.add_argument("--min-len", type=float, default=2.0, help="片段最短秒数（默认 2；低于此值会被丢弃）")
    ap.add_argument("--max-len", type=float, default=12.0, help="片段最长秒数（默认 12）")
    ap.add_argument("--silence-db", type=float, default=-45.0,
                    help="静音判定：低于峰值这么多 dB 才算静音（默认 -45）。"
                         "数值越小越严格→切得越少；语速快、停顿多的素材用 -50 ~ -55 更保数据")
    ap.add_argument("--min-silence", type=float, default=0.5,
                    help="短于此秒数的间隔视为同一句，会被合并（默认 0.5）")
    ap.add_argument("--pad", type=float, default=0.12, help="片段前后留白秒数（默认 0.12）")
    ap.add_argument("--fade", type=float, default=0.05,
                    help="硬切边界淡入淡出秒数（默认 0.05）。max-len 等分切断的边界不是自然静音，"
                         "不做淡出会让模型学到「句尾可以戛然而止」→ 推理吞字")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.is_file():
        sys.exit(f"[FATAL] 找不到输入: {src}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio, sr = sf.read(str(src), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    dur = len(audio) / sr
    print(f"[输入] {src.name}  {dur:.1f}s @ {sr} Hz")

    # 重采样到 32k（GPT-SoVITS 训练器常用 32k；保留原始也可）
    target_sr = 32000
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    intervals = librosa.effects.split(audio, top_db=abs(args.silence_db), frame_length=1024, hop_length=256)
    # 合并过短的静音间隔
    merged: list[list[int]] = []
    min_gap = int(args.min_silence * sr)
    for s, e in intervals:
        if merged and s - merged[-1][1] < min_gap:
            merged[-1][1] = e
        else:
            merged.append([int(s), int(e)])

    # 太长的段落再按 max_len 切：等分成 n 段，避免出现被丢弃的小尾巴
    pad = int(args.pad * sr)
    max_len = int(args.max_len * sr)
    min_len = int(args.min_len * sr)
    pieces: list[tuple[int, int]] = []
    for s, e in merged:
        span = e - s
        if span <= max_len:
            pieces.append((s, e))
        else:
            n = int(np.ceil(span / max_len))
            step = span / n
            for k in range(n):
                pieces.append((int(s + k * step), int(s + (k + 1) * step)))

    kept = 0
    kept_samples = 0.0
    dropped_samples = 0
    faded_count = 0
    for s, e in pieces:
        if e - s < min_len:
            dropped_samples += e - s
            continue
        a = max(0, s - pad)
        b = min(len(audio), e + pad)
        kept += 1
        kept_samples += b - a
        out = out_dir / f"{src.stem}_{kept:04d}.wav"
        seg = audio[a:b].copy()
        # 硬切边界检测：等分切割（max-len）产生的段边界切在有声处，
        # 表现为段首/段尾 10ms 能量明显高于全段均值 → 加淡入/淡出。
        # 自然静音边界（librosa.split 切出来的）本身有 pad 留白，能量低，不会触发。
        fade_n = max(1, int(args.fade * sr))
        win = max(1, int(0.010 * sr))
        head_rms = float(np.sqrt(np.mean(seg[:win] ** 2))) if len(seg) > win else 0.0
        tail_rms = float(np.sqrt(np.mean(seg[-win:] ** 2))) if len(seg) > win else 0.0
        body_rms = float(np.sqrt(np.mean(seg ** 2))) or 1e-9
        faded = 0
        if head_rms > body_rms * 0.25:  # 开头就是响的 → 切在字中间，淡入
            seg[:fade_n] *= np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
            faded += 1
        if tail_rms > body_rms * 0.25:  # 结尾还在响 → 尾音被切，淡出
            seg[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
            faded += 1
        if faded:
            faded_count += 1
        sf.write(str(out), seg, sr)
    total = sum(1 for _ in out_dir.glob("*.wav"))
    kept_s = kept_samples / sr
    drop_s = dropped_samples / sr
    print(f"[切分] 输出 {kept} 段 -> {out_dir}（目录内现有 {total} 个 wav）")
    print(f"[淡出] {faded_count} 段检测到硬切边界，已加 {args.fade*1000:.0f}ms 淡入淡出")
    print(f"[统计] 保留 {kept_s:.1f}s / {dur:.1f}s ({kept_s/dur*100:.1f}%)，"
          f"因短于 {args.min_len}s 丢弃 {drop_s:.1f}s")
    if drop_s > dur * 0.25:
        print(f"[警告] 丢弃比例偏高（>{25}%）。语速快/停顿多时，请减小 --min-len，"
              f"并把 --silence-db 设为更小（如 -55）以少切几刀。")
    print(f"[提示] 下一步: python asr_annotate.py -i \"{out_dir}\" -s <说话人名> -l 中文")
    return 0


if __name__ == "__main__":
    sys.exit(main())
