#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频抽音轨 -> 可直接参与训练的音频（VoiceLab）

用系统 ffmpeg（已装 9.0.2 full build）。
直接解码成 PCM WAV，不做二次有损压缩，所以音质 = 视频里原始音轨的音质。

子命令：
  probe    查看视频里的音频轨规格（编码/采样率/声道/码率）
  extract  单个文件抽音轨
  batch    整个目录批量抽

用法：
  python video2audio.py probe  -i video.mp4
  python video2audio.py extract -i video.mp4                 # -> video.wav
  python tools/video2audio.py extract -i video.mp4 -o data/refs/me.wav --mono
  python tools/video2audio.py batch  -i <视频目录> -d data/raw/me
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

VIDEO_EXT = {".mp4", ".mkv", ".avi", ".flv", ".mov", ".webm", ".wmv", ".m4v", ".ts", ".mpg", ".mpeg", ".3gp"}


def ffmpeg() -> str:
    return "ffmpeg"


def ffprobe() -> str:
    return "ffprobe"


def run(cmd: list[str], quiet: bool = True) -> tuple[int, str]:
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, (r.stdout + r.stderr)


def probe_streams(path: Path) -> list[dict]:
    code, out = run([
        ffprobe(), "-v", "error",
        "-show_entries", "stream=index,codec_type,codec_name,sample_rate,channels,bit_rate,duration:stream_tags=language,title",
        "-of", "json", str(path),
    ])
    if code != 0:
        raise RuntimeError(f"ffprobe 失败: {out.strip()[:300]}")
    return json.loads(out).get("streams", [])


def cmd_probe(args) -> int:
    src = Path(args.input)
    if not src.is_file():
        sys.exit(f"[FATAL] 找不到文件: {src}")
    streams = probe_streams(src)
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    video = [s for s in streams if s.get("codec_type") == "video"]

    print(f"文件: {src.name}  ({src.stat().st_size/1e6:.1f} MB)")
    print(f"  视频轨: {len(video)}   音频轨: {len(audio)}")
    if not audio:
        print("\n  ❌ 这个文件没有音频轨，抽不出东西")
        return 1

    for i, s in enumerate(audio):
        sr = s.get("sample_rate", "?")
        ch = s.get("channels", "?")
        br = s.get("bit_rate")
        dur = s.get("duration", "?")
        br_s = f"{int(br)//1000} kbps" if br and str(br).isdigit() else "?"
        print(f"\n  [音轨 {i}] index={s.get('index')}")
        print(f"    编码    : {s.get('codec_name')}")
        print(f"    采样率  : {sr} Hz")
        print(f"    声道    : {ch}")
        print(f"    码率    : {br_s}")
        print(f"    时长    : {dur} s")
        # 训练可用性提示
        warn = []
        try:
            if int(sr) < 16000:
                warn.append(f"采样率 {sr} Hz 低于 16 kHz，音质偏薄")
        except Exception:
            pass
        try:
            if br and str(br).isdigit() and int(br) // 1000 < 64:
                warn.append(f"码率 {br_s} 偏低，高频可能已被砍掉")
        except Exception:
            pass
        if warn:
            for w in warn:
                print(f"    ⚠ {w}")
        else:
            print(f"    ✓ 规格满足训练要求（≥16 kHz）")
    return 0


def extract_one(src: Path, dest: Path, stream: int = 0, sr: int | None = None,
                mono: bool = False, fmt: str = "wav") -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(src), "-map", f"0:a:{stream}",
           "-vn", "-sn", "-dn"]
    if fmt == "flac":
        cmd += ["-c:a", "flac"]
    else:
        cmd += ["-c:a", "pcm_s16le"]
    if sr:
        cmd += ["-ar", str(sr)]
    if mono:
        cmd += ["-ac", "1"]
    cmd += [str(dest)]
    code, out = run(cmd)
    if code != 0:
        raise RuntimeError(f"ffmpeg 失败: {out.strip()[:400]}")
    if not dest.is_file() or dest.stat().st_size == 0:
        raise RuntimeError(f"输出为空: {dest}")
    return dest


def cmd_extract(args) -> int:
    src = Path(args.input)
    if not src.is_file():
        sys.exit(f"[FATAL] 找不到文件: {src}")
    audio = [s for s in probe_streams(src) if s.get("codec_type") == "audio"]
    if not audio:
        sys.exit(f"[FATAL] {src.name} 没有音频轨")
    if args.stream >= len(audio):
        sys.exit(f"[FATAL] 只有 {len(audio)} 条音轨，请求的是第 {args.stream} 条")

    fmt = args.format
    dest = Path(args.output) if args.output else src.with_suffix("." + ("flac" if fmt == "flac" else "wav"))
    print(f"[抽取] {src.name}  音轨 {args.stream}  ->  {dest.name}")
    extract_one(src, dest, args.stream, args.sr, args.mono, fmt)
    sz = dest.stat().st_size / 1e6
    print(f"  完成  {sz:.1f} MB")
    print(f"\n下一步（如果是带伴奏/环境音的视频）：")
    print(f"  python tools/uvr5.py make-dataset -i \"{dest}\" -o data/raw/<名字>")
    return 0


def cmd_batch(args) -> int:
    root = Path(args.input)
    if not root.is_dir():
        sys.exit(f"[FATAL] 不是目录: {root}")
    out_dir = Path(args.out_dir) if args.out_dir else root / "_audio"
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXT and p.is_file())
    if not files:
        sys.exit(f"[FATAL] 目录里没有视频文件: {root}")
    ok = fail = skip = 0
    for p in files:
        try:
            audio = [s for s in probe_streams(p) if s.get("codec_type") == "audio"]
            if not audio:
                print(f"  · 跳过（无音轨） {p.name}")
                skip += 1
                continue
            dest = out_dir / (p.stem + ("." + args.format))
            extract_one(p, dest, 0, args.sr, args.mono, args.format)
            print(f"  ✓ {p.name} -> {dest.name}  ({dest.stat().st_size/1e6:.1f} MB)")
            ok += 1
        except Exception as e:
            print(f"  ✗ {p.name}: {e}")
            fail += 1
    print(f"\n[完成] 成功 {ok}  跳过 {skip}  失败 {fail}")
    print(f"输出目录: {out_dir}")
    return 0 if fail == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="视频抽音轨（ffmpeg）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("probe", help="查看音频轨规格"); s.add_argument("-i", "--input", required=True); s.set_defaults(func=cmd_probe)

    e = sub.add_parser("extract", help="抽取单个文件")
    e.add_argument("-i", "--input", required=True)
    e.add_argument("-o", "--output")
    e.add_argument("--stream", type=int, default=0, help="第几条音轨（默认 0）")
    e.add_argument("--sr", type=int, help="目标采样率（默认保持原样）")
    e.add_argument("--mono", action="store_true", help="转单声道")
    e.add_argument("--format", choices=["wav", "flac"], default="wav")
    e.set_defaults(func=cmd_extract)

    b = sub.add_parser("batch", help="批量抽取目录")
    b.add_argument("-i", "--input", required=True)
    b.add_argument("-d", "--out-dir")
    b.add_argument("--sr", type=int)
    b.add_argument("--mono", action="store_true")
    b.add_argument("--format", choices=["wav", "flac"], default="wav")
    b.set_defaults(func=cmd_batch)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
