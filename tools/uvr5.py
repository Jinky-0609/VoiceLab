#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UVR5 人声分离封装（基于 audio-separator，UVR5 模型家族的官方 CLI 后端）

部署位置：<项目根>/uvr5/            （独立 venv: .venv-uvr）
模型目录：<项目根>/uvr5/weights/   （首次使用会自动下载，也可手动放入 .onnx/.ckpt）
解释器  ：.venv-uvr/Scripts/python.exe（Windows）或 .venv-uvr/bin/python

子命令：
  models           列出可用/已下载模型
  separate         分离人声与伴奏
  dereverb         去混响/去回声（训练数据必做）
  make-dataset     一键：分离 -> 去混响 -> 切分，直接产出训练片段

示例：
  python uvr5.py separate -i song.flac
  python uvr5.py separate -i song.flac -m UVR-MDX-NET-Voc_FT.onnx
  python uvr5.py dereverb -i out/vocals.wav
  python tools/uvr5.py make-dataset -i song.flac -o data/raw/singer
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
UVR_ROOT = LAB / "uvr5"
WEIGHTS = UVR_ROOT / "weights"
OUT = UVR_ROOT / "out"
VENV_PY = LAB / ".venv-uvr" / "Scripts" / "python.exe"

# 推荐模型（按用途）
RECOMMENDED = {
    "vocals_best":   "Kim_Vocal_2.onnx",
    "vocals_alt":    "UVR-MDX-NET-Voc_FT.onnx",
    "instrumental":  "UVR-MDX-NET-Inst_HQ_3.onnx",
    "dereverb":      "Reverb_HQ_By_FoxJoy.onnx",
}


def ensure_dirs() -> None:
    for d in (WEIGHTS, OUT):
        d.mkdir(parents=True, exist_ok=True)


def get_separator(output_dir: Path | None = None):
    from audio_separator.separator import Separator
    ensure_dirs()
    return Separator(
        log_level=20,
        model_file_dir=str(WEIGHTS),
        output_dir=str(output_dir or OUT),
        output_format="WAV",
        sample_rate=44100,
    )


def local_models() -> list[str]:
    if not WEIGHTS.is_dir():
        return []
    return sorted(p.name for p in WEIGHTS.iterdir()
                  if p.suffix.lower() in {".onnx", ".pth", ".ckpt"})


# ---------------------------------------------------------------- commands

def cmd_models(args) -> int:
    print(f"已下载的模型（{WEIGHTS}）:")
    got = local_models()
    if got:
        for m in got:
            sz = (WEIGHTS / m).stat().st_size / 1e6
            print(f"  {m}   ({sz:.1f} MB)")
    else:
        print("  （空，首次运行 separate 会自动下载）")
    print("\n推荐用途:")
    for k, v in RECOMMENDED.items():
        mark = "✓ 已有" if v in got else "  需下载"
        print(f"  {k:12} {v:32} {mark}")
    return 0


def do_separate(input_path: Path, model: str, out_dir: Path) -> list[Path]:
    sep = get_separator(out_dir)
    print(f"[加载模型] {model}", flush=True)
    sep.load_model(model_filename=model)
    print(f"[分离] {input_path.name}", flush=True)
    files = sep.separate(str(input_path))
    res = [out_dir / f for f in files]
    for r in res:
        if r.exists():
            print(f"  -> {r.name}  ({r.stat().st_size/1e6:.1f} MB)")
    return res


def cmd_separate(args) -> int:
    src = Path(args.input)
    if not src.is_file():
        sys.exit(f"[FATAL] 找不到输入: {src}")
    out_dir = Path(args.out_dir) if args.out_dir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    model = args.model or RECOMMENDED["vocals_best"]
    do_separate(src, model, out_dir)
    print("\n[提示] 人声通常名为 *_(Vocals)_*.wav；下一步建议去混响：")
    print("       python uvr5.py dereverb -i <人声文件>")
    return 0


def cmd_dereverb(args) -> int:
    src = Path(args.input)
    if not src.is_file():
        sys.exit(f"[FATAL] 找不到输入: {src}")
    out_dir = Path(args.out_dir) if args.out_dir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    model = args.model or RECOMMENDED["dereverb"]
    do_separate(src, model, out_dir)
    print("\n[提示] 去混响后输出里的干声那一轨才是训练可用素材。")
    return 0


def cmd_make_dataset(args) -> int:
    """分离 -> 去混响 -> 切分，一步产出训练片段目录。"""
    src = Path(args.input)
    if not src.is_file():
        sys.exit(f"[FATAL] 找不到输入: {src}")
    out_root = Path(args.out_dir) if args.out_dir else LAB / "data" / "raw" / src.stem
    out_root.mkdir(parents=True, exist_ok=True)
    stage1 = out_root / "_stage1_separate"
    stage2 = out_root / "_stage2_dereverb"
    final = out_root
    for d in (stage1, stage2):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("步骤 1/3  分离人声")
    print("=" * 60)
    files = do_separate(src, args.model or RECOMMENDED["vocals_best"], stage1)
    live = [f for f in files if f.exists()]
    # 注意：不能只匹配 "voc"——模型名 Kim_Vocal_2 本身含 voc，两轨都会命中。
    # 必须用输出模板里的 "_(Vocals)_" / "_(Instrumental)_" 区分。
    vocals = [f for f in live if "_(vocals)_" in f.stem.lower()]
    if not vocals:
        vocals = [f for f in live if "instrumental" not in f.stem.lower()]
    if not vocals:
        sys.exit(f"[FATAL] 没找到人声轨，实际输出: {[f.name for f in files]}")
    vocal = vocals[0]
    if "instrumental" in vocal.stem.lower():
        sys.exit(f"[FATAL] 人声轨判定失败（拿到的是伴奏）：{vocal.name}")
    print(f"  人声轨: {vocal.name}")

    print("\n" + "=" * 60)
    print("步骤 2/3  去混响")
    print("=" * 60)
    try:
        dfiles = do_separate(vocal, args.model_dereverb or RECOMMENDED["dereverb"], stage2)
        live = [f for f in dfiles if f.exists()]
        # 去混响模型输出两轨: "(No Reverb)" = 干声, "(Reverb)" = 混响残留
        # 必须优先取干声，不能靠 "voc" 这类宽泛匹配（两轨文件名里都有 Vocals）
        dry = [f for f in live if "no reverb" in f.stem.lower()]
        if not dry:
            dry = [f for f in live if "dry" in f.stem.lower()]
        if not dry:
            dry = [f for f in live if "reverb" not in f.stem.lower()]
        clean = dry[0] if dry else (live[0] if live else None)
        if clean:
            print(f"  干声轨: {clean.name}")
    except Exception as e:
        print(f"  ⚠ 去混响失败（{type(e).__name__}: {e}），退回未去混响的人声")
        clean = vocal

    if clean is None or not clean.exists():
        sys.exit("[FATAL] 去混响阶段没有可用输出")

    print("\n" + "=" * 60)
    print("步骤 3/3  按静音切分")
    print("=" * 60)
    split_py = LAB / "tools" / "split_silence.py"
    cmd = [str(VENV_PY if VENV_PY.exists() else sys.executable), str(split_py),
           "-i", str(clean), "-o", str(final),
           "--min-len", str(args.min_len), "--max-len", str(args.max_len)]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"[FATAL] 切分失败（退出码 {r.returncode}）")

    n = len(list(final.glob("*.wav")))
    print("\n" + "=" * 60)
    print(f"[完成] 训练片段 {n} 个 -> {final}")
    print("下一步：")
    print(f'  用 .venv-asr 的解释器运行 tools/asr_annotate.py -i "{final}" -s {src.stem} -l 中文')
    print("=" * 60)
    return 0


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description="UVR5 人声分离（audio-separator 后端）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("models", help="列出模型").set_defaults(func=cmd_models)

    sp = sub.add_parser("separate", help="分离人声/伴奏")
    sp.add_argument("-i", "--input", required=True)
    sp.add_argument("-m", "--model", help=f"模型文件名（默认 {RECOMMENDED['vocals_best']}）")
    sp.add_argument("-d", "--out-dir")
    sp.set_defaults(func=cmd_separate)

    dp = sub.add_parser("dereverb", help="去混响/去回声")
    dp.add_argument("-i", "--input", required=True)
    dp.add_argument("-m", "--model", help=f"模型文件名（默认 {RECOMMENDED['dereverb']}）")
    dp.add_argument("-d", "--out-dir")
    dp.set_defaults(func=cmd_dereverb)

    mp = sub.add_parser("make-dataset", help="一键：分离+去混响+切分")
    mp.add_argument("-i", "--input", required=True)
    mp.add_argument("-o", "--out-dir")
    mp.add_argument("--model")
    mp.add_argument("--model-dereverb")
    mp.add_argument("--min-len", type=float, default=3.0)
    mp.add_argument("--max-len", type=float, default=12.0)
    mp.set_defaults(func=cmd_make_dataset)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
