#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量语音识别标注，产出 GPT-SoVITS 训练用的 .list 文件。

**引擎：Qwen3-ASR-0.6B**（Qwen3-Omni 底座，52 语种/22 中文方言，
自带标点断句与语种识别）。权重在 `models/asr/Qwen3-ASR-0.6B`，
完全本地，不依赖任何外部安装目录。

比之前的 Paraformer 好在：
  * 多语种（中文/英文/日文/粤语/韩文/阿拉伯语/德法西葡…共 30 语种 + 22 中文方言）
  * 输出自带标点，不用再挂标点恢复模型
  * language=None 时自动识别语种，`.list` 的语言码也跟着自动填

用法：
  # 自动识别语种（推荐，多语种素材直接用）
  .venv-asr\\Scripts\\python.exe tools\\asr_annotate.py -i data\\raw\\myspeaker -s myspeaker

  # 强制指定语种
  .venv-asr\\Scripts\\python.exe tools\\asr_annotate.py -i ... -s myspeaker -l 日文

  # 用更大更准的 1.7B（需先下载权重）
  .venv-asr\\Scripts\\python.exe tools\\asr_annotate.py -i ... -s myspeaker --model models\\asr\\Qwen3-ASR-1.7B

产出：
  <out>/<speaker>.list   每行: 绝对路径|说话人|语言码|文本
  <out>/<speaker>.jsonl  明细（含耗时、时长、识别语种）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = LAB / "models" / "asr" / "Qwen3-ASR-0.6B"

# 界面语言名 -> Qwen3-ASR 的 language 参数值（None = 自动识别）
LANG_MAP = {
    "自动": None,
    "中文": "Chinese",
    "英文": "English",
    "日文": "Japanese",
    "粤语": "Cantonese",
    "韩文": "Korean",
    "德文": "German",
    "法文": "French",
    "西班牙文": "Spanish",
    "俄文": "Russian",
    "意大利文": "Italian",
    "葡萄牙文": "Portuguese",
    "阿拉伯文": "Arabic",
    "泰文": "Thai",
    "越南文": "Vietnamese",
}

# Qwen3-ASR 返回的语种名 -> GPT-SoVITS .list 用的语言码
QWEN_LANG_TO_GSV = {
    "Chinese": "zh", "English": "en", "Japanese": "ja", "Cantonese": "yue",
    "Korean": "ko", "Arabic": "ar", "German": "de", "French": "fr",
    "Spanish": "es", "Portuguese": "pt", "Indonesian": "id", "Italian": "it",
    "Russian": "ru", "Thai": "th", "Vietnamese": "vi", "Turkish": "tr",
    "Hindi": "hi", "Malay": "ms", "Dutch": "nl", "Swedish": "sv",
    "Danish": "da", "Finnish": "fi", "Polish": "pl", "Czech": "cs",
    "Filipino": "fil", "Persian": "fa", "Greek": "el", "Hungarian": "hu",
    "Macedonian": "mk", "Romanian": "ro",
}

AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".wma"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen3-ASR 批量标注 -> GPT-SoVITS .list")
    ap.add_argument("-i", "--input", required=True, help="音频文件或目录")
    ap.add_argument("-o", "--out-dir", help="输出目录（默认与输入同级的 <speaker> 目录）")
    ap.add_argument("-s", "--speaker", required=True, help="说话人名称（写入 .list 第二列）")
    ap.add_argument("-l", "--lang", default="自动", choices=list(LANG_MAP),
                    help="音频语言，默认自动识别")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="Qwen3-ASR 权重目录")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=1, help="推理批大小")
    ap.add_argument("--context", default="", help="上下文提示（如专有名词列表）")
    ap.add_argument("--min-dur", type=float, default=1.0, help="忽略短于该秒数的片段")
    ap.add_argument("--max-dur", type=float, default=30.0, help="忽略长于该秒数的片段")
    args = ap.parse_args()

    src = Path(args.input)
    if src.is_dir():
        files = sorted(p for p in src.rglob("*") if p.suffix.lower() in AUDIO_EXT)
    elif src.is_file():
        files = [src]
    else:
        sys.exit(f"[FATAL] 输入不存在: {src}")
    if not files:
        sys.exit(f"[FATAL] 没有找到音频文件: {src}")

    model_dir = Path(args.model)
    if not (model_dir / "model.safetensors").is_file():
        sys.exit(f"[FATAL] Qwen3-ASR 权重不存在: {model_dir}\n"
                 f"  应包含 model.safetensors；可用 tools/fetch_models.py 下载。")

    out_dir = Path(args.out_dir) if args.out_dir else src.parent / args.speaker
    out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import soundfile as sf
    import torch
    from qwen_asr import Qwen3ASRModel

    print(f"[加载] {model_dir.name}", flush=True)
    t0 = time.perf_counter()
    model = Qwen3ASRModel.from_pretrained(
        str(model_dir),
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map=args.device,
        max_inference_batch_size=max(1, args.batch),
        max_new_tokens=256,
        attn_implementation="sdpa",
    )
    print(f"[加载] {time.perf_counter() - t0:.1f}s  device={args.device}", flush=True)

    # 首次推理有 kernel 编译开销，先用 1 秒静音热身
    try:
        prev = model.max_new_tokens
        model.max_new_tokens = 32
        model.transcribe(audio=(np.zeros(16000, dtype=np.float32), 16000),
                         language=None, context="")
        model.max_new_tokens = prev
        print("[热身] 完成", flush=True)
    except Exception as e:
        print(f"[热身] 跳过: {type(e).__name__}: {str(e)[:60]}", flush=True)

    want_lang = LANG_MAP[args.lang]
    forced_code = QWEN_LANG_TO_GSV.get(want_lang or "", "zh")

    list_lines: list[str] = []
    jsonl_lines: list[str] = []
    skipped = 0
    lang_stat: dict[str, int] = {}

    for i, f in enumerate(files, 1):
        try:
            info = sf.info(str(f))
            dur = info.frames / info.samplerate
        except Exception as e:
            print(f"  ({i}/{len(files)}) 读不出时长，跳过 {f.name}: {e}", flush=True)
            skipped += 1
            continue

        if dur < args.min_dur or dur > args.max_dur:
            print(f"  ({i}/{len(files)}) 跳过 {f.name}  {dur:.2f}s 不在 [{args.min_dur},{args.max_dur}]", flush=True)
            skipped += 1
            continue

        audio, sr = sf.read(str(f), dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        audio = np.ascontiguousarray(audio, dtype=np.float32)

        t1 = time.perf_counter()
        try:
            res = model.transcribe(audio=(audio, sr), language=want_lang, context=args.context)
        except Exception as e:
            print(f"  ({i}/{len(files)}) 识别失败 {f.name}: {type(e).__name__}: {str(e)[:80]}", flush=True)
            skipped += 1
            continue
        dt = time.perf_counter() - t1

        if not res:
            print(f"  ({i}/{len(files)}) 空结果 {f.name}", flush=True)
            skipped += 1
            continue

        text = (getattr(res[0], "text", "") or "").strip()
        detected = getattr(res[0], "language", None)
        # 自动识别时用检测到的语种填 .list；强制指定时用用户给的
        code = QWEN_LANG_TO_GSV.get(detected, forced_code) if want_lang is None else forced_code

        text = text.replace("|", " ").replace("\n", " ").strip()
        if not text:
            print(f"  ({i}/{len(files)}) 空文本 {f.name}", flush=True)
            skipped += 1
            continue

        lang_stat[detected or args.lang] = lang_stat.get(detected or args.lang, 0) + 1
        rtf = dt / dur if dur else 0
        print(f"  ({i}/{len(files)}) {dur:5.2f}s {dt:5.2f}s RTF={rtf:4.2f} [{detected or '-'}->{code}] {text[:38]}", flush=True)

        list_lines.append(f"{f.resolve()}|{args.speaker}|{code}|{text}")
        jsonl_lines.append(json.dumps(
            {"path": str(f.resolve()), "dur": round(dur, 3), "sec": round(dt, 2),
             "lang": code, "detected_language": detected, "text": text},
            ensure_ascii=False))

    list_path = out_dir / f"{args.speaker}.list"
    jsonl_path = out_dir / f"{args.speaker}.jsonl"
    list_path.write_text("\n".join(list_lines) + ("\n" if list_lines else ""), encoding="utf-8")
    jsonl_path.write_text("\n".join(jsonl_lines) + ("\n" if jsonl_lines else ""), encoding="utf-8")

    total_dur = sum(json.loads(l)["dur"] for l in jsonl_lines) if jsonl_lines else 0.0
    print(f"\n[完成] {len(list_lines)} 条 / 跳过 {skipped} 条 / 总时长 {total_dur/60:.1f} 分钟")
    if lang_stat:
        print("  语种分布: " + ", ".join(f"{k}={v}" for k, v in sorted(lang_stat.items(), key=lambda x: -x[1])))
    print(f"  .list  : {list_path}")
    print(f"  .jsonl : {jsonl_path}")
    if total_dur < 300:
        print(f"  [!] 建议至少 5~10 分钟有效语音，当前仅 {total_dur/60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
