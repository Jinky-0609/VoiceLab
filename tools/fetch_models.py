#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一键补齐 GPT-SoVITS 推理所需的三个预训练模型。

从零部署时本脚本把推理所需的预训练权重拉到
`_src/GPT-SoVITS-main/GPT_SoVITS/pretrained_models/` 下，VoiceLab 完全自包含。

用法：
    .venv-tts\\Scripts\\python.exe tools\\fetch_models.py          # 缺什么补什么
    .venv-tts\\Scripts\\python.exe tools\\fetch_models.py --force   # 全部重下
    .venv-tts\\Scripts\\python.exe tools\\fetch_models.py --check   # 只检查不下载

网络说明：HF 官方站在国内常超时，脚本优先用 ModelScope（已验证稳定 ~12MB/s），
失败再退回 HF 镜像 hf-mirror.com。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
PRETRAINED = LAB / "_src" / "GPT-SoVITS-main" / "GPT_SoVITS" / "pretrained_models"

# (本地目录名, ModelScope 仓库, HF 仓库, 需要的文件 glob, 大致体积MB)
TARGETS = [
    (
        "chinese-roberta-wwm-ext-large",
        "hfl/chinese-roberta-wwm-ext-large",
        "hfl/chinese-roberta-wwm-ext-large",
        ["config.json", "pytorch_model.bin", "vocab.txt", "tokenizer.json"],
        1300,
    ),
    (
        "chinese-hubert-base",
        "pengzhendong/chinese-hubert-base",
        "TencentGameMate/chinese-hubert-base",
        ["config.json", "pytorch_model.bin", "preprocessor_config.json"],
        380,
    ),
    (
        "models--nvidia--bigvgan_v2_24khz_100band_256x",
        None,  # ModelScope 上没有，只能走 HF 镜像
        "nvidia/bigvgan_v2_24khz_100band_256x",
        ["config.json", "bigvgan_generator.pt"],
        450,
    ),
]

# 这些是训练/其它框架用的冗余副本，推理用不到，下完就删
REDUNDANT = ["tf_model.h5", "flax_model.msgpack", "bigvgan_discriminator_optimizer.pt",
             "bigvgan_generator_3msteps.pt", "bigvgan_discriminator_optimizer_3msteps.pt"]


def dir_ok(d: Path, needed: list[str]) -> bool:
    """needed 里只要有一个存在且够大就算这个目录可用。
    各模型的必备文件集不同（BERT 要 vocab/tokenizer，HuBERT 要 preprocessor），
    沿用「分组各自独立」的方式，不要拿一份清单硬套所有模型。"""
    if not d.is_dir():
        return False
    return any((d / f).is_file() and (d / f).stat().st_size > 1024 for f in needed)


def dir_size(d: Path) -> int:
    if not d.is_dir():
        return 0
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())


def clean_redundant(d: Path) -> int:
    freed = 0
    for name in REDUNDANT:
        p = d / name
        if p.is_file():
            freed += p.stat().st_size
            p.unlink()
    for sub in (".cache", "nv-modelcard++"):
        p = d / sub
        if p.is_dir():
            freed += dir_size(p)
            shutil.rmtree(p, ignore_errors=True)
    return freed


def try_modelscope(repo: str, local_dir: Path) -> bool:
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("      (未装 modelscope，跳过该源)")
        return False
    try:
        snapshot_download(repo, local_dir=str(local_dir))
        return True
    except Exception as e:
        print(f"      ModelScope 失败: {type(e).__name__}: {str(e)[:120]}")
        return False


def try_hf(repo: str, local_dir: Path, needed: list[str]) -> bool:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("      (未装 huggingface_hub)")
        return False
    ok = True
    for fn in needed:
        done = False
        for attempt in range(1, 6):
            try:
                hf_hub_download(repo, fn, local_dir=str(local_dir))
                print(f"      {fn} OK")
                done = True
                break
            except Exception as e:
                print(f"      {fn} 第{attempt}次失败: {type(e).__name__}")
                time.sleep(4)
        if not done:
            ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="已有也重下")
    ap.add_argument("--check", action="store_true", help="只检查完整性")
    args = ap.parse_args()

    PRETRAINED.mkdir(parents=True, exist_ok=True)
    print(f"目标目录: {PRETRAINED}\n")

    all_ok = True
    for name, ms_repo, hf_repo, needed, approx_mb in TARGETS:
        d = PRETRAINED / name
        status = "完整" if dir_ok(d, needed) else "缺失/不全"
        size_mb = dir_size(d) / 2**20
        print(f"[{name}]  {status}  ({size_mb:.0f} MB / 约 {approx_mb} MB)")
        if args.check:
            if not dir_ok(d, needed):
                all_ok = False
            continue

        if dir_ok(d, needed) and not args.force:
            print("      跳过（已完整）\n")
            continue

        d.mkdir(parents=True, exist_ok=True)
        got = False
        if ms_repo:
            print("      -> 尝试 ModelScope ...", flush=True)
            got = dir_ok(d, needed) or try_modelscope(ms_repo, d)
        if not got or not dir_ok(d, needed):
            print("      -> 尝试 hf-mirror ...", flush=True)
            got = try_hf(hf_repo, d, needed)

        freed = clean_redundant(d)
        if freed:
            print(f"      清理冗余副本 {freed/2**20:.0f} MB")

        if dir_ok(d, needed):
            print(f"      OK  {dir_size(d)/2**20:.0f} MB\n")
        else:
            all_ok = False
            missing = [f for f in needed if not (d / f).is_file()]
            print(f"      !! 仍缺: {missing}\n")

    print("=" * 60)
    if args.check or all_ok:
        # 不用 emoji：Windows 控制台 GBK 会 UnicodeEncodeError
        print("全部就绪 [OK]" if all_ok else "有缺失 [FAIL]")

    # ASR 侧（Qwen3-ASR）单独检查
    check_asr()
    return 0 if all_ok else 1


def check_asr() -> bool:
    """Qwen3-ASR 权重检查。它属于 ASR 工具链（.venv-asr），与 TTS 分开算。"""
    asr_dir = LAB / "models" / "asr" / "Qwen3-ASR-0.6B"
    needed = ["model.safetensors", "config.json", "vocab.json"]
    print()
    if dir_ok(asr_dir, needed):
        print(f"[Qwen3-ASR-0.6B]  完整  ({dir_size(asr_dir)/2**20:.0f} MB)")
        return True
    print(f"[Qwen3-ASR-0.6B]  缺失  ({asr_dir})")
    print("  下载（HF 官方站，用 urllib 绕开 requests 证书问题）：")
    print(f"    .venv-asr\\Scripts\\python.exe tools\\fetch_qwen_asr.py")
    return False


if __name__ == "__main__":
    sys.exit(main())
