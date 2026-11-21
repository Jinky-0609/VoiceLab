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

# 按【单文件】发布的基座权重（上游 git 仓库里不含，pretrained_models/ 只有 .gitignore）。
# 来源：lj1995/GPT-SoVITS —— GPT-SoVITS 官方权重仓。
FILE_TARGETS = [
    ("s1v3.ckpt", "lj1995/GPT-SoVITS", "s1v3.ckpt", 148),
    ("s2Gv3.pth", "lj1995/GPT-SoVITS", "s2Gv3.pth", 733),
]

# 中文多音字模型 G2PW：上游以 zip 发布（XXXXRT/GPT-SoVITS-Pretrained）。
# 加载路径是相对 cwd 的 "GPT_SoVITS/text/G2PWModel"（text/chinese2.py:35），
# 所以解压到两处：仓库内（tts_engine chdir 后的生效位置）与项目根（兼容用法）。
G2PW_REPO = "XXXXRT/GPT-SoVITS-Pretrained"
G2PW_ZIP = "G2PWModel.zip"
G2PW_NEED = ["g2pW.onnx", "bopomofo_to_pinyin_wo_tune_dict.json"]
# 直链目标：上游 git 仓库不含、也不在 HF 上，但推理 import 链上必需。
# fast_langdetect 在语种检测路径上硬要 lid.176.bin，缺失会在模型加载阶段
# 抛 FileNotFoundError（fresh 部署实测踩过）。MD5 与 fast_langdetect 包内
# 常量 FASTTEXT_LARGE_MODEL_MD5 一致，即官方 fastText 语言识别模型。
URL_TARGETS = [
    # (项目相对落地路径, 主源, 备源, MD5, 大致 MB)
    ("_src/GPT-SoVITS-main/GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin",
     "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
     "https://github.com/LlmKira/fast-langdetect/releases/download/models-v1/lid.176.bin",
     "01810bc59c6a3d2b79c79e6336612f65", 125),
]

G2PW_DIRS = [
    LAB / "_src" / "GPT-SoVITS-main" / "GPT_SoVITS" / "text" / "G2PWModel",
    LAB / "GPT_SoVITS" / "text" / "G2PWModel",
]


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


def _shutil_rmtree(p: Path) -> None:
    import shutil
    shutil.rmtree(p, ignore_errors=True)


def fetch_single(name: str, repo: str, filename: str, approx_mb: int) -> bool:
    """拉按单文件发布的基座权重到 models/pretrained/<name>。"""
    dest = LAB / "models" / "pretrained" / name
    if dest.is_file() and dest.stat().st_size > int(approx_mb * 0.9) * 2**20:
        print(f"[{name}]  已在 {dest.relative_to(LAB)}（跳过）")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{name}]  <- {repo}/{filename} (~{approx_mb} MB)")
    tmp = dest.parent
    if try_hf(repo, str(tmp), [filename]):
        got = tmp / filename
        if got.is_file() and got != dest:
            got.replace(dest)
        cache = tmp / ".cache"
        if cache.is_dir():
            _shutil_rmtree(cache)
        ok = dest.is_file() and dest.stat().st_size > int(approx_mb * 0.85) * 2**20
        if dest.is_file():
            print(f"      {'OK' if ok else '!! 大小异常'}  {dest.stat().st_size / 2**20:.0f} MB")
        else:
            print("      !! 失败")
            return False
        return ok
    print(f"      !! 下载失败：{repo}/{filename}")
    return False


def fetch_url(rel_dest: str, url_primary: str, url_backup: str, md5: str, approx_mb: int) -> bool:
    """直链下载：先验 MD5，缺则下（支持 Range 断点续传），下完复验。"""
    import hashlib
    import urllib.request

    def digest(p: Path) -> str:
        h = hashlib.md5()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    dest = LAB / Path(rel_dest)
    if dest.is_file() and digest(dest) == md5:
        print(f"[{dest.name}]  已在 {rel_dest}（跳过）")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in (url_primary, url_backup):
        print(f"[{dest.name}]  <- {url[:70]} (~{approx_mb} MB)", flush=True)
        for attempt in range(1, 4):
            try:
                have = dest.stat().st_size if dest.is_file() else 0
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                if have:
                    req.add_header("Range", f"bytes={have}-")
                with urllib.request.urlopen(req, timeout=300) as r:
                    total = r.headers.get("Content-Length")
                    total = int(total) + have if total else None
                    mode = "ab" if have and r.status == 206 else "wb"
                    if mode == "wb":
                        have = 0
                    with open(dest, mode) as f:
                        while True:
                            chunk = r.read(1 << 20)
                            if not chunk:
                                break
                            f.write(chunk)
                            have += len(chunk)
                            if total and have % (25 * 2**20) < (1 << 20):
                                print(f"      {100 * have / total:5.1f}%  {have / 2**20:.0f} MB", flush=True)
                if total is None or have >= total:
                    break
            except Exception as e:
                print(f"      第{attempt}次失败: {type(e).__name__}: {str(e)[:60]}", flush=True)
                time.sleep(3)
        if dest.is_file() and digest(dest) == md5:
            print(f"      OK  {dest.stat().st_size / 2**20:.0f} MB")
            return True
        if dest.is_file():
            print("      !! MD5 不符，换下一个源")
            dest.unlink()
    print(f"      !! 下载失败: {rel_dest}")
    return False


def g2pw_ok(d: Path) -> bool:
    return d.is_dir() and all((d / f).is_file() for f in G2PW_NEED)


def fetch_g2pw() -> bool:
    """下 G2PWModel.zip 并解压到两处加载路径。"""
    missing = [d for d in G2PW_DIRS if not g2pw_ok(d)]
    if not missing:
        print("[G2PWModel]  两处均已就绪（跳过）")
        return True
    zip_path = LAB / "models" / "G2PWModel.zip"
    if not zip_path.is_file():
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[G2PWModel]  <- {G2PW_REPO}/{G2PW_ZIP} (~561 MB)")
        if not try_hf(G2PW_REPO, str(zip_path.parent), [G2PW_ZIP]):
            print("      !! 下载失败")
            return False
        _shutil_rmtree(zip_path.parent / ".cache")
    import zipfile
    for d in missing:
        print(f"[G2PWModel]  解压 -> {d.relative_to(LAB)}")
        d.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                # zip 顶层是 G2PWModel/，剥掉这层
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    rel = info.filename.split("/", 1)[1] if "/" in info.filename else info.filename
                    if not rel or rel.startswith("__MACOSX"):
                        continue
                    target = d / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, open(target, "wb") as out:
                        out.write(src.read())
        except Exception as e:
            print(f"      !! 解压失败: {type(e).__name__}: {str(e)[:100]}")
            return False
    ok = all(g2pw_ok(d) for d in missing)
    if ok and zip_path.is_file():
        zip_path.unlink()  # 561 MB，解压完就没用了
        print("      已删除 zip（省 561 MB）")
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
        print("目录权重 [OK]" if all_ok else "目录权重 [FAIL]")

    # ---- 单文件基座 s1/s2（上游 git 仓库不含）----
    print()
    for name, repo, filename, mb in FILE_TARGETS:
        dest = LAB / "models" / "pretrained" / name
        if args.check:
            ok = dest.is_file() and dest.stat().st_size > int(mb * 0.85) * 2**20
            print(f"[{name}]  {'完整' if ok else '缺失'}  ({dest})")
            all_ok = all_ok and ok
        else:
            all_ok = fetch_single(name, repo, filename, mb) and all_ok

    # ---- 直链目标（fast_langdetect，缺失会在模型加载期崩）----
    print()
    for rel, up, ub, want_md5, mb in URL_TARGETS:
        dest = LAB / Path(rel)
        if args.check:
            import hashlib
            h = hashlib.md5()
            if dest.is_file():
                with open(dest, "rb") as f:
                    for c in iter(lambda: f.read(1 << 20), b""):
                        h.update(c)
            ok = dest.is_file() and h.hexdigest() == want_md5
            print(f"[{dest.name}]  {'完整' if ok else '缺失/校验不过'}  ({rel})")
            all_ok = all_ok and ok
        else:
            all_ok = fetch_url(rel, up, ub, want_md5, mb) and all_ok

    # ---- G2PW 中文多音字（zip 发布，需解压到两处）----
    print()
    if args.check:
        for d in G2PW_DIRS:
            ok = g2pw_ok(d)
            print(f"[G2PWModel]  {'完整' if ok else '缺失'}  ({d})")
            all_ok = all_ok and ok
    else:
        all_ok = fetch_g2pw() and all_ok

    print()
    print("=" * 60)
    print("全部就绪 [OK]" if all_ok else "有缺失 [FAIL]")

    # ASR 侧（Qwen3-ASR）单独检查
    check_asr()
    return 0 if all_ok else 1


def check_asr() -> bool:
    """Qwen3-ASR 权重检查。它属于 ASR 工具链（.venv-asr），与 TTS 分开算。

    必须校验 model.safetensors 的**体积**：下载中途被打断会留下几百 MB 的残缺
    文件，只判"存在且够大"会误报完整，推理时才在加载权重处崩（实测踩过）。
    """
    asr_dir = LAB / "models" / "asr" / "Qwen3-ASR-0.6B"
    big = asr_dir / "model.safetensors"
    min_big = int(1.6 * 2**30)  # 真实 1.79 GB，留余量防上游换版
    needed = ["config.json", "vocab.json"]
    small_ok = all((asr_dir / f).is_file() and (asr_dir / f).stat().st_size > 1024
                   for f in needed)
    print()
    if big.is_file() and big.stat().st_size >= min_big and small_ok:
        print(f"[Qwen3-ASR-0.6B]  完整  ({dir_size(asr_dir)/2**20:.0f} MB)")
        return True
    if big.is_file():
        print(f"[Qwen3-ASR-0.6B]  残缺  model.safetensors "
              f"{big.stat().st_size/2**20:.0f} MB < 预期 {min_big/2**20:.0f} MB（下载被中断？）")
    else:
        print(f"[Qwen3-ASR-0.6B]  缺失  ({asr_dir})")
    print("  续传（urllib 支持 Range 断点续传，重跑即可接着下）：")
    print("    .venv-asr" + chr(92) + "Scripts" + chr(92) + "python.exe tools" + chr(92) + "fetch_qwen_asr.py")
    return False


if __name__ == "__main__":
    sys.exit(main())
