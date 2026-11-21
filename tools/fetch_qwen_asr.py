# -*- coding: utf-8 -*-
"""下载 Qwen3-ASR 权重（urllib + certifi，绕开 requests 的 SSL 证书问题）。

用法：
    .venv-asr\\Scripts\\python.exe tools\\fetch_qwen_asr.py              # 0.6B（默认）
    .venv-asr\\Scripts\\python.exe tools\\fetch_qwen_asr.py --size 1.7B  # 更大更准
    .venv-asr\\Scripts\\python.exe tools\\fetch_qwen_asr.py --endpoint https://hf-mirror.com

为什么不用 huggingface_hub：它的 requests 链路缺根证书时会
SSLCertVerificationError；urllib + certifi 不受影响。

源站：默认自动探测 —— 先试 huggingface.co 直连（境外快），
不通或超时则切 hf-mirror.com（境内可用）。给了 --endpoint 或设了
环境变量 HF_ENDPOINT 时直接采用，大文件失败还会自动换另一个源重试。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]

REPOS = {
    "0.6B": "Qwen/Qwen3-ASR-0.6B",
    "1.7B": "Qwen/Qwen3-ASR-1.7B",
}
SMALL_FILES = [
    "config.json", "generation_config.json", "preprocessor_config.json",
    "tokenizer_config.json", "chat_template.json", "vocab.json", "merges.txt",
]
BIG_FILE = "model.safetensors"
OFFICIAL = "https://huggingface.co"
MIRROR = "https://hf-mirror.com"


def resolve_endpoint(explicit: str | None = None) -> str:
    """确定首选源站：--endpoint > 环境变量 HF_ENDPOINT > 自动探测。"""
    for src, label in ((explicit, "--endpoint"),
                       (os.environ.get("HF_ENDPOINT"), "HF_ENDPOINT")):
        if src:
            src = src.rstrip("/")
            print(f"源站: {src}  (来自 {label})", flush=True)
            return src

    probe = f"{OFFICIAL}/{REPOS['0.6B']}/resolve/main/{SMALL_FILES[0]}"
    print("探测 huggingface.co 直连 ...", end="", flush=True)
    try:
        req = urllib.request.Request(probe, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            if r.status == 200:
                r.read(64)
                print(" 可用", flush=True)
                return OFFICIAL
            print(f" HTTP {r.status}", end="", flush=True)
    except Exception as e:
        print(f" {type(e).__name__}", end="", flush=True)
    print(f"，改用镜像 {MIRROR}", flush=True)
    return MIRROR


def human(n: float) -> str:
    return f"{n / 2**20:.1f} MB"


def download(url: str, dst: Path, label: str = "") -> bool:
    for attempt in range(1, 6):
        have = dst.stat().st_size if dst.is_file() else 0
        headers = {"User-Agent": "Mozilla/5.0"}
        if have:
            headers["Range"] = f"bytes={have}-"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                total = r.headers.get("Content-Length")
                total = int(total) + have if total else None
                mode = "ab" if have and r.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                t0 = time.time()
                with open(dst, mode) as f:
                    while True:
                        chunk = r.read(1024 * 512)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        if total and have % (50 * 2**20) < 512 * 1024:
                            sp = have / max(time.time() - t0, 0.1) / 2**20
                            print(f"    {100*have/total:5.1f}%  {human(have)}  {sp:.1f}MB/s", flush=True)
            size = dst.stat().st_size
            if total is None or size >= total:
                print(f"  {label or dst.name} OK  {human(size)}", flush=True)
                return True
        except Exception as e:
            print(f"  {label or dst.name} 第{attempt}次失败: {type(e).__name__}: {str(e)[:70]}", flush=True)
            time.sleep(3)
    return dst.is_file() and dst.stat().st_size > 1024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="0.6B", choices=list(REPOS), help="模型规格")
    ap.add_argument("--endpoint", help="强制指定 HF 源站（默认自动探测，失败自动换源）")
    args = ap.parse_args()

    repo = REPOS[args.size]
    dst_dir = LAB / "models" / "asr" / f"Qwen3-ASR-{args.size}"
    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"目标: {dst_dir}")
    print(f"仓库: {repo}\n")

    first = resolve_endpoint(args.endpoint)
    endpoints = [first] + [e for e in (OFFICIAL, MIRROR) if e != first]
    base = f"{first}/{repo}/resolve/main/"
    ok = True
    for fn in SMALL_FILES:
        target = dst_dir / fn
        if target.is_file() and target.stat().st_size > 0:
            continue
        if not download(base + fn, target):
            print(f"  (警告) {fn} 下载失败，通常不影响推理")

    big = dst_dir / BIG_FILE
    # 完整性按体积判定：0.6B 真实 1.79 GB / 1.7B 更大。
    # 只判 ">100MB" 会把下载中断留下的半截文件当成已就绪（实测踩过 184MB 截断）。
    min_big = {"0.6B": int(1.6 * 2**30), "1.7B": int(3.2 * 2**30)}[args.size]
    if big.is_file() and big.stat().st_size < min_big:
        print(f"  {BIG_FILE} 残缺（{big.stat().st_size/2**20:.0f} MB < 预期 "
              f"{min_big/2**20:.0f} MB），断点续传", flush=True)
    need = not (big.is_file() and big.stat().st_size >= min_big)
    if need:
        got = False
        for ep in endpoints:
            if download(f"{ep}/{repo}/resolve/main/{BIG_FILE}", big):
                got = True
                break
            print(f"  !! {ep} 取 {BIG_FILE} 失败，尝试下一个源", flush=True)
        if not got:
            ok = False
            print("  !! 权重下载失败（两个源都试过）", flush=True)
    else:
        print(f"  {BIG_FILE} 已存在  {human(big.stat().st_size)}")

    print("\n=== 结果 ===")
    for f in sorted(dst_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name:32} {human(f.stat().st_size)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
