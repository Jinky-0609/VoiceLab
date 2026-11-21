# -*- coding: utf-8 -*-
"""下载 Qwen3-ASR 权重（urllib + certifi，绕开 requests 的 SSL 证书问题）。

用法：
    .venv-asr\\Scripts\\python.exe tools\\fetch_qwen_asr.py              # 0.6B（默认）
    .venv-asr\\Scripts\\python.exe tools\\fetch_qwen_asr.py --size 1.7B  # 更大更准

为什么不用 huggingface_hub：本机 huggingface_hub 走的 requests 缺根证书，
访问 huggingface.co 会 SSLCertVerificationError；urllib 用 certifi 就正常。
"""
from __future__ import annotations

import argparse
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
    args = ap.parse_args()

    repo = REPOS[args.size]
    dst_dir = LAB / "models" / "asr" / f"Qwen3-ASR-{args.size}"
    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"目标: {dst_dir}")
    print(f"仓库: {repo}\n")

    base = f"https://huggingface.co/{repo}/resolve/main/"
    ok = True
    for fn in SMALL_FILES:
        target = dst_dir / fn
        if target.is_file() and target.stat().st_size > 0:
            continue
        if not download(base + fn, target):
            print(f"  (警告) {fn} 下载失败，通常不影响推理")

    big = dst_dir / BIG_FILE
    need = not (big.is_file() and big.stat().st_size > 100 * 2**20)
    if need:
        if not download(base + BIG_FILE, big):
            ok = False
            print("  !! 权重下载失败", flush=True)
    else:
        print(f"  {BIG_FILE} 已存在  {human(big.stat().st_size)}")

    print("\n=== 结果 ===")
    for f in sorted(dst_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name:32} {human(f.stat().st_size)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
