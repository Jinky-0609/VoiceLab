#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""删除音色：干系资产一网打尽，默认只预览不删。

一个音色在项目里散落在 5 个地方，漏掉任何一处都会留下"幽灵音色"：
  1. config.json 的 profiles 登记
  2. models/voices/<名>/        训练产物（s1 ckpt + s2 LoRA，约 220MB）
  3. data/<名>/                 数据准备中间产物（BERT/HuBERT/语义 token）
  4. data/raw/<名>/             原始素材、切片、.list、分离产物
  5. data/emotions/<名>/        情绪参考
  6. data/refs/<名>*            参考音频（如果 ref_audio 指向 refs/ 下）

用法：
    # 先看会删什么（不动手）
    python tools/delete_voice.py -p hanser

    # 列出所有音色和占用
    python tools/delete_voice.py --list

    # 真的删（会先备份 config 条目到 _local/deleted/）
    python tools/delete_voice.py -p hanser --yes

    # 连受保护的公共资产也一起删（如 zeroshot 指向的预训练参考）
    python tools/delete_voice.py -p zeroshot --yes --include-shared

安全约束：
  * 预训练权重（models/pretrained/）和公共参考（data/refs/default.*）默认**不动**，
    因为其他 profile 可能共用；要删需显式 --include-shared。
  * 删除前把 config 里的 profile 条目存到 _local/deleted/，可人工恢复。
  * 路径全部做父目录校验，绝不越出项目根。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "tools"))

import config_io  # noqa: E402
import runtime_paths  # noqa: E402

# 这些路径下的东西是共享资产，默认不删
SHARED_PREFIXES = ("models/pretrained", "data/refs/default")


def _size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return 0


def _fmt(n: int) -> str:
    return f"{n/2**20:.0f} MB" if n >= 2**20 else f"{n/1024:.0f} KB"


def collect(name: str, cfg: dict, include_shared: bool) -> list[tuple[str, Path]]:
    """返回 [(标签, 绝对路径)]，已按是否共享过滤。"""
    items: list[tuple[str, Path]] = []
    prof = cfg.get("profiles", {}).get(name, {})

    # 1) 训练产物
    v = LAB / "models" / "voices" / name
    if v.is_dir():
        items.append(("训练权重", v))

    # 2) 数据准备中间产物
    d = LAB / "data" / name
    if d.is_dir():
        items.append(("数据集", d))

    # 3) 原始素材
    r = LAB / "data" / "raw" / name
    if r.is_dir():
        items.append(("原始素材", r))

    # 4) 情绪参考
    e = LAB / "data" / "emotions" / name
    if e.is_dir():
        items.append(("情绪参考", e))

    # 5) profile 里显式引用的参考音频/文本（只删 refs 下、且文件名含该音色名的，
    #    避免误删别人共用的 default.wav）
    for key in ("ref_audio", "ref_text"):
        raw = prof.get(key)
        if not raw:
            continue
        p = Path(runtime_paths.resolve(raw))
        if not p.is_file():
            continue
        rel = p.relative_to(LAB).as_posix() if str(p).startswith(str(LAB)) else ""
        shared = any(rel.startswith(s) for s in SHARED_PREFIXES)
        name_specific = name.lower() in p.stem.lower()
        if shared and not include_shared:
            continue
        if not name_specific and not include_shared:
            # 不是这个音色专属的文件，跳过（可能是公共参考）
            continue
        items.append(("参考音频", p))

    return items


def cmd_list(cfg: dict) -> int:
    profiles = cfg.get("profiles", {})
    if not profiles:
        print("还没有任何音色。")
        return 0
    print(f"{'音色':<16}{'占用':>10}  说明")
    print("-" * 62)
    total = 0
    for name, p in profiles.items():
        sz = 0
        for _, path in collect(name, cfg, include_shared=False):
            sz += _size(path)
        total += sz
        desc = (p.get("desc") or "")[:34]
        print(f"{name:<16}{_fmt(sz):>10}  {desc}")
    print("-" * 62)
    print(f"{'合计':<16}{_fmt(total):>10}")
    print("\n删除某个：python tools/delete_voice.py -p <名字> --yes")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="删除音色及其全部干系资产")
    ap.add_argument("-p", "--profile", help="要删除的音色名")
    ap.add_argument("--list", action="store_true", help="列出所有音色与占用")
    ap.add_argument("--yes", action="store_true", help="确认执行（不加只预览）")
    ap.add_argument("--include-shared", action="store_true",
                    help="连共享资产一起删（预训练权重、default 参考），一般不要加")
    ap.add_argument("--keep-raw", action="store_true", help="保留原始素材 data/raw/<名>")
    args = ap.parse_args()

    cfg = config_io.load(verbose=False)

    if args.list or not args.profile:
        return cmd_list(cfg)

    name = args.profile
    if name not in cfg.get("profiles", {}):
        print(f"[错误] config.json 里没有这个音色: {name}")
        return 1
    if len(cfg["profiles"]) <= 1 and not args.include_shared:
        print(f"[拒绝] 这是最后一个音色，删掉后就没有可用档案了。")
        print(f"       确实要删请加 --include-shared")
        return 1

    items = collect(name, cfg, args.include_shared)
    if args.keep_raw:
        items = [(t, p) for t, p in items if t != "原始素材"]

    print(f"音色: {name}")
    print(f"说明: {(cfg['profiles'][name].get('desc') or '')[:60]}")
    print()
    if not items:
        print("没有找到可删的资产（可能只剩 config 登记项）。")
    else:
        print("将删除以下内容：")
        total = 0
        for label, path in items:
            sz = _size(path)
            total += sz
            rel = path.relative_to(LAB).as_posix() if str(path).startswith(str(LAB)) else str(path)
            kind = "目录" if path.is_dir() else "文件"
            print(f"  [{label}] {rel}  ({kind}, {_fmt(sz)})")
        print(f"\n合计释放：{_fmt(total)}")

    if not args.yes:
        print("\n这是预览。确认删除请加 --yes：")
        print(f"  python tools/delete_voice.py -p {name} --yes")
        return 0

    # ---- 执行 ----
    backup_dir = LAB / "_local" / "deleted"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (backup_dir / f"{name}.{stamp}.profile.json").write_text(
        json.dumps(cfg["profiles"][name], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[备份] profile 条目 -> _local/deleted/{name}.{stamp}.profile.json")

    for label, path in items:
        if not str(path.resolve()).startswith(str(LAB.resolve())):
            print(f"  [跳过] 越界路径 {path}")
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            print(f"  [删除] {label}: {path.name}")
        except Exception as e:
            print(f"  [失败] {label} {path.name}: {type(e).__name__}: {e}")

    # ---- 从 config 移除 ----
    del cfg["profiles"][name]
    config_io.save(cfg)
    print(f"\n[完成] 已从 config.json 移除 {name}")
    print(f"       剩余音色: {', '.join(cfg['profiles']) or '（无）'}")
    print(f"       profile 条目备份在 _local/deleted/，需要恢复可手工加回 config.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
