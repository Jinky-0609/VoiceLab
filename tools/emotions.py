#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""情绪参考库 + 管理 CLI。

## v3 的情绪是怎么来的（先读这段）

GPT-SoVITS v3 **没有** emotion 参数。情绪完全由**参考音频的韵律**承载：
低沉的参考给低沉，带笑的参考给微笑语调。上游提供 `aux_ref_audio_paths`
（TTS.py:1149/1275-1286）把多段参考的特征一起融合，这是唯一的情绪通道。

所以"给某音色加情绪"= 为这个音色登记一段**同一人、带该情绪**的短音频。

## 诚实边界（前端与文档都如实标注）

* 效果强弱取决于参考音频本身，不是强保证。平庸的参考给不出戏剧化效果。
* 参考必须是**同一个人**，否则是在做音色混合而非情绪迁移。
* 不给情绪参考时，行为与原来完全一致（不破坏已有用法）。

## 用法

作为库（voicelab.py 会 import）：load / save / add / remove / resolve

作为 CLI：

    python tools/emotions.py add -p zhongcheng -n 开心 -f 开心录音.wav -t "音频里说的逐字文本"
    python tools/emotions.py list -p zhongcheng
    python tools/emotions.py remove -p zhongcheng -n 开心

    # 然后合成时
    python tools/voicelab.py speak -p zhongcheng -t "你好" --emotion 开心
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
EMO_ROOT = LAB / "data" / "emotions"

BUILTIN = ["平静", "开心", "生气", "悲伤", "温柔", "激动", "严肃", "耳语"]

SAFE = re.compile(r"[\w\u4e00-\u9fff\-]{1,32}")


# ------------------------------------------------------------------ 库接口

def _spk_dir(speaker: str) -> Path:
    if not SAFE.fullmatch(speaker or ""):
        raise ValueError(f"非法的音色名: {speaker!r}（只能用中英文/数字/下划线/连字符）")
    return EMO_ROOT / speaker


def _meta_path(speaker: str) -> Path:
    return _spk_dir(speaker) / "emotions.json"


def load(speaker: str) -> dict:
    """读某音色的情绪表；没有则返回空 dict。"""
    try:
        p = _meta_path(speaker)
    except ValueError:
        return {}
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save(speaker: str, data: dict) -> None:
    d = _spk_dir(speaker)
    d.mkdir(parents=True, exist_ok=True)
    _meta_path(speaker).write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def add(speaker: str, name: str, audio_rel: str, text: str = "", lang: str = "中文") -> dict:
    """登记一条情绪参考。audio_rel 为项目相对路径。"""
    if not SAFE.fullmatch(name or ""):
        raise ValueError("情绪名只能用中英文/数字/下划线/连字符，最长 32 字符")
    ap = LAB / audio_rel
    if not ap.is_file():
        raise ValueError(f"音频不存在: {audio_rel}")
    data = load(speaker)
    data[name] = {
        "audio": str(audio_rel).replace("\\", "/"),
        "text": (text or "").strip(),
        "lang": lang or "中文",
        "added": time.strftime("%Y-%m-%d %H:%M"),
    }
    save(speaker, data)
    return data[name]


def remove(speaker: str, name: str) -> bool:
    data = load(speaker)
    if name not in data:
        return False
    del data[name]
    save(speaker, data)
    return True


def resolve(speaker: str, names: list[str]) -> list[dict]:
    """把情绪名解析成可直接喂给引擎的条目，忽略缺失项。"""
    data = load(speaker)
    out = []
    for n in names or []:
        it = data.get(n)
        if not it:
            continue
        ap = LAB / it.get("audio", "")
        if ap.is_file():
            out.append({"name": n, "audio": str(ap),
                        "text": it.get("text", ""), "lang": it.get("lang", "中文")})
    return out


# ------------------------------------------------------------------ CLI

def cmd_add(a) -> int:
    src = Path(a.file)
    if not src.is_file():
        sys.exit(f"[FATAL] 音频不存在: {src}")

    try:
        import soundfile as sf
        dur = sf.info(str(src)).duration
    except Exception as e:
        sys.exit(f"[FATAL] 读不了音频: {type(e).__name__}: {e}")

    if dur < 1.5:
        print(f"[警告] 只有 {dur:.1f}s，太短会让效果不稳定（建议 3~10s）")
    elif dur > 15:
        print(f"[警告] 有 {dur:.1f}s，太长会稀释情绪特征（建议 3~10s），可先切短")
    else:
        print(f"[检查] 时长 {dur:.1f}s ✓")

    dest_dir = EMO_ROOT / a.profile
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{a.name}.wav"

    if src.suffix.lower() != ".wav":
        try:
            import soundfile as sf
            data, sr = sf.read(str(src), dtype="float32", always_2d=True)
            sf.write(str(dest), data, sr, subtype="PCM_16")
            print(f"[转换] {src.suffix} -> wav")
        except Exception as e:
            sys.exit(f"[FATAL] 格式转换失败: {e}")
    else:
        import shutil
        shutil.copyfile(src, dest)

    rel = str(dest.relative_to(LAB)).replace("\\", "/")
    emotions_entry = add(a.profile, a.name, rel, a.text or "", a.lang)
    print(f"[完成] {a.profile} / {a.name} -> {rel}  ({emotions_entry['added']})")
    print(f"       合成：--emotion {a.name}")
    if not a.text:
        print("[提示] 没给 --text。融合不依赖它，但填上更稳。")
    return 0


def cmd_list(a) -> int:
    data = load(a.profile)
    if not data:
        print(f"{a.profile} 还没有登记任何情绪参考。")
        print(f"  登记：python tools/emotions.py add -p {a.profile} -n 开心 -f 录音.wav -t \"逐字文本\"")
        print(f"  内置名字参考：{' / '.join(BUILTIN)}")
        return 0

    print(f"{a.profile} 的情绪参考（{len(data)} 个）：\n")
    for name, it in data.items():
        ap = LAB / it.get("audio", "")
        try:
            import soundfile as sf
            dur = f"{sf.info(str(ap)).duration:.1f}s" if ap.is_file() else "文件缺失"
        except Exception:
            dur = "?"
        print(f"  [{name}] {dur}")
        print(f"      {it.get('audio')}")
        print(f"      text: {it.get('text') or '(空)'}")
        print(f"      登记: {it.get('added','')}\n")

    first = list(data)[0]
    print(f"合成：python tools/voicelab.py speak -p {a.profile} -t \"...\" --emotion {first}")
    print("可叠加多个：--emotion 开心 --emotion 激动")
    return 0


def cmd_remove(a) -> int:
    if remove(a.profile, a.name):
        print(f"[完成] 已移除 {a.profile} / {a.name}（音频文件保留在 data/emotions/）")
        return 0
    sys.exit(f"[FATAL] 没有这个情绪: {a.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description="管理情绪参考音频（v3 的情绪由参考音频承载）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("add", help="登记一条情绪参考")
    p.add_argument("-p", "--profile", required=True, help="音色档案名")
    p.add_argument("-n", "--name", required=True, help="情绪名（如 开心/生气/温柔）")
    p.add_argument("-f", "--file", required=True, help="参考音频（3~10 秒、单人、干声）")
    p.add_argument("-t", "--text", help="该音频里的逐字文本")
    p.add_argument("-l", "--lang", default="中文", help="参考音频语言（默认 中文）")
    p.set_defaults(func=cmd_add)

    q = sub.add_parser("list", help="列出某音色的情绪参考")
    q.add_argument("-p", "--profile", required=True)
    q.set_defaults(func=cmd_list)

    r = sub.add_parser("remove", help="删除情绪登记")
    r.add_argument("-p", "--profile", required=True)
    r.add_argument("-n", "--name", required=True)
    r.set_defaults(func=cmd_remove)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
