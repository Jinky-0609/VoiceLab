#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""文本情绪分段（实验性）：把长文本切成小句，逐句判定情绪，再拼接。

    .venv-tts/Scripts/python.exe tools/auto_emotion.py -t "你怎么能这样！我很失望……"

## 设计取舍（重要，别指望它是 LLM）

本项目没有本地 LLM，也不联网。所以这里是**规则 + 词典**方案：
  * 明确、可解释、零依赖、毫秒级
  * 对"感叹/疑问/省略号/哭声/笑声/语气词"这类**显式**信号很准
  * 对反讽、含蓄表达会误判——这是这类方法的固有上限

因此它被定位为**实验性**：生成后要试听，不满意可手动改每句的情绪再合成。

## 情绪如何落到音频

每句合成时把对应情绪的参考音频作为 aux 参考传给引擎（与手动选情绪同一机制），
逐段生成后用等功率交叉淡入拼接，避免爆音。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "tools"))

# ---------------------------------------------------------------- 规则词典

# 词 -> 情绪。命中越具体优先级越高；一句里取命中最多的情绪。
LEXICON = {
    "开心": ["哈哈", "嘻嘻", "开心", "高兴", "太好了", "棒", "真棒", "喜欢", "爱了",
             "有趣", "好玩", "顺利", "成功", "恭喜", "贺", "笑", "嘿嘿", "哇", "耶"],
    "生气": ["混蛋", "滚", "闭嘴", "气死", "可恶", "讨厌", "烦", "过分", "凭什么",
             "搞什么", "岂有此理", "怒", "恨", "你敢", "少来"],
    "悲伤": ["难过", "伤心", "失望", "哭", "眼泪", "遗憾", "可惜", "算了",
             "没办法", "对不起", "抱歉", "失去", "走了", "回不来", "唉", "呜"],
    "激动": ["快", "冲", "上啊", "来了", "终于", "竟然", "居然", "天哪", "我的天",
             "不敢相信", "太厉害", "震惊"],
    "温柔": ["谢谢你", "慢慢", "没关系", "别怕", "放心", "乖", "亲爱", "轻轻",
             "好好休息", "照顾", "抱抱"],
    "严肃": ["记住", "注意", "必须", "严禁", "规定", "命令", "警告", "重要",
             "认真", "否则"],
    "耳语": ["悄悄", "小声", "嘘", "秘密", "偷偷", "别告诉"],
    "平静": [],   # 兜底
}

# 标点/符号的强信号
# (正则, 情绪, 权重)。单符号权重低，连续符号权重高。
PUNCT_SIGNAL = [
    (re.compile(r"[!！]{2,}"), "激动", 2),
    (re.compile(r"[!！]"), "激动", 1),
    (re.compile(r"\.{3,}|。{3,}|…"), "悲伤", 2),
    (re.compile(r"[?？]{2,}"), "困惑", 2),
    (re.compile(r"[哈嘿嘻]{2,}"), "开心", 2),
    (re.compile(r"[呜哭]{2,}"), "悲伤", 2),
    (re.compile(r"[怒哼]{2,}"), "生气", 2),
    (re.compile(r"[~～]{2,}"), "开心", 1),
]

# 句末标点切分
SENT_SPLIT = re.compile(r"(?<=[。！？；!?;\n])")


def split_sentences(text: str, max_chars: int = 60) -> list[str]:
    """按句末标点切句；过长的再按逗号切。"""
    out = []
    for part in SENT_SPLIT.split(text):
        part = part.strip()
        if not part:
            continue
        if len(part) <= max_chars:
            out.append(part)
            continue
        # 长句按次级标点再切
        buf = ""
        for seg in re.split(r"(?<=[，,、])", part):
            if len(buf) + len(seg) <= max_chars:
                buf += seg
            else:
                if buf:
                    out.append(buf.strip())
                buf = seg
        if buf.strip():
            out.append(buf.strip())
    return out


def detect_emotion(sentence: str) -> tuple[str, int, list[str]]:
    """返回 (情绪, 置信分数, 命中依据)。分数 0 表示无明显倾向（落到平静）。"""
    hits: list[str] = []
    scores: dict[str, int] = {}

    for emo, words in LEXICON.items():
        for w in words:
            if w and w in sentence:
                scores[emo] = scores.get(emo, 0) + 1
                hits.append(f"词:{w}")

    for pat, emo, weight in PUNCT_SIGNAL:
        if pat.search(sentence):
            scores[emo] = scores.get(emo, 0) + weight
            hits.append(f"符:{pat.pattern}x{weight}")

    if not scores:
        return "平静", 0, []

    best = max(scores.items(), key=lambda kv: kv[1])
    extra = sum(v for k, v in scores.items() if k != best[0])
    # 置信度：领先度归一化到 0~1
    conf = round((best[1] - extra * 0.5) / max(best[1], 1), 2)
    return best[0], max(conf, 0.0), hits


def plan(text: str, available: list[str], max_chars: int = 60) -> list[dict]:
    """把文本切成句子并给出每句的情绪建议。

    available：该音色已登记的情绪名。没登记的情绪会回退到"平静"（=不加情绪）。
    """
    avail = set(available)
    segs = []
    for i, s in enumerate(split_sentences(text, max_chars), 1):
        emo, conf, hits = detect_emotion(s)
        usable = emo in avail
        if not usable:
            emo_final = ""
            note = f"未登记「{emo}」情绪，将用默认音色"
        else:
            emo_final = emo
            note = ""
        segs.append({
            "i": i, "text": s, "chars": len(s),
            "emotion": emo_final,
            "suggest": emo,
            "confidence": conf,
            "hits": hits[:4],
            "note": note,
        })
    return segs


# ---------------------------------------------------------------- 音频拼接

def concat_with_crossfade(paths: list[Path], out_path: Path,
                          sr: int = 24000, fade_ms: int = 45) -> Path:
    """等功率交叉淡入拼接，避免段间爆音。

    交叉淡入而不是硬拼接：硬切会在拼接点产生"咔"声（波形不连续）。
    """
    import numpy as np
    import soundfile as sf

    clips = []
    for p in paths:
        a, r = sf.read(str(p), dtype="float32", always_2d=True)
        a = a.mean(axis=1)
        if r != sr:
            n = int(round(len(a) * sr / r))
            a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)
        clips.append(a)

    if not clips:
        raise ValueError("没有可拼接的音频")

    fade = max(1, int(sr * fade_ms / 1000))
    result = clips[0]
    for nxt in clips[1:]:
        f = min(fade, len(result), len(nxt))
        if f < 8:
            result = np.concatenate([result, nxt])
            continue
        # 等功率曲线
        t = np.linspace(0, np.pi / 2, f, dtype=np.float32)
        a_win = np.cos(t)
        b_win = np.sin(t)
        head = result[-f:] * a_win
        tail = nxt[:f] * b_win
        result = np.concatenate([result[:-f], head + tail, nxt[f:]])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), result, sr, subtype="PCM_16")
    return out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="文本情绪分段（实验性，规则+词典）")
    ap.add_argument("-t", "--text", required=True)
    ap.add_argument("-p", "--profile", help="音色名（用于检查已登记情绪）")
    ap.add_argument("--max-chars", type=int, default=60)
    args = ap.parse_args()

    import emotions
    avail = list(emotions.load(args.profile).keys()) if args.profile else []
    if args.profile:
        print(f"该音色已登记情绪: {', '.join(avail) or '（无）'}\n")

    for s in plan(args.text, avail, args.max_chars):
        tag = s["emotion"] or f"({s['suggest']}-未登记)"
        print(f"[{s['i']}] {tag:12s} conf={s['confidence']:.2f}  {s['text']}")
        if s["hits"]:
            print(f"      依据: {', '.join(s['hits'])}")
