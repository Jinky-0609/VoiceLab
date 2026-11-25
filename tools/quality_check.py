#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""音质检测：把"听起来不对"变成可量化的数字。

    # 单条检测（对比参考音频判断音色是否漂移）
    .venv-tts/Scripts/python.exe tools/quality_check.py -f out/x.wav -r data/refs/ref.wav

    # 批量扫描最近产物
    .venv-tts/Scripts/python.exe tools/quality_check.py --scan out/web --limit 10

    # 只跑快速信号检查（不加载模型，秒级）
    .venv-tts/Scripts/python.exe tools/quality_check.py -f x.wav --fast

## 为什么需要它

实际排障中遇到过三类"数值看着正常但听起来不对"：
  1. int16 未归一化 -> 削顶噪音（mean|abs| 0.89，正常 0.04）
  2. 英文缺 nltk   -> 静音音频（mean|abs| 0.0008）
  3. v4 配 v3 LoRA -> 电音失真（高频占比异常偏高）

前两类现在有守卫拦，第三类没有客观指标。本工具补上：
  * **削顶率**：样本接近满量程的比例
  * **高频占比**：电音/刺耳时显著升高（实测 v3 6.9% vs 错配 9%+）
  * **频谱平坦度**：噪声化程度，失真会拉高
  * **音色相似度**（可选，需 torch）：与参考音频的 CNHuBERT 特征余弦相似度
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "tools"))

import numpy as np  # noqa: E402

SR = 24000
# 阈值来自实测：正常语音 vs 已知问题样本
TH = {
    "clip_max": 0.02,        # 削顶率 > 2% 视为有问题
    "silent_min": 0.02,      # mean|abs| < 0.02 视为过轻/静音
    "silent_hard": 0.004,    # 低于此值基本是哑音频
    "hf_max": 0.14,          # 高频占比 > 14% 提示电音（实测正常 5~9%）
    "flat_max": 0.55,        # 频谱平坦度上限（噪声化）
    # 音色相似度阈值：用 4 组样本标定（同色 0.827/0.883，跨色 0.584/0.713），
    # 间隙 0.114，取中值 0.770。低于它就说明音色明显偏离参考。
    "sim_min": 0.77,
}


def load_audio(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf
    a, sr = sf.read(path, dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    if sr != SR:
        n = int(round(len(a) * SR / sr))
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)
        sr = SR
    return a, sr


def signal_metrics(a: np.ndarray, sr: int) -> dict:
    """纯信号层指标，不需要任何模型。"""
    if len(a) == 0:
        return {"error": "空音频"}
    peak = float(np.abs(a).max())
    mean_abs = float(np.abs(a).mean())
    clip = float((np.abs(a) > 0.98).mean())

    # 频谱
    n_fft = 1024
    hop = 256
    if len(a) < n_fft:
        a = np.pad(a, (0, n_fft - len(a)))
    frames = []
    for i in range(0, len(a) - n_fft + 1, hop):
        w = a[i:i + n_fft] * np.hanning(n_fft)
        frames.append(np.abs(np.fft.rfft(w)))
    S = np.array(frames) if frames else np.zeros((1, n_fft // 2 + 1))
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    total = float(S.sum()) + 1e-9
    hf = float(S[:, freqs > 6000].sum() / total)

    # 频谱平坦度（几何均值/算术均值）：噪声化会接近 1
    Sm = S + 1e-10
    flat = float(np.exp(np.log(Sm).mean(axis=1)).mean() / (Sm.mean(axis=1).mean() + 1e-10))

    # 静音占比
    win = max(2, int(0.05 * sr))
    n = len(a) // win
    e = np.sqrt((a[:n * win].reshape(n, win) ** 2).mean(axis=1)) if n else np.array([0.0])
    sil = float((e < 0.02 * (e.max() + 1e-9)).mean())

    return {
        "duration": round(len(a) / sr, 2),
        "peak": round(peak, 4),
        "mean_abs": round(mean_abs, 5),
        "clip_ratio": round(clip, 5),
        "hf_ratio": round(hf, 4),
        "flatness": round(flat, 4),
        "silence_ratio": round(sil, 4),
    }


def judge(m: dict) -> tuple[str, list[str]]:
    """根据指标给出结论与原因。"""
    issues = []
    if m.get("error"):
        return "error", [m["error"]]
    if m["mean_abs"] < TH["silent_hard"]:
        issues.append(f"几乎静音（mean|abs|={m['mean_abs']}）")
    elif m["mean_abs"] < TH["silent_min"]:
        issues.append(f"音量偏轻（mean|abs|={m['mean_abs']}）")
    if m["clip_ratio"] > TH["clip_max"]:
        issues.append(f"削顶严重（{m['clip_ratio']*100:.1f}%）——可能是 int16 未归一化")
    if m["hf_ratio"] > TH["hf_max"]:
        issues.append(f"高频占比偏高（{m['hf_ratio']*100:.1f}%）——听起来会像电音/刺耳")
    if m["flatness"] > TH["flat_max"]:
        issues.append(f"频谱过于平坦（{m['flatness']}）——可能噪声化")
    return ("warn" if issues else "ok"), issues


def voice_similarity(a: np.ndarray, sr: int, ref: np.ndarray, ref_sr: int) -> float | None:
    """用 CNHuBERT 提取说话人特征做余弦相似度。

    这是"音色是否漂移"的客观依据。加载模型较重（~10s），所以只在
    显式要求时执行。失败返回 None（不阻断其他检查）。
    """
    try:
        import torch
    except ImportError:
        return None
    try:
        from transformers import HubertModel
        model_dir = LAB / "_src" / "GPT-SoVITS-main" / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base"
        if not model_dir.is_dir():
            return None
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        # 复用全局缓存，批量扫描时只加载一次
        global _HUBERT
        try:
            _HUBERT
        except NameError:
            _HUBERT = None
        if _HUBERT is None:
            _HUBERT = HubertModel.from_pretrained(str(model_dir)).to(dev).eval()
        m = _HUBERT

        def embed(x, r):
            if r != 16000:
                n = int(round(len(x) * 16000 / r))
                x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
            t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(dev)
            with torch.no_grad():
                h = m(t).last_hidden_state          # (1, T, 768)
            return h.mean(dim=1).squeeze(0)

        e1 = embed(a, sr)
        e2 = embed(ref, ref_sr)
        cos = float(torch.nn.functional.cosine_similarity(e1, e2, dim=0))
        return round(cos, 4)
    except Exception:
        return None


def check_one(path: str, ref: str | None, fast: bool, want_json: bool) -> dict:
    a, sr = load_audio(path)
    m = signal_metrics(a, sr)
    verdict, issues = judge(m)

    sim = None
    if ref and not fast:
        try:
            ra, rsr = load_audio(ref)
            sim = voice_similarity(a, sr, ra, rsr)
            if sim is not None and sim < TH["sim_min"]:
                issues.append(f"音色与参考偏离（相似度 {sim} < {TH['sim_min']}）")
                verdict = "warn"
        except Exception as e:
            issues.append(f"相似度计算失败: {type(e).__name__}")

    out = {"file": str(path), "verdict": verdict, "issues": issues, **m}
    if sim is not None:
        out["voice_similarity"] = sim

    if want_json:
        print(json.dumps(out, ensure_ascii=False))
    else:
        icon = {"ok": "✓", "warn": "!", "error": "✗"}[verdict]
        print(f"  {icon} {Path(path).name}")
        print(f"      {m.get('duration','?')}s | peak={m.get('peak')} | mean|abs|={m.get('mean_abs')} "
              f"| 削顶={m.get('clip_ratio',0)*100:.1f}% | 高频={m.get('hf_ratio',0)*100:.1f}% "
              f"| 平坦度={m.get('flatness')}")
        if sim is not None:
            print(f"      音色相似度: {sim}")
        for i in issues:
            print(f"      ⚠ {i}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="音质检测（削顶/电音/音色漂移）")
    ap.add_argument("-f", "--file", help="要检测的音频")
    ap.add_argument("-r", "--ref", help="参考音频（用于音色相似度）")
    ap.add_argument("--scan", help="扫描目录下最近的文件")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--fast", action="store_true", help="只做信号检查，不加载模型")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    if args.scan:
        d = Path(args.scan)
        if not d.is_dir():
            sys.exit(f"[FATAL] 目录不存在: {d}")
        files = sorted(d.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)[:args.limit]
        if not files:
            sys.exit(f"[FATAL] {d} 下没有 wav")
        results = [check_one(str(f), args.ref, args.fast, args.json) for f in files]
        bad = [r for r in results if r["verdict"] != "ok"]
        print(f"\n扫描 {len(files)} 个，其中 {len(bad)} 个有问题")
        return 0 if not bad else 1

    if not args.file:
        sys.exit("[FATAL] 需要 -f 或 --scan")
    out = check_one(args.file, args.ref, args.fast, args.json)
    return 0 if out["verdict"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
