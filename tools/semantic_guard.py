#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""语义坍缩守卫（音频层）—— VoiceLab 自有实现，不依赖任何外部项目。

背景：v3 模型在「参考音频语言与目标语言不一致」等场景下会进入 token 坍缩，
表现为循环乱音（实测 28 字文本生成 52.9 秒）。旧实现挂在别的项目的
semantic_stability 模块上，该模块已不存在，守卫随之消失——本文件把它补回来。

与旧版差别：旧版在 T2S token 层拦截（equal_token_collapse /
periodic_token_collapse / length_with_repetition），需要在采样循环内打桩；
本模块在**音频层**按症状判定，效果一致（拒绝保存垃圾音频），但零侵入、
上游升级不破坏。三条规则全部用 out/ 里 81 个历史样本校准过：

    1) length_explosion      时长远超文本可解释范围
       dur > max(10, 0.8 * 字数)         正常 0.19 s/字，乱音 1.9 s/字
    2) periodic_repetition   频谱帧间自相似出现近完美循环
       autocorr >= 0.72 且 dur >= 2s     乱音 0.828，正常样本 p99 = 0.597
    3) no_pause_drone        长时间无停顿的连续音墙
       dur >= 12s 且 静音帧占比 < 3%     乱音 52.9s 仅 0.5% 低于峰值 2% 的帧

另有一条数值完整性检查（saturated_audio / non_finite_audio）：正常语音
mean|abs| ≈ 0.04、峰值 ≈ 0.5；若接近满量程方波（mean|abs| > 0.5）说明
量化/归一化出错，实测 int16 被当 float 写盘时该值为 0.89。

阈值可在此文件或通过 voicelab.py --guard-attempts / --no-guard 调整。
"""
from __future__ import annotations

import numpy as np

# ---- 阈值（校准记录见模块 docstring）----
LEN_FACTOR = 0.8      # 秒/字 上限系数
LEN_FLOOR = 10.0      # 秒，短文本绝对下限
ACORR_TH = 0.72       # 帧间自相似
ACORR_MIN_DUR = 2.0   # 秒，短于此不做循环判定（统计不可靠）
DRONE_DUR = 12.0      # 秒
DRONE_SIL = 0.03      # 静音帧占比上限
SATURATED_MEAN = 0.5  # mean|abs| 上限：正常 ~0.04，削顶方波 ~0.89
NEAR_SILENT_MEAN = 0.004  # mean|abs| 下限：正常语音 ≥0.02；英文缺 nltk 时实测 0.00078


class SemanticGenerationError(RuntimeError):
    """生成坍缩：多次尝试后音频仍不通过守卫。"""


def _stft_frames(x: np.ndarray, sr: int) -> np.ndarray:
    hop = max(1, int(0.020 * sr))
    win = max(2, int(0.050 * sr))
    n = 1 + max(0, (len(x) - win) // hop)
    if n < 10:
        return np.zeros((0, 2))
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    fr = x[idx] * np.hanning(win)
    return np.abs(np.fft.rfft(fr, 1024, axis=1))


def analyze(audio, sr: int, nchars: int) -> list[str]:
    """返回违规原因列表；空 = 通过。

    audio: 一维数组，**float [-1, 1] 量纲**（tts_engine 已负责归一化）。
    voicelab 依据结果决定换 seed 重试还是报错。
    """
    raw = np.asarray(audio).ravel()
    if raw.size == 0:
        return ["empty_audio"]

    # 数值完整性：先抓量纲/量化错误，这类问题换 seed 也无意义
    if not np.all(np.isfinite(raw)):
        return ["non_finite_audio"]
    x = raw.astype(np.float32)
    peak = float(np.abs(x).max())
    if peak > 1.5:
        # 仍是 int16 之类的错误量纲
        return ["bad_scale_audio"]
    reasons: list[str] = []
    if float(np.abs(x).mean()) > SATURATED_MEAN:
        reasons.append("saturated_audio")

    # 近静音：英文前端缺 nltk 数据时，整条链路零异常但产出哑音频
    # （实测 0.84s / peak 0.0094 / mean|abs| 0.00078）。这类结果必须拦下。
    mean_abs = float(np.abs(x).mean())
    if mean_abs < NEAR_SILENT_MEAN:
        reasons.append("near_silent")

    dur = len(x) / float(sr)

    # 1) 长度爆炸
    if dur > max(LEN_FLOOR, LEN_FACTOR * max(1, nchars)):
        reasons.append("length_explosion")

    # 2) 周期循环
    if dur >= ACORR_MIN_DUR:
        S = _stft_frames(x, sr)
        if len(S) > 20:
            Sp = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-9)
            hop_s = 0.020
            lag0 = max(5, int(0.10 / hop_s))
            lag1 = min(len(S) - 1, int(3.0 / hop_s))
            best = 0.0
            for lag in range(lag0, lag1):
                best = max(best, float((Sp[: len(S) - lag] * Sp[lag:]).sum(1).mean()))
            if best >= ACORR_TH:
                reasons.append("periodic_repetition")

    # 3) 无声墙（长时连续、无呼吸停顿）
    if dur >= DRONE_DUR:
        win = max(2, int(0.050 * sr))
        n = len(x) // win
        e = np.sqrt((x[: n * win].reshape(n, win) ** 2).mean(1))
        if n and float((e < 0.02 * e.max()).mean()) < DRONE_SIL:
            reasons.append("no_pause_drone")

    return reasons


def check(audio, sr: int, nchars: int) -> None:
    """不通过则抛 ValueError（单次尝试用）；voicelab 负责换 seed 重试。"""
    r = analyze(audio, sr, nchars)
    if r:
        raise ValueError(",".join(r))

