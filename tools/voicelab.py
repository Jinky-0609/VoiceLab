#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VoiceLab —— 独立音色实验室 CLI

设计原则：
  * 完全自包含：推理走本地上游仓库 _src/GPT-SoVITS-main（见 tools/tts_engine.py），
    预训练权重与配置路径全部**相对项目根**，整个目录可任意移动或换机器；
  * 运行时缓存（nltk / HF）落在项目内 runtime/，不写用户主目录；
  * 输出默认写入 <项目根>/out；
  * 音色/语言全部显式传参；
  * 合成结果过语义坍缩守卫（tools/semantic_guard.py），拦下乱音。

用法示例：
  python voicelab.py list
  .venv-tts/Scripts/python.exe tools/voicelab.py speak -p myvoice -t "你好，这是测试。" -l 中文
  .venv-tts/Scripts/python.exe tools/voicelab.py batch -p myvoice -i lines.jsonl -d out/batch01
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "tools"))

import runtime_paths  # noqa: E402
import semantic_guard  # noqa: E402

CONFIG_PATH = LAB / "config.json"
OUT_DIR = LAB / "out"


def _sovits_version(sovits_path: str) -> str:
    """从 .pth 魔数判版本（与 tts_engine._detect_version 同规则）。"""
    vmap = {b"03": "v3", b"04": "v4", b"05": "v2Pro", b"06": "v2ProPlus",
            b"07": "v5dev", b"08": "v5turbo"}
    try:
        with open(sovits_path, "rb") as f:
            return vmap.get(f.read(2), "v3")
    except OSError:
        return "v3"


def _is_pretrained_profile(p: dict) -> bool:
    """纯预训练对（零样本）才允许换基座。"""
    return "pretrained" in str(p.get("gpt", "")).lower()


# ---------------------------------------------------------------- config

def load_config() -> dict:
    """读 config.json；缺失时从 config.example.json 自动引导一份（见 config_io）。"""
    import config_io
    return config_io.load()


def read_text_arg(value: str | None, fallback: str | None = None) -> str | None:
    """允许传「文本」或「文本文件路径」两种形式。"""
    if value is None:
        return fallback
    p = Path(value)
    if p.is_file():
        return p.read_text(encoding="utf-8").strip()
    return value.strip()


# ---------------------------------------------------------------- engine

# 引擎缓存：按 (gpt, sovits) 做 key。
#
# 曾经是单个全局 _ENGINE 无条件复用，导致两个真 bug：
#   1. 切换音色无效——换 profile 后仍用上一个音色的权重合成；
#   2. 界面"预热引擎"形同虚设——预热 A 后合成 B，缓存把 B 的请求也返回成 A
#      （或反过来触发一次意料之外的重载，用户看到"预热了还是要等几十秒"）。
# 现在按权重路径缓存，一次只保留最近一个（语音模型很吃显存，多留会 OOM）。
_ENGINE = None
_ENGINE_KEY: tuple[str, str] | None = None


def get_engine(cfg: dict, gpt: str, sovits: str):
    """按 (gpt, sovits) 缓存引擎；权重变了必须重新加载。"""
    global _ENGINE, _ENGINE_KEY
    key = (str(Path(gpt).resolve()), str(Path(sovits).resolve()))
    if _ENGINE is not None and _ENGINE_KEY == key:
        return _ENGINE

    from tts_engine import TTSInferencer  # noqa: E402

    runtime_paths.ensure_env()

    # 上游 tools/i18n/i18n.py 用 os.path.relpath(__file__)，要求 cwd 与代码同盘符。
    os.chdir(LAB)

    # 换权重前先把旧引擎释放掉，否则 8GB 卡上会两份权重叠加导致 OOM
    if _ENGINE is not None:
        old_ver = getattr(_ENGINE, "model_version", "?")
        print(f"[切换] 释放旧引擎（{old_ver}）以便加载新权重", flush=True)
        try:
            del _ENGINE.tts
        except Exception:
            pass
        _ENGINE = None
        _ENGINE_KEY = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    t0 = time.perf_counter()
    engine = TTSInferencer(gpt_path=gpt, sovits_path=sovits)
    print(f"[加载] {time.perf_counter() - t0:.1f}s  version={engine.model_version}  device={engine.device}", flush=True)
    _ENGINE = engine
    _ENGINE_KEY = key
    return engine


def engine_ready_for(gpt: str, sovits: str) -> bool:
    """当前缓存是否正好是这对权重（供 webui 判断"预热是否命中"）。"""
    if _ENGINE is None or _ENGINE_KEY is None:
        return False
    key = (str(Path(gpt).resolve()), str(Path(sovits).resolve()))
    return _ENGINE_KEY == key


# ---------------------------------------------------------------- commands

def cmd_list(cfg: dict, args=None) -> int:
    print("可用音色档案（profiles）:\n")
    for name, p in cfg["profiles"].items():
        print(f"  [{name}] {p.get('desc', '')}")
        print(f"      gpt    : {p['gpt']}")
        print(f"      sovits : {p['sovits']}")
        print(f"      ref    : {p.get('ref_audio')}")
        print()
    print("支持语言（--lang）:")
    print("  " + " / ".join(cfg["languages"]))
    return 0


def resolve_profile(cfg: dict, args) -> dict:
    profiles = cfg["profiles"]
    if args.profile not in profiles:
        sys.exit(f"[FATAL] 未知 profile: {args.profile}\n可选: {', '.join(profiles)}")
    p = dict(profiles[args.profile])
    p.setdefault("ref_lang", "日文")
    if args.gpt:
        p["gpt"] = args.gpt
    if args.sovits:
        p["sovits"] = args.sovits
    if args.ref_audio:
        p["ref_audio"] = args.ref_audio
    if args.ref_text:
        p["ref_text"] = args.ref_text
    if args.ref_lang:
        p["ref_lang"] = args.ref_lang
    # 配置里的路径相对项目根；换成绝对路径供引擎使用
    for k in ("gpt", "sovits", "ref_audio", "ref_text"):
        if p.get(k):
            p[k] = str(runtime_paths.resolve(p[k]))
    for k in ("gpt", "sovits"):
        if not Path(p[k]).is_file():
            sys.exit(f"[FATAL] {k} 权重不存在: {p[k]}")
    if not Path(p["ref_audio"]).is_file():
        sys.exit(f"[FATAL] 参考音频不存在: {p['ref_audio']}")

    # 情绪参考：解析成绝对路径，作为上游 aux 参考参与韵律融合
    p["aux_ref_audio_paths"] = []
    names = getattr(args, "emotion", None) or []
    if names:
        import emotions as emo
        got = emo.resolve(args.profile, names)
        found = {g["name"] for g in got}
        missing = [n for n in names if n not in found]
        if missing:
            print(f"[情绪] 未登记或文件缺失，已忽略: {', '.join(missing)}\n"
                  f"       可用: {', '.join(emo.load(args.profile).keys()) or '（无，先用 tools/emotions.py 登记）'}",
                  flush=True)
        p["aux_ref_audio_paths"] = [g["audio"] for g in got]
        if got:
            print(f"[情绪] 融合参考: {', '.join(g['name'] for g in got)}", flush=True)
    return p


def synth(cfg: dict, prof: dict, text: str, lang: str, out_path: Path, args) -> Path:
    """合成一句。过语义坍缩守卫，失败换 seed 重试（默认 2 次尝试）。

    跨语言参考（如日文参考→中文目标）会让 v3 模型进入 token 坍缩、产出循环乱音。
    守卫在音频层判定，乱音一律不落盘。--no-guard 可关掉（仅用于取证）。
    """
    import numpy as np
    import soundfile as sf

    if lang not in cfg["languages"]:
        sys.exit(f"[FATAL] 不支持的语言: {lang}\n可选: {' / '.join(cfg['languages'])}")

    prompt_text = read_text_arg(prof.get("ref_text"))
    if not prompt_text:
        sys.exit("[FATAL] 缺少参考文本（ref_text）。v3 模型不支持 ref_free，必须提供。")

    d = cfg.get("defaults", {})
    # 版本选择。
    #
    # 关键约束（踩过的坑）：微调音色的 .pth 自带版本标记，且**绑定了它训练时的
    # 基座**。本项目的音色全部是 v3 训练（魔数 b'03'），如果把 base 换成
    # s2Gv4 而不重训 LoRA，就是"v4 架构 + v3 微调权重"的错配，输出会有明显的
    # 电音/失真。所以这里不允许跨版本替换微调音色的基座，只做明确提示。
    #
    # 零样本（纯预训练对）没有 LoRA 绑定，才能真正在 v3/v4 之间切换做对比。
    want_ver = getattr(args, "gsv_version", None)
    sovits = prof["sovits"]
    own_ver = _sovits_version(sovits)

    if want_ver and want_ver != own_ver:
        if own_ver != "v3" or not _is_pretrained_profile(prof):
            sys.exit(
                f"[FATAL] 该音色的权重是 {own_ver} 版本（{Path(sovits).name}），"
                f"不能直接当 {want_ver} 用。\n"
                f"        微调音色的 LoRA 与训练时的基座架构绑定，跨版本会失真。\n"
                f"        想看 {want_ver} 效果：用 zeroshot 音色（纯预训练）再选 {want_ver}，\n"
                f"        或按新版本重新训练该音色。"
            )
        # 纯预训练对：允许换基座做对比
        if want_ver == "v4":
            p = runtime_paths.resolve("models/pretrained/gsv-v4-pretrained/s2Gv4.pth")
            voc = LAB / "_src" / "GPT-SoVITS-main" / "GPT_SoVITS" / "pretrained_models" / \
                  "gsv-v4-pretrained" / "vocoder.pth"
            if not (p and Path(p).is_file()):
                sys.exit("[FATAL] 缺 v4 基座权重。运行 tools/fetch_models.py 下载 "
                         "gsv-v4-pretrained/s2Gv4.pth")
            if not voc.is_file():
                sys.exit("[FATAL] 缺 v4 声码器 vocoder.pth。运行 tools/fetch_models.py")
            sovits = str(p)
            print(f"[版本] {want_ver} 基座 {Path(p).name}", flush=True)

    if want_ver:
        # 零样本的"权重版本"是它引用的预训练基座，不是音色自身的微调版本
        basis = "零样本预训练基座" if _is_pretrained_profile(prof) else "该音色微调版本"
        print(f"[版本] 使用 {want_ver}（{basis} {own_ver}）", flush=True)

    engine = get_engine(cfg, prof["gpt"], sovits)

    attempts = 1 if args.no_guard else max(1, args.guard_attempts)
    rng = random.Random(args.seed if args.seed is not None else None)
    last_reason = None

    for i in range(attempts):
        seed = args.seed if (args.seed is not None and args.seed >= 0) else rng.randint(0, 2**31 - 1)
        t0 = time.perf_counter()
        sr, audio = engine.infer(
            text=text,
            ref_audio_path=prof["ref_audio"],
            prompt_text=prompt_text,
            text_language=lang,
            prompt_language=prof["ref_lang"],
            how_to_cut=args.how_to_cut or d.get("how_to_cut", "不切"),
            top_k=args.top_k if args.top_k is not None else d.get("top_k", 20),
            top_p=args.top_p if args.top_p is not None else d.get("top_p", 0.6),
            temperature=args.temperature if args.temperature is not None else d.get("temperature", 0.6),
            speed=args.speed if args.speed is not None else d.get("speed", 1.0),
            sample_steps=args.sample_steps if args.sample_steps is not None else d.get("sample_steps", 16),
            pause_second=d.get("pause_second", 0.3),
            if_sr=bool(args.if_sr),
            seed=seed,
            aux_ref_audio_paths=prof.get("aux_ref_audio_paths") or [],
        )
        dt = time.perf_counter() - t0
        audio = np.asarray(audio, dtype=np.float32)
        dur = len(audio) / sr

        if args.no_guard:
            break
        reasons = semantic_guard.analyze(audio, sr, nchars=len(text))
        if not reasons:
            break
        last_reason = ",".join(reasons)
        print(f"[守卫] 第 {i + 1}/{attempts} 次尝试被拒 ({last_reason}) "
              f"音频 {dur:.2f}s 文本 {len(text)}字  seed={seed}", flush=True)
    else:
        raise semantic_guard.SemanticGenerationError(
            f"生成坍缩：{attempts} 次尝试均被守卫拒绝（{last_reason}）。"
            "\n最常见原因是参考音频语言与目标语言不一致 —— 请改用目标语言的同种语言参考音频。"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, sr)
    print(f"[合成] {dt:5.2f}s  音频 {dur:5.2f}s  RTF={dt / dur:4.2f}  -> {out_path}", flush=True)
    return out_path


def out_path_for(value: str | None, default_name: str = "speak.wav") -> Path:
    """-o / -d 解析：绝对路径原样；相对路径按「相对项目根」理解，
    已带 out/ 前缀的不再二次拼接（避免 out/out/...）。"""
    if not value:
        return OUT_DIR / default_name
    p = Path(value)
    if p.is_absolute():
        return p
    parts = p.parts
    if parts and parts[0].lower() == "out":
        return LAB / p
    return OUT_DIR / p


def cmd_speak(cfg: dict, args) -> int:
    prof = resolve_profile(cfg, args)
    text = read_text_arg(args.text)
    if not text:
        sys.exit("[FATAL] 缺少 --text")
    out = out_path_for(args.out)
    synth(cfg, prof, text, args.lang, out, args)
    return 0


def cmd_batch(cfg: dict, args) -> int:
    prof = resolve_profile(cfg, args)
    src = Path(args.input)
    if not src.is_file():
        sys.exit(f"[FATAL] 找不到输入文件: {src}")
    out_dir = out_path_for(args.out_dir, default_name="batch")
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[批处理] {len(lines)} 条 -> {out_dir}", flush=True)
    for i, item in enumerate(lines, 1):
        text = item["text"]
        lang = item.get("lang", args.lang)
        name = item.get("out") or f"{i:04d}.wav"
        if not name.lower().endswith(".wav"):
            name += ".wav"
        print(f"  ({i}/{len(lines)}) {text[:32]}", flush=True)
        synth(cfg, prof, text, lang, out_dir / name, args)
    print(f"[完成] 输出目录: {out_dir}")
    return 0


# ---------------------------------------------------------------- main

def main() -> int:
    cfg = load_config()
    defaults = cfg.get("defaults", {})

    ap = argparse.ArgumentParser(
        prog="voicelab",
        description="VoiceLab 独立音色实验室 —— 完全自包含，可整体移动，输出落在 <项目根>/out",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    lp = sub.add_parser("list", help="列出所有音色档案与支持语言")
    lp.set_defaults(func=cmd_list)

    def add_common(p):
        p.add_argument("-p", "--profile", default="zeroshot", help="音色档案名（默认 zeroshot）")
        p.add_argument("-l", "--lang", default="中文", help="目标语言（默认 中文）")
        p.add_argument("--gpt", help="覆盖 GPT 权重路径")
        p.add_argument("--sovits", help="覆盖 SoVITS 权重路径")
        p.add_argument("-r", "--ref-audio", help="覆盖参考音频路径")
        p.add_argument("-R", "--ref-text", help="覆盖参考文本（文本或文件路径）")
        p.add_argument("--ref-lang", help="覆盖参考文本语言")
        p.add_argument("--speed", type=float, help=f"语速（默认 {defaults.get('speed', 1.0)}）")
        p.add_argument("--sample-steps", type=int, help=f"v3 采样步数（默认 {defaults.get('sample_steps', 16)}）")
        p.add_argument("--top-k", type=int, help="GPT top_k")
        p.add_argument("--top-p", type=float, help="GPT top_p")
        p.add_argument("--temperature", type=float, help="GPT temperature")
        p.add_argument("--how-to-cut", help="切句方式")
        p.add_argument("--if-sr", action="store_true", help="启用超分辨率（仅 v3）")
        p.add_argument("--seed", type=int, help="随机种子（默认每次随机；坍缩重试时会另取种子）")
        p.add_argument("--emotion", action="append", default=None, metavar="名称",
                       help="情绪参考名（可重复，如 --emotion 开心 --emotion 激动）。"
                            "用 tools/emotions.py 登记；作为上游 aux 参考参与韵律融合。"
                            "v3 没有 emotion 参数，情绪完全由参考音频承载。")
        p.add_argument("--version", dest="gsv_version", choices=["v3", "v4"],
                       help="GPT-SoVITS 版本。v3=默认；v4 换用 s2Gv4 + 独立声码器"
                            "（s1 与 v3 共用）。不指定则按 s2 权重文件名/魔数自动判断。")
        p.add_argument("--guard-attempts", type=int, default=2, help="语义守卫重试次数（默认 2）")
        p.add_argument("--no-guard", action="store_true", help="关闭语义坍缩守卫（仅排障用）")

    sp = sub.add_parser("speak", help="合成单句")
    add_common(sp)
    sp.add_argument("-t", "--text", required=True, help="要合成的文本（文本或文件路径）")
    sp.add_argument("-o", "--out", help="输出 wav 路径（相对项目根或 out/，或绝对路径）")
    sp.set_defaults(func=cmd_speak)

    bp = sub.add_parser("batch", help="批量合成（JSONL：每行 {\"text\":..,\"lang\":..,\"out\":..}）")
    add_common(bp)
    bp.add_argument("-i", "--input", required=True, help="JSONL 输入文件")
    bp.add_argument("-d", "--out-dir", help="输出目录")
    bp.set_defaults(func=cmd_batch)

    args = ap.parse_args()
    return args.func(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
