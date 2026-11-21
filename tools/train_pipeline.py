#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
方案3：v3 LoRA 微调流水线（照抄上游 webui.py 的配置生成逻辑）

五个阶段：
  1  get-text            文本清洗 + BERT 特征
  2  get-hubert-wav32k   HuBERT 语义特征
  3  get-semantic        语义 token（v3 必须）
  4  s1_train            GPT(s1) 微调，起点 = s1v3.ckpt
  5  s2_train_v3_lora    SoVITS(s2) v3 LoRA，起点 = s2Gv3.pth

产出的 LoRA (.pth < 100MB) 就是上游 `if_lora_v3` 检测逻辑认识的那种格式，
`voicelab.py` 可直接加载。

用法（在项目根目录，用 .venv-tts 解释器）：
  # 只做数据准备（1~3）
  python tools/train_pipeline.py prepare -s 小明 -l data/raw/小明/小明.list

  # 完整训练
  python tools/train_pipeline.py train -s 小明 -l .../小明.list --epochs-s1 15 --epochs-s2 8

  # 只训练（数据已备好）
  python tools/train_pipeline.py train-only -s 小明
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "tools"))
import runtime_paths  # noqa: E402
import config_io  # noqa: E402

CONFIG = config_io.load()
REPO = LAB / "_src" / "GPT-SoVITS-main"
GSV = REPO / "GPT_SoVITS"
# 训练与推理共用同一个项目内虚拟环境（config.json 里的路径相对项目根）
PYTHON = str(runtime_paths.resolve(CONFIG.get("python"))) or runtime_paths.project_python()

PRETRAINED_S1 = LAB / "models" / "pretrained" / "s1v3.ckpt"
PRETRAINED_S2G = LAB / "models" / "pretrained" / "s2Gv3.pth"
PRE_ROOT = REPO / "GPT_SoVITS" / "pretrained_models"
BERT = PRE_ROOT / "chinese-roberta-wwm-ext-large"
CNHUBERT = PRE_ROOT / "chinese-hubert-base"
S2CONFIG = GSV / "configs" / "s2.json"

# LoRA rank=32：与本项目既有音色保持一致，产物体积小（<100MB）
LORA_RANK = 32


def die(msg: str) -> None:
    sys.exit(f"[FATAL] {msg}")


def run(cmd: list[str], env_extra: dict, stage: str) -> None:
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_extra.items()})
    # 让 prepare_datasets/*.py 能 import 到 text / tools 包；
    # train/libs 放的是训练期额外依赖（tensorboard 等），借道 PYTHONPATH 引入，
    # 不污染 .venv-tts。
    env["PYTHONPATH"] = os.pathsep.join(
        [str(GSV), str(REPO), str(LAB / "train" / "libs"), env.get("PYTHONPATH", "")]
    )
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    runtime_paths.ensure_env()
    for k in ("NLTK_DATA", "HF_HOME", "XDG_CACHE_HOME"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    # torch>=2.6 把 torch.load 的 weights_only 默认改成 True，
    # 而 GPT-SoVITS 的续训 checkpoint 里存了 pathlib.WindowsPath，
    # 会抛 UnpicklingError: Unsupported global。用官方开关恢复旧行为。
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    print(f"\n===== {stage} =====", flush=True)
    print("  " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(REPO), env=env)
    if r.returncode != 0:
        die(f"{stage} 失败（退出码 {r.returncode}）")


def check_prereq() -> None:
    if not REPO.is_dir():
        die(f"找不到上游训练仓库: {REPO}")
    for p, name in ((PRETRAINED_S1, "s1v3.ckpt"), (PRETRAINED_S2G, "s2Gv3.pth"),
                    (BERT, "chinese-roberta-wwm-ext-large"), (CNHUBERT, "chinese-hubert-base")):
        if not p.exists():
            die(f"缺少 {name}: {p}")


# ---------------------------------------------------------------- 数据准备

def stage_prepare(speaker: str, list_path: Path) -> Path:
    if not list_path.is_file():
        die(f"找不到 .list: {list_path}")
    n = sum(1 for l in list_path.read_text(encoding="utf-8").splitlines() if l.strip())
    exp_dir = LAB / "data" / speaker
    exp_dir.mkdir(parents=True, exist_ok=True)

    base_env = {
        "inp_text": str(list_path.resolve()),
        "inp_wav_dir": str(list_path.resolve().parent),
        "exp_name": speaker,
        "opt_dir": str(exp_dir.resolve()),
        "i_part": "0",
        "all_parts": "1",
        # 1-get-text.py 自己读这个（BERT 特征提取）
        "bert_pretrained_dir": str(BERT),
        # text/chinese2.py:36 给 g2pw 的 ONNX 加载器读这个：不设成绝对路径，
        # transformers 会把 "GPT_SoVITS/pretrained_models/..." 当成 HF 仓库名并报
        # HFValidationError（和推理侧踩过的是同一个坑）
        "bert_path": str(BERT),
        "cnhubert_base_dir": str(CNHUBERT),
        "pretrained_s2G": str(PRETRAINED_S2G),
        "s2config_path": str(S2CONFIG),
        "is_half": "True",
        "version": "v3",
        "_CUDA_VISIBLE_DEVICES": "0",
    }
    print(f"[数据准备] {n} 条  ->  {exp_dir}")

    run([PYTHON, "-s", "GPT_SoVITS/prepare_datasets/1-get-text.py"], base_env, "1/3 get-text")
    run([PYTHON, "-s", "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"], base_env, "2/3 get-hubert")
    run([PYTHON, "-s", "GPT_SoVITS/prepare_datasets/3-get-semantic.py"], base_env, "3/3 get-semantic")

    # 合并分片。照抄上游 webui.py:1105-1112（文本）与 1214-1221（语义）。
    # 注意：即使只有 1 个分片，脚本产出的也是带后缀的 "2-name2text-0.txt"，
    # 而 s1_train 读的是不带后缀的 "2-name2text.txt"，所以这一步不能省。
    def part_sort(pattern: str) -> list[Path]:
        fs = list(exp_dir.glob(pattern))
        return sorted(fs, key=lambda p: int(p.stem.rsplit("-", 1)[-1]) if p.stem.rsplit("-", 1)[-1].isdigit() else 0)

    txt_parts = part_sort("2-name2text-*.txt")
    if not txt_parts:
        die(f"数据准备没产出 2-name2text-*.txt（看上面 1-get-text 的日志）: {exp_dir}")
    lines: list[str] = []
    for f in txt_parts:
        lines += f.read_text(encoding="utf-8").strip("\n").split("\n")
        f.unlink()
    (exp_dir / "2-name2text.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sem_parts = part_sort("6-name2semantic-*.tsv")
    if not sem_parts:
        die(f"数据准备没产出 6-name2semantic-*.tsv（看上面 3-get-semantic 的日志）: {exp_dir}")
    sem_lines: list[str] = ["item_name\tsemantic_audio"]
    for f in sem_parts:
        sem_lines += f.read_text(encoding="utf-8").strip("\n").split("\n")
        f.unlink()
    (exp_dir / "6-name2semantic.tsv").write_text("\n".join(sem_lines) + "\n", encoding="utf-8")

    print(f"[合并] 2-name2text.txt {len(lines)} 行 / 6-name2semantic.tsv {len(sem_lines) - 1} 行")

    for f in ("2-name2text.txt", "6-name2semantic.tsv"):
        if not (exp_dir / f).is_file():
            die(f"数据准备缺产物: {exp_dir / f}")
    print(f"[数据准备] 完成: {exp_dir}")
    return exp_dir


# ---------------------------------------------------------------- 训练

def stage_train_s1(speaker: str, exp_dir: Path, epochs: int, batch: int) -> Path:
    import yaml
    base = GSV / "configs" / "s1longer-v2.yaml"
    data = yaml.safe_load(base.read_text(encoding="utf-8"))
    out_dir = exp_dir / "logs_s1_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    data["pretrained_s1"] = str(PRETRAINED_S1)
    data["train"]["epochs"] = epochs
    data["train"]["batch_size"] = batch
    # 保持每轮存权重：s1_train.py:82 每历元写 last.ckpt，配合这里能断点续训。
    # 历史遗留的历元文件由 collect() 末尾的 prune_epochs() 统一清理，
    # 不用调大这个值——那样会让崩溃后无法续训。
    data["train"]["save_every_n_epoch"] = 1
    data["train"]["if_save_every_weights"] = True
    data["train"]["if_save_latest"] = True
    data["train"]["if_dpo"] = False
    data["train"]["exp_name"] = speaker
    data["train"]["half_weights_save_dir"] = str(LAB / "models" / "voices" / speaker / "s1")
    data["train_semantic_path"] = str(exp_dir / "6-name2semantic.tsv")
    data["train_phoneme_path"] = str(exp_dir / "2-name2text.txt")
    data["output_dir"] = str(out_dir)

    cfg = exp_dir / "tmp_s1.yaml"
    cfg.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True), encoding="utf-8")
    run([PYTHON, "-s", "GPT_SoVITS/s1_train.py", "--config_file", str(cfg)],
        {"_CUDA_VISIBLE_DEVICES": "0", "hz": "25hz"}, f"4/5 s1_train (GPT, {epochs} epochs)")
    return out_dir


def stage_train_s2(speaker: str, exp_dir: Path, epochs: int, batch: int, s1_ckpt: str) -> Path:
    data = json.loads(S2CONFIG.read_text(encoding="utf-8"))
    data["train"]["batch_size"] = batch
    data["train"]["epochs"] = epochs
    data["train"]["text_low_lr_rate"] = 0.4
    data["train"]["pretrained_s2G"] = str(PRETRAINED_S2G)
    data["train"]["pretrained_s2D"] = ""
    data["train"]["if_save_latest"] = True
    data["train"]["if_save_every_weights"] = True
    # 同上：保留每轮断点以支持续训（s2_train_v3_lora.py:351-367 用 if_save_latest
    # 覆盖同一个 G_233333333333.pth，本身不累积），历元 LoRA 由 prune_epochs() 清理。
    data["train"]["save_every_epoch"] = 1
    data["train"]["gpu_numbers"] = "0"
    data["train"]["grad_ckpt"] = False  # batch 调小后 6GB 显存够用，开梯度检查点反而更慢
    data["train"]["lora_rank"] = LORA_RANK
    data["model"]["version"] = "v3"
    # 关键：exp_dir 必须是「数据集目录」本身，不能另开子目录。
    # module/data_utils.py:288-293 会在 exp_dir 下找 2-name2text.txt / 4-cnhubert / 5-wav32k，
    # 上游 webui.py:535 也是 data.exp_dir = s2_ckpt_dir = exp_root/exp_name。
    data["data"]["exp_dir"] = str(exp_dir)
    data["s2_ckpt_dir"] = str(exp_dir)
    data["save_weight_dir"] = str(LAB / "models" / "voices" / speaker / "s2")
    data["name"] = speaker
    data["version"] = "v3"

    cfg = exp_dir / "tmp_s2_v3.json"
    cfg.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    run([PYTHON, "-s", "GPT_SoVITS/s2_train_v3_lora.py", "--config", str(cfg)],
        {"_CUDA_VISIBLE_DEVICES": "0", "hz": "25hz"}, f"5/5 s2_train_v3_lora ({epochs} epochs, rank={LORA_RANK})")
    return exp_dir


LANG_NAME = {"zh": "中文", "ja": "日文", "en": "英文", "yue": "粤语", "ko": "韩文"}


def find_list(speaker: str, list_path: Path | None) -> Path | None:
    """没显式给 --list 时，在 data/raw/*/ 下找 <speaker>.list。"""
    if list_path and Path(list_path).is_file():
        return Path(list_path)
    for p in sorted((LAB / "data" / "raw").glob(f"**/{speaker}.list")):
        return p
    return None


def pick_reference(list_path: Path | None, speaker: str) -> tuple[Path, Path, str]:
    """从该说话人的 .list 挑一条片段当推理参考音。

    选法（对音质影响极大）：
    - 6.5~8s 为最佳区间：够长有韵律感，又不会因为直播尾音被切
    - **选「中间平稳」的而不是「最长」的**：最长片段往往是语速快、
      尾音被切片切断的，参考音带切音会导致合成时吞字、破音
    - 文本以完整标点收尾优先（说明切片没切断尾音）
    """
    fb_audio = LAB / "data" / "refs" / "default.wav"
    fb_text = LAB / "data" / "refs" / "default.txt"
    if not list_path:
        print("  [参考音] 未找到 .list，回退到默认日文参考（普通话说话人请手动改中文）")
        return fb_audio, fb_text, "日文"

    import soundfile as sf

    cands: list[tuple[float, str, str, str]] = []
    for line in Path(list_path).read_text(encoding="utf-8").splitlines():
        parts = line.split("|", 3)
        if len(parts) < 4 or not parts[0].strip():
            continue
        wav, _spk, lang_code, text = (p.strip() for p in parts)
        try:
            info = sf.info(wav)
            dur = info.frames / info.samplerate
        except Exception:
            continue
        if 6.5 <= dur <= 8.0 and len(text) >= 15:
            # 尾音完整优先（文本以标点收尾）
            tidy = 2.0 if text[-1] in "。！？!?，," else 0.0
            # 语速 3~6 字/秒最好，太快尾音容易被切
            rate = len(text) / dur
            rate_score = 1.5 if 3.0 <= rate <= 6.0 else 0.0
            # 末尾能量检测：参考音尾音被切片硬切 → 模型学到「句尾戛然而止」→ 推理吞字。
            # 末 200ms RMS 若仍接近全段均值（没有自然衰减），说明切在有声处，重罚排除。
            tail_ok = True
            try:
                import numpy as np
                data, sr_ = sf.read(wav, dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                n_tail = int(0.2 * sr_)
                tail_rms = float(np.sqrt(np.mean(data[-n_tail:] ** 2))) if len(data) > n_tail else 0.0
                body_rms = float(np.sqrt(np.mean(data ** 2))) or 1e-9
                if tail_rms > body_rms * 0.45:
                    tail_ok = False
            except Exception:
                pass
            cands.append((dur + tidy + rate_score + (1.0 if tail_ok else -10.0), wav, text, lang_code))

    if not cands:
        print("  [参考音] 没有 6.5~8s 的候选，回退默认日文参考")
        return fb_audio, fb_text, "日文"

    cands.sort(key=lambda x: -x[0])
    dur, wav, text, lang_code = cands[0]
    if cands[0][0] <= 0:
        print("  [参考音][警告] 所有候选末尾能量都偏高（可能被硬切），已选相对最优，建议重切片段")
    lang_name = LANG_NAME.get(lang_code, "中文")

    # ref_text 在 voicelab.py 里按「文件路径」解析，所以落一个 txt
    ref_dir = LAB / "data" / "refs"
    ref_dir.mkdir(parents=True, exist_ok=True)
    out_txt = ref_dir / f"{speaker}_ref.txt"
    out_txt.write_text(text, encoding="utf-8")
    print(f"  [参考音] {Path(wav).name}  {dur:.1f}s  {lang_name}  -> {out_txt.name}")
    return Path(wav), out_txt, lang_name


def prune_epochs(speaker: str, keep_gpt: int | None, keep_sovits: int | None) -> int:
    """只保留指定历元（以及最新）的权重，删掉其余历元文件。

    上游训练器每轮都导出一个带历元号的权重且从不清理：
      models/voices/<spk>/s1/<name>-e<N>.ckpt          148 MB
      models/voices/<spk>/s2/<name>_e<N>_s<S>_l32.pth  72 MB
    25 轮 s1 + 15 轮 s2 就是 4.7 GB/音色。训练一结束就清，避免再堆积。

    keep_* 为 None 时只保留 mtime 最新的那个（即本次训练最后一轮）。
    返回释放的字节数。
    """
    import re

    voice_dir = LAB / "models" / "voices" / speaker
    if not voice_dir.is_dir():
        return 0

    freed = 0
    # (子目录, glob, 从文件名抓历元号的正则, 保留到第几轮)
    plan = [
        ("s1", "*.ckpt", re.compile(r"-e(\d+)\.ckpt$", re.I), keep_gpt),
        ("s2", "*.pth", re.compile(r"_e(\d+)_s\d+_l\d+\.pth$", re.I), keep_sovits),
    ]
    for sub, pattern, rx, keep_to in plan:
        d = voice_dir / sub
        if not d.is_dir():
            continue
        files = []
        for f in d.glob(pattern):
            if not f.is_file():
                continue
            # 判别器权重不是可用产物，跳过（也不要误删，留着无害）
            if pattern == "*.pth" and (f.name.startswith("D_") or "s2D" in f.name):
                continue
            m = rx.search(f.name)
            epoch = int(m.group(1)) if m else -1
            files.append((epoch, f.stat().st_mtime, f))
        if not files:
            continue

        # 要保留的：最新 mtime 的那个；keep_to 给定时再加上该历元
        keep = {max(files, key=lambda x: x[1])[2]}
        if keep_to:
            keep |= {f for (e, _t, f) in files if e == keep_to}

        for epoch, _mtime, f in files:
            if f in keep:
                continue
            try:
                freed += f.stat().st_size
                f.unlink()
                print(f"  [清理] 删除旧历元 {sub}/{f.name}")
            except OSError as e:
                print(f"  [清理][跳过] {f.name}: {e}")

    if freed:
        print(f"  [清理] 共释放 {freed / 2**30:.2f} GB")
    return freed


def collect(speaker: str, s1_dir: Path, s2_dir: Path, list_path: Path | None = None) -> None:
    """归拢训练产物到 models/voices/<speaker>/，并生成可直接用的 profile 片段。

    真正可用的权重由 if_save_every_weights 写到：
      models/voices/<speaker>/s1/<name>-e<epoch>.ckpt   （voicelab.py 直接可加载）
      models/voices/<speaker>/s2/<name>_e<epoch>_s<step>_l<rank>.pth（LoRA，判别器要排除）
    所以直接扫 voice_dir，而不是去训练日志目录里翻。
    """
    voice_dir = LAB / "models" / "voices" / speaker
    got: dict[str, Path | None] = {"gpt": None, "sovits": None}

    for pat, key in (("*.ckpt", "gpt"), ("*.pth", "sovits")):
        cands = sorted(voice_dir.rglob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
        for c in cands:
            if key == "sovits" and (c.name.startswith("D_") or "s2D" in c.name):
                continue
            got[key] = c
            break

    print("\n===== 产物 =====")
    for k, v in got.items():
        print(f"  {k:7}: {v if v else '未找到'}")

    if got["gpt"] and got["sovits"]:
        ref_audio, ref_text, ref_lang = pick_reference(list_path, speaker)
        snippet = {
            speaker: {
                "desc": f"方案3 微调音色: {speaker}",
                "gpt": str(got["gpt"].resolve()),
                "sovits": str(got["sovits"].resolve()),
                "ref_audio": str(ref_audio.resolve()),
                "ref_text": str(ref_text.resolve()),
                "ref_lang": ref_lang,
            }
        }
        out = voice_dir / "profile_snippet.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(snippet, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[提示] 把 {out} 里的内容合并进 {LAB / 'config.json'} 的 profiles。")
        print("       参考音已自动取自该说话人的片段，语言随素材（普通话=中文）。")
    else:
        print("\n[警告] 没凑齐 gpt + sovits，先看上面的训练日志。")


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description="GPT-SoVITS v3 LoRA 微调流水线")
    ap.add_argument("stage", choices=["prepare", "train", "train-only"])
    ap.add_argument("-s", "--speaker", required=True, help="说话人名称")
    ap.add_argument("-l", "--list", help=".list 文件路径（prepare/train 需要）")
    ap.add_argument("--epochs-s1", type=int, default=15, help="GPT 训练轮数（默认 15）")
    ap.add_argument("--epochs-s2", type=int, default=8, help="SoVITS LoRA 轮数（默认 8）")
    ap.add_argument("--batch-s1", type=int, default=8)
    ap.add_argument("--batch-s2", type=int, default=4)
    ap.add_argument("--s1-ckpt", help="train-only 时指定已有的 GPT 权重")
    ap.add_argument(
        "--keep-epochs",
        type=int,
        nargs="*",
        metavar="N",
        help="训练后除了最新一轮，额外保留这些历元的权重（默认只留最新）。"
             "例如 --keep-epochs 15 25；不传则只保留最后一轮。",
    )
    args = ap.parse_args()

    check_prereq()
    exp_dir = LAB / "data" / args.speaker
    list_path = find_list(args.speaker, Path(args.list) if args.list else None)

    if args.stage in ("prepare", "train"):
        if not list_path:
            die("需要 --list（或在 data/raw/*/ 下存在 <speaker>.list）")
        exp_dir = stage_prepare(args.speaker, list_path)
        if args.stage == "prepare":
            print("\n[完成] 仅数据准备。下一步: python train_pipeline.py train-only -s " + args.speaker)
            return 0

    if not (exp_dir / "2-name2text.txt").is_file():
        die(f"数据未准备好，缺 {exp_dir / '2-name2text.txt'}（先跑 prepare）")

    s1_dir = stage_train_s1(args.speaker, exp_dir, args.epochs_s1, args.batch_s1)
    s2_dir = stage_train_s2(args.speaker, exp_dir, args.epochs_s2, args.batch_s2, args.s1_ckpt or "")
    collect(args.speaker, s1_dir, s2_dir, list_path)

    # 训练完立刻清掉中间历元，避免像以前那样每音色堆 4.7 GB。
    # keep_gpt/keep_sovits 指向本次训练最后一轮；mtime 最新的那个也一定保住。
    print("", flush=True)
    print("===== 清理中间历元 =====", flush=True)
    prune_epochs(args.speaker, keep_gpt=args.epochs_s1, keep_sovits=args.epochs_s2)
    for e in sorted(set(args.keep_epochs or [])):
        if e not in (args.epochs_s1, args.epochs_s2):
            print(f"  [提示] --keep-epochs {e}: 不在本次训练轮次内，未额外保留")
    print("\n[完成] 微调结束。用 voicelab.py speak -p <speaker> 试听。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
