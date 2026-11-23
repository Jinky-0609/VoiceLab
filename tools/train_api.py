#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VoiceLab 训练面板（FastAPI 路由 + 后台任务 + 实时日志）。

挂到 tools/webui.py 上使用：from train_api import router; app.include_router(router)

与合成不同，训练是分钟级长任务，所以：
  * 一律作为后台进程跑，输出实时进环形缓冲，前端按 offset 增量拉日志；
  * 每步产物先验后报（例如切片数、标注条数），不靠"命令没报错"判断成功；
  * 任何一步失败都能单独重跑，已完成步骤的产物会被识别并复用。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "tools"))

import runtime_paths  # noqa: E402

router = APIRouter(prefix="/api/train")

PY_TTS = runtime_paths.project_python(".venv-tts")
PY_ASR = runtime_paths.project_python(".venv-asr")
PY_UVR = runtime_paths.project_python(".venv-uvr")

RAW = LAB / "data" / "raw"
JOBS: dict[str, dict] = {}
_lock = threading.Lock()


# ------------------------------------------------------------------ 任务底座

class Job:
    def __init__(self, jid: str, kind: str, speaker: str):
        self.id = jid
        self.kind = kind
        self.speaker = speaker
        self.status = "queued"      # queued|running|done|error|cancelled
        self.started = time.time()
        self.ended: float | None = None
        self.lines: list[str] = []
        self.step = ""
        self.proc: subprocess.Popen | None = None
        self.error = ""

    def log(self, s: str) -> None:
        for ln in str(s).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            if ln.strip():
                self.lines.append(ln.rstrip())
        if len(self.lines) > 4000:
            self.lines = self.lines[-3000:]

    def as_dict(self, offset: int = 0) -> dict:
        return {
            "id": self.id, "kind": self.kind, "speaker": self.speaker,
            "status": self.status, "step": self.step, "error": self.error,
            "started": self.started, "ended": self.ended,
            "elapsed": round((self.ended or time.time()) - self.started, 1),
            "total": len(self.lines),
            "lines": self.lines[offset:],
            "can_cancel": self.status in ("queued", "running"),
        }


def _run_cmd(job: Job, cmd: list[str], step: str) -> int:
    """跑一个子进程，逐行把输出送进日志；返回退出码。"""
    job.step = step
    job.log(f"\n$ {' '.join(str(c) for c in cmd)}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    try:
        job.proc = subprocess.Popen(
            cmd, cwd=str(LAB), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
    except Exception as e:
        job.log(f"[启动失败] {type(e).__name__}: {e}")
        return -1
    assert job.proc.stdout is not None
    for line in job.proc.stdout:
        job.log(line)
    code = job.proc.wait()
    job.proc = None
    job.log(f"[退出] code={code}")
    return code


# ------------------------------------------------------------------ 输入侧

ALLOWED = {".wav", ".mp3", ".flac", ".m4a", ".mp4", ".aac", ".ogg", ".opus", ".wma"}
MAX_UPLOAD = 500 * 2**20


def _safe_speaker(name: str) -> str:
    """说话人名会直接变成目录名和 profile 名，必须收紧。"""
    n = (name or "").strip()
    if not n:
        raise HTTPException(400, "说话人名称不能为空")
    if not re.fullmatch(r"[\w\u4e00-\u9fff\-]{1,32}", n):
        raise HTTPException(400, "名称只能用中英文/数字/下划线/连字符，最长 32 字符")
    if n.startswith("."):
        raise HTTPException(400, "名称不能以点开头")
    return n


@router.post("/upload")
async def upload(
    speaker: str = Form(...),
    file: UploadFile = File(...),
) -> dict:
    sp = _safe_speaker(speaker)
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, f"不支持的格式 {suffix}，可用：{', '.join(sorted(ALLOWED))}")

    dest_dir = RAW / sp
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"source{suffix}"

    size = 0
    with open(dest, "wb") as f:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "文件超过 500MB，请先在本地压缩或截短")
            f.write(chunk)
    if size < 1024:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "文件太小，可能上传失败")
    return {"path": str(dest.relative_to(LAB)), "size": size, "speaker": sp}


@router.get("/speakers")
def speakers() -> dict:
    """列出可训练/已训练的对象及其当前进度，供前端展示。"""
    out = []
    if RAW.is_dir():
        for d in sorted(RAW.iterdir()):
            if not d.is_dir():
                continue
            src = next((p for p in d.iterdir()
                        if p.suffix.lower() in ALLOWED and p.stem.startswith("source")), None)
            seg = d / "segments"
            n_seg = len(list(seg.glob("*.wav"))) if seg.is_dir() else 0
            lst = d / f"{d.name}.list"
            n_list = 0
            if lst.is_file():
                n_list = sum(1 for ln in lst.read_text(encoding="utf-8", errors="replace").splitlines()
                             if ln.strip())
            exp = LAB / "data" / d.name
            prepared = (exp / "2-name2text.txt").is_file()
            voice = LAB / "models" / "voices" / d.name
            s1 = sorted(voice.glob("s1/*.ckpt")) if (voice / "s1").is_dir() else []
            s2 = sorted(voice.glob("s2/*.pth")) if (voice / "s2").is_dir() else []
            out.append({
                "speaker": d.name, "has_source": src is not None,
                "source": src.name if src else "", "segments": n_seg,
                "list_lines": n_list, "prepared": prepared,
                "trained": bool(s1 and s2),
                "s1": s1[-1].name if s1 else "", "s2": s2[-1].name if s2 else "",
            })
    return {"speakers": out}


# ------------------------------------------------------------------ 各阶段

def _stage_clean(job: Job, sp: str, dereverb: bool = True) -> int:
    """去背景音：UVR5 分离人声（可选去混响）。

    这是训练质量的关键一步——带 BGM 的素材会把伴奏学进音色。实测分离度约
    18 dB（每 8 份人声仍混 1 份伴奏），所以分离后仍建议人工抽检片段。
    """
    d = RAW / sp
    srcs = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED
            and not p.stem.endswith("_Vocals")]
    if not srcs:
        job.log("[失败] 目录里没有原始音频")
        return 1
    src = sorted(srcs, key=lambda p: p.stat().st_size, reverse=True)[0]
    work = src

    # 非 wav 先抽音频（libsndfile 不支持 m4a/aac）
    if work.suffix.lower() != ".wav":
        wav = d / "source.wav"
        code = _run_cmd(job, [PY_TTS, str(LAB / "tools" / "video2audio.py"),
                              "extract", "-i", str(work), "-o", str(wav), "--mono"], "抽取音频")
        if code != 0:
            return code
        work = wav

    out_dir = d / "clean"
    out_dir.mkdir(exist_ok=True)
    code = _run_cmd(job, [PY_UVR, str(LAB / "tools" / "uvr5.py"),
                          "separate", "-i", str(work), "-d", str(out_dir)], "分离人声/伴奏")
    if code != 0:
        return code

    vocals = sorted(out_dir.glob("*Vocals*.wav"), key=lambda p: p.stat().st_size, reverse=True)
    if not vocals:
        job.log("[失败] 没有产出人声轨")
        return 2
    job.log(f"[检查] 人声轨: {vocals[0].name}")

    if dereverb:
        code = _run_cmd(job, [PY_UVR, str(LAB / "tools" / "uvr5.py"),
                              "dereverb", "-i", str(vocals[0]), "-d", str(out_dir)], "去混响")
        if code != 0:
            job.log("[警告] 去混响失败，继续用人声轨（混响会被训练学进去）")
        else:
            dry = sorted(out_dir.glob("*No Reverb*.wav"), key=lambda p: p.stat().st_size, reverse=True)
            if dry:
                job.log(f"[检查] 干声: {dry[0].name}")

    # 统一成 source.wav，供后续切分使用
    best = sorted(out_dir.glob("*.wav"), key=lambda p: (
        "No Reverb" not in p.name, -p.stat().st_size))[0]
    final = d / "cleaned.wav"
    shutil.copyfile(best, final)
    job.log(f"[完成] 干净人声 -> {final.relative_to(LAB)}（源自 {best.name}）")
    return 0


def _stage_segments(job: Job, sp: str, max_len: float, min_len: float) -> int:
    """按静音切分。优先用去背景音后的 cleaned.wav，否则用原始音频。"""
    d = RAW / sp
    cleaned = d / "cleaned.wav"
    if cleaned.is_file():
        work = cleaned
        job.log(f"[输入] 使用已去背景音的 {cleaned.name}")
    else:
        srcs = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED
                and not p.stem.endswith(("_Vocals", "_Instrumental")) and "agogue" not in p.name]
        if not srcs:
            job.log("[失败] 目录里没有音频文件")
            return 1
        work = sorted(srcs, key=lambda p: p.stat().st_size, reverse=True)[0]
        if work.suffix.lower() != ".wav":
            job.step = "抽取音频"
            out_wav = d / "source.wav"
            code = _run_cmd(job, [PY_TTS, str(LAB / "tools" / "video2audio.py"),
                                  "extract", "-i", str(work), "-o", str(out_wav), "--mono"], "抽取音频")
            if code != 0:
                return code
            work = out_wav
        job.log("[提示] 未去背景音。带 BGM/混响的素材会把伴奏学进音色，建议先跑「去背景音」")
    seg = d / "segments"
    seg.mkdir(exist_ok=True)
    code = _run_cmd(job, [PY_TTS, str(LAB / "tools" / "split_silence.py"),
                          "-i", str(work), "-o", str(seg),
                          "--min-len", str(min_len), "--max-len", str(max_len)], "按静音切分")
    if code != 0:
        return code
    n = len(list(seg.glob("*.wav")))
    job.log(f"[检查] 切出 {n} 个片段")
    if n == 0:
        job.log("[失败] 没有切出任何片段 —— 可能整段没有静音停顿，或录音太短")
        return 2
    return 0


def _find_list(sp: str) -> Path | None:
    """定位 .list。

    asr_annotate.py 把产物写进 <输出目录>/<speaker>/<speaker>.list，
    而传入的 -i 是 segments/，所以实际落在 data/raw/<sp>/<sp>/<sp>.list。
    不要硬编码单一位置——两个可能都查。
    """
    d = RAW / sp
    cands = [
        d / f"{sp}.list",
        d / sp / f"{sp}.list",
    ]
    for c in cands:
        if c.is_file() and c.stat().st_size > 0:
            return c
    # 兜底：任何深度的同名 list
    found = sorted(d.rglob(f"{sp}.list"), key=lambda p: p.stat().st_size, reverse=True)
    return found[0] if found else None


def _stage_asr(job: Job, sp: str, lang: str) -> int:
    """.list 自动标注。ASR 在 .venv-asr 里，必须用那个解释器。"""
    seg = RAW / sp / "segments"
    if not seg.is_dir() or not list(seg.glob("*.wav")):
        job.log("[失败] 没有片段可标注（先跑切分）")
        return 1
    code = _run_cmd(job, [PY_ASR, str(LAB / "tools" / "asr_annotate.py"),
                          "-i", str(seg), "-s", sp, "-l", lang], "ASR 自动标注")
    if code != 0:
        return code
    lst = _find_list(sp)
    if not lst:
        job.log(f"[失败] 没有产出 .list（在 {seg.parent} 下找过）")
        return 2
    job.log(f"[定位] {lst.relative_to(LAB)}")
    n = sum(1 for ln in lst.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip())
    job.log(f"[检查] .list 共 {n} 条")
    if n < 5:
        job.log(f"[警告] 只有 {n} 条标注，训练效果会很差（建议 ≥20 条 / 10 分钟以上干声）")
    return 0


def _stage_prepare(job: Job, sp: str) -> int:
    lst = _find_list(sp)
    if not lst:
        job.log("[失败] 缺少 .list（先跑标注）")
        return 1
    code = _run_cmd(job, [PY_TTS, str(LAB / "tools" / "train_pipeline.py"),
                          "prepare", "-s", sp, "-l", str(lst)], "数据准备")
    if code != 0:
        return code
    exp = LAB / "data" / sp
    need = ["2-name2text.txt", "6-name2semantic.tsv"]
    missing = [f for f in need if not (exp / f).is_file()]
    if missing:
        job.log(f"[失败] 数据准备不完整，缺：{', '.join(missing)}")
        return 2
    job.log("[检查] 2-name2text.txt / 6-name2semantic.tsv 均已生成")
    return 0


def _stage_train(job: Job, sp: str, epochs_s1: int, epochs_s2: int,
                 batch_s1: int, batch_s2: int) -> int:
    exp = LAB / "data" / sp
    if not (exp / "2-name2text.txt").is_file():
        job.log("[失败] 数据未准备好（先跑数据准备）")
        return 1
    code = _run_cmd(job, [PY_TTS, str(LAB / "tools" / "train_pipeline.py"),
                          "train-only", "-s", sp,
                          "--epochs-s1", str(epochs_s1), "--epochs-s2", str(epochs_s2),
                          "--batch-s1", str(batch_s1), "--batch-s2", str(batch_s2)], "训练")
    if code != 0:
        return code
    voice = LAB / "models" / "voices" / sp
    s1 = list((voice / "s1").glob("*.ckpt")) if (voice / "s1").is_dir() else []
    s2 = list((voice / "s2").glob("*.pth")) if (voice / "s2").is_dir() else []
    if not s1 or not s2:
        job.log(f"[失败] 训练结束但产物不全：s1={len(s1)} s2={len(s2)}")
        return 2
    job.log(f"[检查] 产物 s1={s1[-1].name}  s2={s2[-1].name}")
    return 0


def _register_profile(job: Job, sp: str) -> None:
    """把训练产物写进 config.json，并挑一段参考音频，让用户训完立刻能合成。"""
    import config_io
    cfg = config_io.load(verbose=False)
    voice = LAB / "models" / "voices" / sp
    s1 = sorted((voice / "s1").glob("*.ckpt"))[-1]
    s2 = sorted((voice / "s2").glob("*.pth"))[-1]

    # 参考音频：优先用最长的片段（音色最稳），并找同名 .txt 当参考文本
    seg = RAW / sp / "segments"
    cands = sorted(seg.glob("*.wav"), key=lambda p: p.stat().st_size, reverse=True) if seg.is_dir() else []
    ref_audio = ""
    ref_text = ""
    if cands:
        ref_audio = str(cands[0].relative_to(LAB)).replace("\\", "/")
        lst = _find_list(sp)
        if lst:
            for ln in lst.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = ln.split("\t")
                if len(parts) >= 4 and Path(parts[0]).stem == cands[0].stem:
                    ref_text = parts[3].strip()
                    break
    if not ref_text:
        ref_text = "（请填写参考音频内容，逐字）"
        job.log("[警告] 没能自动匹配参考文本，请在 config.json 里补 ref_text")

    cfg.setdefault("profiles", {})[sp] = {
        "desc": f"方案3 微调音色：{sp}（训练 {time.strftime('%Y-%m-%d')}）",
        "gpt": str(s1.relative_to(LAB)).replace("\\", "/"),
        "sovits": str(s2.relative_to(LAB)).replace("\\", "/"),
        "ref_audio": ref_audio,
        "ref_text": ref_text,
        "ref_lang": "中文",
    }
    config_io.save(cfg)
    job.log(f"[完成] 已写入 config.json 的 profiles：{sp}")


def _workflow(job: Job, steps: list[str], opts: dict) -> None:
    sp = job.speaker
    job.status = "running"
    try:
        for st in steps:
            if job.status == "cancelled":
                return
            job.log(f"\n===== {st} =====")
            if st == "clean":
                # 可选步骤：只有用户显式勾选才在 steps 里出现
                code = _stage_clean(job, sp, opts.get("dereverb", True))
            elif st == "segments":
                code = _stage_segments(job, sp, opts.get("max_len", 12.0), opts.get("min_len", 2.0))
            elif st == "asr":
                code = _stage_asr(job, sp, opts.get("lang", "中文"))
            elif st == "prepare":
                code = _stage_prepare(job, sp)
            elif st == "train":
                code = _stage_train(job, sp, opts.get("epochs_s1", 15), opts.get("epochs_s2", 8),
                                    opts.get("batch_s1", 4), opts.get("batch_s2", 3))
            else:
                job.log(f"[跳过] 未知步骤 {st}")
                continue
            if code != 0:
                job.status = "error"
                job.error = f"步骤「{st}」失败（退出码 {code}）"
                return
        if job.status == "cancelled":
            return
        if "train" in steps:
            _register_profile(job, sp)
        job.status = "done"
        job.step = "完成"
    except Exception as e:
        job.status = "error"
        job.error = f"{type(e).__name__}: {e}"
        job.log(f"[异常] {job.error}")
    finally:
        job.ended = time.time()


@router.post("/start")
def start(body: dict) -> dict:
    sp = _safe_speaker(body.get("speaker", ""))
    # 默认流程不含 clean —— 去背景音是**可选**步骤，只有用户勾选才会出现在 steps 里。
    # 素材本身已是干净干声时没必要跑（UVR5 有损，约 18 dB 分离度）。
    steps = body.get("steps") or ["segments", "asr", "prepare", "train"]
    steps = [s for s in steps if s in ("clean", "segments", "asr", "prepare", "train")]
    if not steps:
        raise HTTPException(400, "没有可执行的步骤")
    if "clean" in steps and "segments" not in steps:
        raise HTTPException(400, "去背景音之后必须接切分（clean 的产物需要切成训练片段）")
    # 允许从任意步骤续跑：对应的前置产物在磁盘上就会被复用。
    # 首次任务由前端传完整 steps；重跑某一步只传那一步即可。

    with _lock:
        running = [j for j in JOBS.values() if j.status == "running"]
        if running:
            raise HTTPException(409, f"已有训练任务在跑（{running[0].speaker}），训练很吃显存，请等它结束")

    jid = f"{int(time.time())}-{sp}"
    job = Job(jid, "train", sp)
    JOBS[jid] = job
    opts = {k: body[k] for k in
            ("max_len", "min_len", "lang", "epochs_s1", "epochs_s2", "batch_s1", "batch_s2")
            if k in body}
    threading.Thread(target=_workflow, args=(job, steps, opts), daemon=True).start()
    return {"job": jid}


@router.get("/job/{jid}")
def job_state(jid: str, offset: int = 0) -> dict:
    job = JOBS.get(jid)
    if not job:
        raise HTTPException(404, "任务不存在（服务重启会清空）")
    return job.as_dict(offset)


@router.post("/cancel/{jid}")
def cancel(jid: str) -> dict:
    job = JOBS.get(jid)
    if not job:
        raise HTTPException(404)
    job.status = "cancelled"
    if job.proc and job.proc.poll() is None:
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(job.proc.pid)],
                           capture_output=True)
            job.log("[已请求停止] 训练子进程被终止，中间产物保留可重跑")
        except Exception as e:
            job.log(f"[停止失败] {e}")
    job.ended = time.time()
    return {"cancelled": True}


# ------------------------------------------------------------------ 音色删除

def _voice_assets(name: str) -> list[dict]:
    """列出某音色的全部干系资产（与 tools/delete_voice.py 同一套规则）。"""
    import delete_voice as dv
    import config_io
    cfg = config_io.load(verbose=False)
    out = []
    for label, path in dv.collect(name, cfg, include_shared=False):
        try:
            rel = path.relative_to(LAB).as_posix()
        except ValueError:
            rel = str(path)
        out.append({"label": label, "path": rel,
                    "size": dv._fmt(dv._size(path)),
                    "is_dir": path.is_dir()})
    return out


@router.get("/voice/{name}/assets")
def voice_assets(name: str) -> dict:
    """删除前预览：列出会删掉什么、总共释放多少。"""
    import config_io
    cfg = config_io.load(verbose=False)
    if name not in cfg.get("profiles", {}):
        raise HTTPException(404, f"没有这个音色: {name}")
    import delete_voice as dv
    assets = _voice_assets(name)
    total_bytes = 0
    for label, path in dv.collect(name, cfg, include_shared=False):
        total_bytes += dv._size(path)
    return {"ok": True, "speaker": name, "assets": assets,
            "total": dv._fmt(total_bytes), "total_bytes": total_bytes}


@router.delete("/voice/{name}")
def voice_delete(name: str) -> dict:
    """删除音色及其全部资产，profile 条目先备份到 _local/deleted/。"""
    import config_io
    import delete_voice as dv
    cfg = config_io.load(verbose=False)
    if name not in cfg.get("profiles", {}):
        raise HTTPException(404, f"没有这个音色: {name}")
    if len(cfg["profiles"]) <= 1:
        raise HTTPException(400, "这是最后一个音色，删掉后就没有可用档案了")

    # 有训练在跑时不允许删，避免删掉正在写的目录
    with _lock:
        running = [j for j in JOBS.values() if j.status == "running"]
        if running:
            raise HTTPException(409, f"有训练任务在跑（{running[0].speaker}），请等它结束再删")

    items = dv.collect(name, cfg, include_shared=False)
    total = sum(dv._size(p) for _, p in items)

    backup = LAB / "_local" / "deleted"
    backup.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (backup / f"{name}.{stamp}.profile.json").write_text(
        json.dumps(cfg["profiles"][name], ensure_ascii=False, indent=2), encoding="utf-8")

    removed, errors = [], []
    for label, path in items:
        if not str(path.resolve()).startswith(str(LAB.resolve())):
            errors.append(f"越界路径 {path}")
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(str(path.relative_to(LAB)))
        except Exception as e:
            errors.append(f"{path.name}: {type(e).__name__}")

    del cfg["profiles"][name]
    config_io.save(cfg)
    return {"deleted": name, "removed": removed, "errors": errors,
            "freed": dv._fmt(total), "remaining": list(cfg["profiles"])}
