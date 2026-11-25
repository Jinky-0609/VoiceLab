#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VoiceLab 图形界面（FastAPI + 内置单页前端）。

    .venv-tts/Scripts/python.exe tools/webui.py                  # http://127.0.0.1:8765
    .venv-tts/Scripts/python.exe tools/webui.py --port 9000 --no-browser

为什么值得有个 Web 界面：CLI 每次合成都要加载模型 25~60 秒，本服务把引擎常驻，
首句之后每次合成只要几秒。

设计约束：
  * 复用 voicelab.py 的 get_engine/synth 与 semantic_guard，不重复实现守卫/重试；
  * 前端零外部依赖（无 CDN），断网可用；
  * 上游 TTS 非线程安全 -> 全局锁串行推理，队列可撤回、进行中不可中断（不假装能取消）。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from typing import Any

import sys

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "tools"))

import config_io          # noqa: E402
import runtime_paths      # noqa: E402
import semantic_guard     # noqa: E402

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse  # noqa: E402

WEB_DIR = LAB / "out" / "web"
WEB_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_PATH = WEB_DIR / "history.json"
HISTORY_MAX = 60

app = FastAPI(title="VoiceLab", version="0.1")

# ------------------------------------------------------------------ 状态

_lock = threading.Lock()          # 串行化推理
_job_lock = threading.Lock()
_state: dict[str, Any] = {
    "engine_loaded": False,
    "engine_info": None,
    "loading": False,
    "busy": False,
    "current": None,               # 正在合成的任务描述
    "queue": [],                   # 排队中的任务 id
    "cancel": set(),               # 被撤回的排队任务
}
_history: list[dict] = []


def _load_history() -> None:
    """载入历史；服务重启后残留的 running/queued 条目必须归一，否则前端永远转圈。"""
    global _history
    if HISTORY_PATH.is_file():
        try:
            raw = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return
        for it in raw:
            if it.get("status") in ("running", "queued"):
                it["status"] = "error"
                it["error"] = "服务已重启，该任务中断（可点重试）"
            it.pop("position", None)
        _history = raw


def _save_history() -> None:
    try:
        HISTORY_PATH.write_text(
            json.dumps(_history[:HISTORY_MAX], ensure_ascii=False, indent=1),
            encoding="utf-8")
    except OSError:
        pass


class _Args:
    """synth() 需要一个 argparse.Namespace 形状的参数袋。"""

    def __init__(self, **kw):
        for k in ("how_to_cut", "top_k", "top_p", "temperature", "speed",
                  "sample_steps", "seed"):
            setattr(self, k, None)
        self.if_sr = False
        self.no_guard = False
        self.guard_attempts = 2
        self.gsv_version = None      # None=自动（按 s2 文件判断）；"v4" 强制用 v4 基座
        self.emotion = None          # 情绪名（已在 data/emotions 里登记的）
        self.prosody = None          # 参数化情绪预设（不需要素材）
        self.prosody_strength = 1.0
        self.__dict__.update(kw)


# ------------------------------------------------------------------ 引擎

def _get_engine(cfg: dict, gpt: str, sovits: str):
    import voicelab
    return voicelab.get_engine(cfg, gpt, sovits)


def _profile_status(cfg: dict, name: str) -> dict:
    """把一个 profile 的可用性算清楚给前端——别等点了合成才报错。"""
    prof = cfg["profiles"].get(name)
    if not prof:
        return {"ok": False, "reason": "未知音色"}
    p = dict(prof)

    # 权重是硬前提：没有 gpt/sovits 这个档案根本没法工作
    missing = []
    for key, label in (("gpt", "GPT 权重"), ("sovits", "SoVITS 权重")):
        v = p.get(key)
        if not v:
            missing.append(f"{label}未配置")
        elif runtime_paths.resolve(v) is None or not Path(runtime_paths.resolve(v)).is_file():
            missing.append(f"{label}缺失")

    # 参考素材：对**零样本**只是"默认值"，不是硬条件——用户完全可以在
    # 合成时用「高级 → 参考音频」临时指定。所以这里不判为不可用，
    # 只标记 needs_ref，前端提示"选好音色后拖入参考音频"即可。
    # （曾经把缺失当成不可用，导致零样本整体变灰不可选，是设计错误。）
    needs_ref = False
    ra = p.get("ref_audio")
    if not ra or runtime_paths.resolve(ra) is None or not Path(runtime_paths.resolve(ra)).is_file():
        needs_ref = True
    rt = p.get("ref_text")
    if rt:
        rp = runtime_paths.resolve(rt)
        is_file = bool(rp and Path(rp).is_file())
        if not is_file and len(str(rt)) > 120:
            needs_ref = True                 # 长路径才当文件看待
    if str(rt or "").strip().startswith("（"):
        needs_ref = True

    is_zeroshot = "pretrained" in str(p.get("gpt", "")).lower()
    if not is_zeroshot and needs_ref:
        # 微调音色必须自带参考音频才能定音色，否则等于没音色
        missing.append("参考音频缺失")

    reason = "、".join(missing)
    return {
        "ok": not missing,
        "needs_ref": needs_ref,
        "is_zeroshot": is_zeroshot,
        "reason": reason,
        "ref_lang": p.get("ref_lang", ""),
        "desc": p.get("desc", ""),
        # 档案自带的参考音频（相对路径），前端用它判断"界面没填时是否还有默认值"
        "_ref_audio": (p.get("ref_audio") or "") if not needs_ref else "",
    }


# ------------------------------------------------------------------ API

@app.get("/api/config")
def api_config() -> dict:
    cfg = config_io.load()
    profiles = []
    for name in cfg["profiles"]:
        st = _profile_status(cfg, name)
        profiles.append({"name": name, **st})
    cur = _state["engine_info"]
    v4_path = LAB / "models" / "pretrained" / "gsv-v4-pretrained" / "s2Gv4.pth"
    voc = LAB / "_src" / "GPT-SoVITS-main" / "GPT_SoVITS" / "pretrained_models" / \
          "gsv-v4-pretrained" / "vocoder.pth"
    return {
        "profiles": profiles,
        "versions": {"v3": True, "v4": v4_path.is_file() and voc.is_file()},
        "languages": cfg.get("languages", []),
        "defaults": cfg.get("defaults", {}),
        "engine": cur,
        "engine_loaded": _state["engine_loaded"],
        "loading": bool(_state["loading"]),   # 前端轮询用：预热/加载是否进行中
        "busy": _state["busy"],
        "queue": list(_state["queue"]),
    }


# ------------------------------------------------------------------ 音色管理

@app.get("/api/voices")
def api_voices() -> dict:
    """音色管理页用：完整信息（含路径、大小、版本）与文件存在性。"""
    import voicelab
    cfg = config_io.load()
    out = []
    for name, p in cfg["profiles"].items():
        if name.startswith("_"):
            continue
        st = _profile_status(cfg, name)
        gpt = runtime_paths.resolve(p.get("gpt") or "")
        sov = runtime_paths.resolve(p.get("sovits") or "")
        size = 0
        for f in (gpt, sov):
            if f and Path(f).is_file():
                size += Path(f).stat().st_size
        out.append({
            "name": name, "desc": p.get("desc", ""),
            "ref_lang": p.get("ref_lang", ""),
            "gpt": p.get("gpt", ""), "sovits": p.get("sovits", ""),
            "ref_audio": p.get("ref_audio", ""), "ref_text": p.get("ref_text", ""),
            "version": voicelab._sovits_version(str(sov)) if sov else "?",
            "size_mb": round(size / 2**20, 1),
            "ok": st["ok"], "needs_ref": st.get("needs_ref", False),
            "is_zeroshot": st.get("is_zeroshot", False),
            "reason": st.get("reason", ""),
        })
    return {"voices": out}


@app.patch("/api/voices/{name}")
def api_voice_update(name: str, body: dict) -> dict:
    """改描述 / 参考音频 / 参考文本 / 参考语言。"""
    cfg = config_io.load()
    if name not in cfg["profiles"]:
        raise HTTPException(404, "没有这个音色")
    p = cfg["profiles"][name]
    allowed = ("desc", "ref_audio", "ref_text", "ref_lang")
    touched = []
    for k in allowed:
        if k in body and body[k] is not None:
            p[k] = str(body[k]).strip()
            touched.append(k)
    if not touched:
        raise HTTPException(400, "没有可更新的字段")
    config_io.save(cfg)
    return {"updated": name, "fields": touched}


@app.post("/api/voices/{name}/duplicate")
def api_voice_duplicate(name: str, body: dict) -> dict:
    """复制一个音色档案（只复制配置，权重共享路径）。"""
    cfg = config_io.load()
    if name not in cfg["profiles"]:
        raise HTTPException(404, "没有这个音色")
    new = (body.get("as") or "").strip()
    if not new:
        raise HTTPException(400, "需要新名称")
    if new in cfg["profiles"]:
        raise HTTPException(400, f"{new} 已存在")
    import emotions
    prof = dict(cfg["profiles"][name])
    prof["desc"] = (prof.get("desc", "") + "（副本）").strip()
    cfg["profiles"][new] = prof
    config_io.save(cfg)
    return {"created": new}


# ------------------------------------------------------------------ 情绪参考

@app.get("/api/emotions/{profile}")
def api_emotions(profile: str) -> dict:
    """列出某音色已登记的情绪参考。"""
    import emotions
    data = emotions.load(profile)
    out = []
    for name, it in data.items():
        ap = LAB / it.get("audio", "")
        out.append({
            "name": name, "audio": it.get("audio", ""),
            "text": it.get("text", ""), "lang": it.get("lang", "中文"),
            "exists": ap.is_file(),
            "duration": _audio_duration(ap),
        })
    return {"profile": profile, "emotions": out}


def _audio_duration(p: Path) -> float | None:
    try:
        import soundfile as sf
        return round(sf.info(str(p)).duration, 2) if p.is_file() else None
    except Exception:
        return None


@app.post("/api/emotions/{profile}")
async def api_emotion_add(profile: str, name: str = Form(...),
                          text: str = Form(""), lang: str = Form("中文"),
                          file: UploadFile = File(...)) -> dict:
    """上传并登记一条情绪参考（音频 + 逐字文本）。"""
    import emotions
    suffix = Path(file.filename or "e.wav").suffix.lower()
    if suffix not in (".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"):
        raise HTTPException(400, f"不支持的格式 {suffix}")
    dest_dir = LAB / "data" / "emotions" / profile
    dest_dir.mkdir(parents=True, exist_ok=True)
    raw = dest_dir / f"_upload{suffix}"
    size = 0
    with open(raw, "wb") as f:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            if size > 60 * 2**20:
                f.close(); raw.unlink(missing_ok=True)
                raise HTTPException(413, "音频超过 60MB")
            f.write(chunk)

    dest = dest_dir / f"{name}.wav"
    if suffix != ".wav":
        try:
            import video2audio
            video2audio.extract_one(raw, dest, sr=24000, mono=True)
            raw.unlink(missing_ok=True)
        except Exception as e:
            raw.unlink(missing_ok=True)
            raise HTTPException(500, f"转码失败：{e}")
    elif raw != dest:
        raw.replace(dest)

    rel = str(dest.relative_to(LAB)).replace("\\", "/")
    try:
        emotions.add(profile, name, rel, text, lang)
    except ValueError as e:
        raise HTTPException(400, str(e))
    d = _audio_duration(dest)
    warn = ""
    if d and d < 2.5:
        warn = f"只有 {d}s，太短会不稳定（建议 3~10s）"
    elif d and d > 20:
        warn = f"有 {d:.0f}s，偏长会稀释情绪特征（建议 3~10s）"
    return {"ok": True, "name": name, "audio": rel, "duration": d, "warning": warn}


@app.delete("/api/emotions/{profile}/{name}")
def api_emotion_del(profile: str, name: str) -> dict:
    import emotions
    if not emotions.remove(profile, name):
        raise HTTPException(404, "没有这个情绪")
    return {"removed": name}


# ------------------------------------------------------------------ 自动情绪

@app.post("/api/auto_emotion")
def api_auto_emotion(body: dict) -> dict:
    """文本按句切分 + 自动判情绪，返回可编辑的分段表。

    实验性功能：规则+词典判定，允许（也鼓励）前端手动纠正每一句的情绪。
    """
    import auto_emotion
    import emotions
    text = (body.get("text") or "").strip()
    profile = body.get("profile") or ""
    if not text:
        raise HTTPException(400, "文本为空")
    max_chars = int(body.get("max_chars") or 60)
    max_chars = max(20, min(max_chars, 300))

    avail = list(emotions.load(profile).keys()) if profile else []
    segs = auto_emotion.plan(text, avail, max_chars)
    return {"segments": segs, "available": avail, "profile": profile}


@app.post("/api/emotion_render")
def api_emotion_render(body: dict) -> dict:
    """按「每句一个情绪」的方案逐段合成，再交叉淡入拼接成一整条。

    body: {profile, segments:[{text, emotion, lang?}], opts?}
    逐段都过语义守卫；任何一段失败会记录并跳过，最后报告成功率。
    """
    import auto_emotion
    import emotions
    import soundfile as sf
    import voicelab

    cfg = config_io.load()
    pid = body.get("profile", "")
    segs = body.get("segments") or []
    if pid not in cfg["profiles"]:
        raise HTTPException(400, "未知音色")
    if not segs:
        raise HTTPException(400, "没有分段")

    st = _profile_status(cfg, pid)
    if not st["ok"]:
        raise HTTPException(400, f"该音色暂不可用：{st['reason']}")

    jid = uuid.uuid4().hex[:8]
    BATCH_JOBS[jid] = {"id": jid, "status": "running", "done": 0,
                       "total": len(segs), "errors": [], "started": time.time(),
                       "zip": None, "kind": "emotion"}

    def worker():
        job = BATCH_JOBS[jid]
        prof_vo = dict(cfg["profiles"][pid])
        for k in ("gpt", "sovits", "ref_audio", "ref_text"):
            if prof_vo.get(k):
                prof_vo[k] = str(runtime_paths.resolve(prof_vo[k]))

        opts = body.get("opts") or {}
        args = _Args(**{k: opts.get(k) for k in
                        ("how_to_cut", "top_k", "top_p", "temperature", "speed",
                         "sample_steps", "seed") if opts.get(k) is not None})
        args.no_guard = bool(opts.get("no_guard", False))
        args.guard_attempts = int(opts.get("guard_attempts", 2))
        args.gsv_version = opts.get("gsv_version") or None
        args.prosody = opts.get("prosody") or None
        args.prosody_strength = float(opts.get("prosody_strength") or 1.0)

        outdir = WEB_DIR / f"emo_{jid}"
        outdir.mkdir(parents=True, exist_ok=True)
        made: list[Path] = []

        with _lock:
            for i, s in enumerate(segs, 1):
                if job["status"] == "cancelled":
                    break
                text = (s.get("text") or "").strip()
                if not text:
                    job["done"] = i
                    continue
                emo = (s.get("emotion") or "").strip()
                au = None
                if emo:
                    got = emotions.resolve(pid, [emo])
                    if got:
                        au = [g["audio"] for g in got]
                if au:
                    prof_vo["aux_ref_audio_paths"] = au
                else:
                    prof_vo.pop("aux_ref_audio_paths", None)
                seg_out = outdir / f"{i:03d}.wav"
                try:
                    voicelab.synth(cfg, prof_vo, text, s.get("lang") or "中文",
                                   seg_out, args)
                    made.append(seg_out)
                except Exception as e:
                    job["errors"].append({"i": i, "text": text[:30],
                                          "error": f"{type(e).__name__}: {str(e)[:120]}"})
                job["done"] = i

        if made:
            final = WEB_DIR / f"emo_{jid}.wav"
            try:
                auto_emotion.concat_with_crossfade(made, final)
                job["zip"] = f"/audio/{final.name}"
                job["segments_made"] = len(made)
            except Exception as e:
                job["errors"].append({"i": 0, "text": "", "error": f"拼接失败: {e}"})

        job["status"] = "cancelled" if job["status"] == "cancelled" else "done"
        job["elapsed"] = round(time.time() - job["started"], 1)

    threading.Thread(target=worker, daemon=True).start()
    return {"job": jid, "total": len(segs)}


# ------------------------------------------------------------------ 长文本分段

@app.post("/api/split")
def api_split(body: dict) -> dict:
    """把长文本切成适合合成的段落，并给出估算时间轴（用于生成 srt）。

    body: {text, max_chars?}
    """
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "文本为空")
    max_chars = int(body.get("max_chars") or 120)
    max_chars = max(20, min(max_chars, 500))

    import re
    # 按句末标点切，再按长度合并/拆分
    parts = re.split(r"(?<=[。！？；!?;\n])", text)
    chunks, buf = [], ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) <= max_chars:
            buf += p
        else:
            if buf:
                chunks.append(buf)
            while len(p) > max_chars:
                chunks.append(p[:max_chars])
                p = p[max_chars:]
            buf = p
    if buf:
        chunks.append(buf)

    # 粗估时长：实测中文约 0.19s/字（0.15~0.30 区间）
    segs, t = [], 0.0
    for i, c in enumerate(chunks, 1):
        dur = round(len(c) * 0.19 + 0.25, 2)
        segs.append({"i": i, "text": c, "chars": len(c),
                     "start": round(t, 2), "end": round(t + dur, 2)})
        t += dur
    return {"segments": segs, "total_chars": len(text),
            "estimated_duration": round(t, 1), "max_chars": max_chars}


BATCH_JOBS: dict[str, dict] = {}


@app.post("/api/batch")
def api_batch(body: dict) -> dict:
    """批量合成：一次提交多行文本，后台逐条跑，最后打包 zip。

    body: {profile, lang, lines: [{text, lang?, out?}], opts?}
    """
    cfg = config_io.load()
    pid = body.get("profile", "")
    lines = body.get("lines") or []
    if pid not in cfg["profiles"]:
        raise HTTPException(400, "未知音色")
    if not lines:
        raise HTTPException(400, "没有待合成的文本")
    if len(lines) > 500:
        raise HTTPException(400, "单次最多 500 条，更多请分批")

    st = _profile_status(cfg, pid)
    if not st["ok"]:
        raise HTTPException(400, f"该音色暂不可用：{st['reason']}")

    default_lang = body.get("lang") or "中文"
    opts = body.get("opts") or {}
    jid = uuid.uuid4().hex[:8]
    BATCH_JOBS[jid] = {
        "id": jid, "status": "running", "done": 0, "total": len(lines),
        "errors": [], "started": time.time(), "zip": None,
    }

    def worker():
        import voicelab
        import zipfile
        job = BATCH_JOBS[jid]
        prof_vo = dict(cfg["profiles"][pid])
        for k in ("gpt", "sovits", "ref_audio", "ref_text"):
            if prof_vo.get(k):
                prof_vo[k] = str(runtime_paths.resolve(prof_vo[k]))

        args = _Args(**{k: opts.get(k) for k in
                        ("how_to_cut", "top_k", "top_p", "temperature", "speed",
                         "sample_steps", "seed") if opts.get(k) is not None})
        args.no_guard = bool(opts.get("no_guard", False))
        args.guard_attempts = int(opts.get("guard_attempts", 2))
        args.gsv_version = opts.get("gsv_version") or None
        args.prosody = opts.get("prosody") or None
        args.prosody_strength = float(opts.get("prosody_strength") or 1.0)

        outdir = WEB_DIR / f"batch_{jid}"
        outdir.mkdir(parents=True, exist_ok=True)
        made = []
        with _lock:
            for i, item in enumerate(lines, 1):
                if job["status"] == "cancelled":
                    break
                text = (item.get("text") or "").strip()
                if not text:
                    job["errors"].append({"i": i, "text": "", "error": "空文本"})
                    job["done"] = i
                    continue
                lang = item.get("lang") or default_lang
                name = item.get("out") or f"{i:04d}.wav"
                if not name.lower().endswith(".wav"):
                    name += ".wav"
                try:
                    voicelab.synth(cfg, prof_vo, text, lang, outdir / name, args)
                    made.append(name)
                except SystemExit as e:
                    job["errors"].append({"i": i, "text": text[:40], "error": str(e)[:200]})
                except semantic_guard.SemanticGenerationError as e:
                    job["errors"].append({"i": i, "text": text[:40],
                                          "error": "语义坍缩被拦下"})
                except Exception as e:
                    job["errors"].append({"i": i, "text": text[:40],
                                          "error": f"{type(e).__name__}: {str(e)[:160]}"})
                job["done"] = i

        # 打包
        if made:
            zpath = WEB_DIR / f"batch_{jid}.zip"
            try:
                with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
                    for n in made:
                        zf.write(outdir / n, arcname=n)
                job["zip"] = f"/batchzip/{jid}"
            except Exception as e:
                job["errors"].append({"i": 0, "text": "", "error": f"打包失败: {e}"})

        job["status"] = "cancelled" if job["status"] == "cancelled" else "done"
        job["elapsed"] = round(time.time() - job["started"], 1)

    threading.Thread(target=worker, daemon=True).start()
    return {"job": jid, "total": len(lines)}


@app.get("/api/batch/{jid}")
def api_batch_status(jid: str) -> dict:
    j = BATCH_JOBS.get(jid)
    if not j:
        raise HTTPException(404, "任务不存在")
    return {k: v for k, v in j.items() if k != "lines"}


@app.post("/api/batch/{jid}/cancel")
def api_batch_cancel(jid: str) -> dict:
    j = BATCH_JOBS.get(jid)
    if not j:
        raise HTTPException(404)
    j["status"] = "cancelled"
    return {"cancelled": True}


@app.get("/batchzip/{jid}")
def batch_zip(jid: str):
    p = (WEB_DIR / f"batch_{jid}.zip").resolve()
    if not str(p).startswith(str(WEB_DIR.resolve())) or not p.is_file():
        raise HTTPException(404, "打包文件不存在")
    return FileResponse(str(p), media_type="application/zip",
                        filename=f"voicelab_batch_{jid}.zip")


@app.get("/api/resource")
def api_resource() -> dict:
    """实时资源占用。

    合成慢常常不是 bug 而是资源竞争（实测：显存被占 4.6GB 时
    RTF 从 6 涨到 39，用户会误以为程序坏了）。这里直接把状况暴露给界面。
    """
    out = {"ram": None, "vram": None, "warning": ""}
    try:
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_uint32),
                        ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                        ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                        ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                        ("ullAvailExtendedVirtual", ctypes.c_uint64)]

        m = _MS()
        m.dwLength = ctypes.sizeof(m)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            g = 2 ** 30
            out["ram"] = {
                "total_gb": round(m.ullTotalPhys / g, 1),
                "avail_gb": round(m.ullAvailPhys / g, 1),
                "percent": int(m.dwMemoryLoad),
                "commit_avail_gb": round(m.ullAvailPageFile / g, 1),
            }
    except Exception:
        pass

    try:
        import shutil as _sh
        info = _sh.which("nvidia-smi")
        if info:
            r = subprocess.run(
                [info, "--query-gpu=memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=4)
            if r.returncode == 0 and r.stdout.strip():
                parts = [p.strip() for p in r.stdout.strip().splitlines()[0].split(",")]
                used, total = int(parts[0]), int(parts[1])
                out["vram"] = {
                    "used_mb": used, "total_mb": total,
                    "percent": int(used * 100 / total) if total else 0,
                    "gpu_util": int(parts[2]) if len(parts) > 2 else None,
                }
    except Exception:
        pass

    # 给出人话提示
    warns = []
    ram = out.get("ram")
    if ram:
        # 阈值按实测调：提交内存耗尽时 G2PW onnx 会 "bad allocation"；
        # 物理内存低于 4GB 时 RTF 会明显上升（8GB 卡实测从 6 涨到 39）。
        if ram["commit_avail_gb"] < 3:
            warns.append(f"提交内存只剩 {ram['commit_avail_gb']}GB，"
                         f"合成可能失败（bad allocation），建议关掉一些程序")
        elif ram["avail_gb"] < 4:
            warns.append(f"物理内存只剩 {ram['avail_gb']}GB，合成会明显变慢")
    vram = out.get("vram")
    if vram and vram["percent"] > 80:
        warns.append(f"显存已用 {vram['percent']}%（{vram['used_mb']}MB），"
                     f"合成会明显变慢，关掉其他用 GPU 的程序可恢复")
    out["warning"] = "；".join(warns)
    return out


@app.get("/api/history")
def api_history() -> JSONResponse:
    return JSONResponse(_history[:HISTORY_MAX])


@app.post("/api/preload")
def api_preload(profile: str) -> dict:
    """提前把模型加载好——第一次最痛，之后每次秒级。

    注意执行顺序：必须在**返回前**把 _state["loading"] 置 True，
    否则前端立刻轮询时会看到 loading=False 并误判「已经就绪」，
    造成「点了预热却毫无作用」的观感（真实 bug，已修）。
    """
    cfg = config_io.load()
    if profile not in cfg["profiles"]:
        raise HTTPException(404, "未知音色")
    if _state["loading"]:
        return {"started": False, "msg": "正在加载中"}
    p = cfg["profiles"][profile]
    for k in ("gpt", "sovits"):
        f = runtime_paths.resolve(p.get(k))
        if not f or not Path(f).is_file():
            raise HTTPException(400, f"{k} 权重不存在")

    _state["loading"] = True      # 先置位，再起线程（见上面注释）

    def worker():
        try:
            with _lock:
                eng = _get_engine(cfg, str(runtime_paths.resolve(p["gpt"])),
                                  str(runtime_paths.resolve(p["sovits"])))
                _state["engine_info"] = {
                    "profile": profile,
                    "version": eng.model_version,
                    "device": eng.device,
                    "load_secs": round(eng._load_secs, 1),
                }
                _state["engine_loaded"] = True
        except Exception as e:
            _state["engine_info"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            _state["engine_loaded"] = False
        finally:
            _state["loading"] = False

    threading.Thread(target=worker, daemon=True).start()
    return {"started": True}


def _job_payload(pid: str, text: str, lang: str, opts: dict) -> dict:
    return {"profile": pid, "text": text, "lang": lang, "opts": opts}


@app.post("/api/speak")
def api_speak(body: dict) -> dict:
    cfg = config_io.load()
    pid = body.get("profile", "")
    text = (body.get("text") or "").strip()
    lang = body.get("lang") or "中文"
    opts = body.get("opts") or {}

    if pid not in cfg["profiles"]:
        raise HTTPException(400, "未知音色")
    if not text:
        raise HTTPException(400, "请输入要合成的文本")
    if len(text) > 2000:
        raise HTTPException(400, "单次文本请控制在 2000 字内，长文请分批")
    if lang not in cfg.get("languages", []):
        raise HTTPException(400, f"不支持的语言: {lang}")
    st = _profile_status(cfg, pid)
    if not st["ok"]:
        raise HTTPException(400, f"该音色暂不可用：{st['reason']}")

    jid = uuid.uuid4().hex[:8]
    with _job_lock:
        _state["queue"].append(jid)
    threading.Thread(target=_run_job, args=(jid, cfg, pid, text, lang, opts),
                     daemon=True).start()
    return {"job": jid, "position": len(_state["queue"])}


@app.get("/api/job/{jid}")
def api_job(jid: str) -> dict:
    for item in _history:
        if item.get("job") == jid:
            return item
    with _job_lock:
        pos = (_state["queue"].index(jid) + 1) if jid in _state["queue"] else 0
    if _state.get("running_job") == jid:
        return {"job": jid, "status": "running"}
    if pos:
        return {"job": jid, "status": "queued", "position": pos}
    if jid in _state["cancel"]:
        return {"job": jid, "status": "cancelled"}
    return {"job": jid, "status": "unknown"}


@app.post("/api/cancel/{jid}")
def api_cancel(jid: str) -> dict:
    with _job_lock:
        if jid in _state["queue"]:
            _state["queue"].remove(jid)
            _state["cancel"].add(jid)
            return {"cancelled": True}
    return {"cancelled": False, "msg": "该任务已开始推理，无法中断（CUDA 层不支持取消）"}


def _friendly_error(raw: str) -> str:
    """把底层报错翻译成人话。内存类错误必须给可操作建议——
    G2PW 的 onnx 在提交内存见底时报 bad allocation，重试往往就能过。"""
    low = raw.lower()
    if "bad allocation" in low or "memoryerror" in low or "failed to allocate" in low:
        return (raw[:180] + "\n系统内存不足（提交空间耗尽）。处理："
                "① 直接点重试，常常即可恢复；② 关闭其他吃内存的程序"
                "（浏览器/其他 python 进程）；③ 若 GPU 显存也紧张，"
                "先跑 nvidia-smi 看占用。")
    if "cuda out of memory" in low or "cublas" in low:
        return raw[:180] + "\n显存不足：关闭其他用 GPU 的程序（ComfyUI/游戏），或缩短文本后重试。"
    return raw[:300]


def _run_job(jid: str, cfg: dict, pid: str, text: str, lang: str, opts: dict) -> None:
    with _job_lock:
        if jid in _state["cancel"]:
            _state["cancel"].discard(jid)
            return
    import voicelab

    prof_vo = dict(cfg["profiles"][pid])
    for k in ("gpt", "sovits", "ref_audio", "ref_text"):
        if prof_vo.get(k):
            prof_vo[k] = str(runtime_paths.resolve(prof_vo[k]))
    for k in ("ref_audio", "ref_text", "ref_lang", "speed", "sample_steps",
              "top_k", "top_p", "temperature", "how_to_cut", "seed"):
        if opts.get(k) not in (None, ""):
            prof_vo[k] = opts[k] if k in ("ref_audio", "ref_text", "ref_lang") else prof_vo.get(k)

    args = _Args(**{k: opts.get(k) for k in
                    ("how_to_cut", "top_k", "top_p", "temperature", "speed",
                     "sample_steps", "seed") if opts.get(k) is not None})
    args.no_guard = bool(opts.get("no_guard", False))
    args.guard_attempts = int(opts.get("guard_attempts", 2))
    args.gsv_version = opts.get("gsv_version") or None
    args.prosody = opts.get("prosody") or None
    args.prosody_strength = float(opts.get("prosody_strength") or 1.0)

    # 情绪：合成的入口是 synth()，它只认 prof["aux_ref_audio_paths"]，
    # 所以必须在这里把情绪名解析成实际音频路径。
    # （之前只在 CLI 的 resolve_profile 里做，界面选了情绪等于没生效。）
    # 情绪参考（aux）在 v3 上属于**实验性**能力：
    # 实测会带来机械音/轻微叠层感，且上游 raw_audio 污染问题只能部分规避。
    # 因此默认不使用，必须显式传 aux_emotion:true 才启用。
    # 常规情绪表达请用 prosody（参数化，纯后处理，无副作用）。
    args.emotion = None
    emo_name = opts.get("emotion")
    enable_aux_emo = bool(opts.get("aux_emotion", False))
    if emo_name and not enable_aux_emo:
        print(f"[情绪] 已忽略「{emo_name}」：情绪参考在 v3 上仍是实验特性，"
              f"默认关闭（会有机械音）。建议改用参数化情绪（prosody）。", flush=True)
        emo_name = None
    if emo_name:
        import emotions as _emo
        got = _emo.resolve(pid, [emo_name] if isinstance(emo_name, str) else list(emo_name))
        if got:
            args.emotion = emo_name
            prof_vo["aux_ref_audio_paths"] = [g["audio"] for g in got]
            print(f"[情绪] 融合参考: {', '.join(g['name'] for g in got)}", flush=True)
        else:
            print(f"[情绪] 未登记或文件缺失，已忽略: {emo_name}", flush=True)

    fname = f"{time.strftime('%H%M%S')}_{jid}.wav"
    out = WEB_DIR / fname

    entry = {"job": jid, "status": "running", "profile": pid, "text": text,
             "lang": lang, "ts": time.time(), "file": fname}
    _history.insert(0, entry)
    _save_history()

    with _lock:
        _state["busy"] = True
        _state["running_job"] = jid
        with _job_lock:
            if jid in _state["queue"]:
                _state["queue"].remove(jid)
        t0 = time.perf_counter()
        try:
            voicelab.synth(cfg, prof_vo, text, lang, out, args)
            eng = getattr(voicelab, "_ENGINE", None)
            import soundfile as sf
            info = sf.info(str(out))
            entry.update({
                "status": "done", "url": f"/audio/{fname}",
                "secs": round(time.perf_counter() - t0, 2),
                "duration": round(info.duration, 2),
                "sr": info.samplerate,
                "engine": (eng.model_version if eng else None),
            })
            if eng:
                _state["engine_info"] = {"profile": pid, "version": eng.model_version,
                                         "device": eng.device,
                                         "load_secs": round(eng._load_secs, 1)}
                _state["engine_loaded"] = True
        except SystemExit as e:                      # voicelab 用 sys.exit 报错
            entry.update({"status": "error", "error": _friendly_error(str(e)),
                          "secs": round(time.perf_counter() - t0, 2)})
        except semantic_guard.SemanticGenerationError as e:
            entry.update({"status": "collapsed", "error": str(e)[:400],
                          "secs": round(time.perf_counter() - t0, 2)})
        except Exception as e:
            entry.update({"status": "error",
                          "error": _friendly_error(f"{type(e).__name__}: {e}"),
                          "secs": round(time.perf_counter() - t0, 2)})
        finally:
            _state["busy"] = False
            _state["running_job"] = None
    _save_history()


@app.get("/audio/{fname}")
def audio(fname: str):
    p = (WEB_DIR / Path(fname).name).resolve()
    if not str(p).startswith(str(WEB_DIR.resolve())) or not p.is_file():
        raise HTTPException(404, "文件不存在")
    return FileResponse(str(p), media_type="audio/wav",
                        headers={"Accept-Ranges": "bytes"})


REFS_DIR = LAB / "data" / "refs"
REF_EXT = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".mp4"}
REF_MAX = 60 * 2**20          # 参考音频 60MB 足够（几秒到几分钟）


@app.post("/api/ref/upload")
async def ref_upload(file: UploadFile = File(...),
                     lang: str = Form("中文"),
                     text: str = Form("")) -> dict:
    """上传零样本参考音频。

    m4a/mp3/aac 等 soundfile 读不了的格式，先用 ffmpeg 转成 24k 单声道 wav。
    （libsndfile 不支持 AAC，这是用户实际踩过的坑。）
    """
    suffix = Path(file.filename or "ref.wav").suffix.lower()
    if suffix not in REF_EXT:
        raise HTTPException(400, f"不支持的格式 {suffix}；可用："
                                 f"{', '.join(sorted(REF_EXT))}")
    REFS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    raw = REFS_DIR / f"upload_{stamp}{suffix}"
    size = 0
    with open(raw, "wb") as f:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            if size > REF_MAX:
                f.close()
                raw.unlink(missing_ok=True)
                raise HTTPException(413, "文件超过 60MB，参考音频不需要这么大")
            f.write(chunk)
    if size < 1024:
        raw.unlink(missing_ok=True)
        raise HTTPException(400, "文件太小，可能上传失败")

    dest = raw if suffix == ".wav" else REFS_DIR / f"upload_{stamp}.wav"
    converted = False
    if suffix != ".wav":
        try:
            import video2audio
            video2audio.extract_one(raw, dest, sr=24000, mono=True)
            converted = True
            raw.unlink(missing_ok=True)
        except Exception as e:
            raw.unlink(missing_ok=True)
            raise HTTPException(500, f"转码失败（{suffix} -> wav）：{e}")

    # 读规格，给前端展示；顺便校验时长是否适合做参考
    try:
        import soundfile as sf
        info = sf.info(str(dest))
        dur = info.duration
        sr = info.samplerate
    except Exception as e:
        raise HTTPException(500, f"生成的 wav 读不了：{e}")

    warn = ""
    if dur < 2.5:
        warn = f"只有 {dur:.1f}s，太短会让音色不稳（建议 3~10s）"
    elif dur > 20:
        warn = f"有 {dur:.0f}s，偏长会稀释特征（建议 3~10s）"

    txt_path = dest.with_suffix(".txt")
    if text.strip():
        txt_path.write_text(text.strip(), encoding="utf-8")

    return {
        "audio": str(dest.relative_to(LAB)).replace("\\", "/"),
        "text": str(txt_path.relative_to(LAB)).replace("\\", "/") if text.strip() else "",
        "duration": round(dur, 2), "sr": sr,
        "converted": converted, "from": suffix, "warning": warn,
    }


@app.delete("/api/history/{jid}")
def api_delete(jid: str) -> dict:
    for i, it in enumerate(_history):
        if it.get("job") == jid:
            f = WEB_DIR / Path(it.get("file", "")).name
            if f.is_file():
                try:
                    f.unlink()
                except OSError:
                    pass
            _history.pop(i)
            _save_history()
            return {"deleted": True}
    raise HTTPException(404)


@app.get("/")
def index() -> HTMLResponse:
    page = LAB / "tools" / "webui.html"
    if not page.is_file():
        raise HTTPException(500, "缺少 tools/webui.html")
    return HTMLResponse(page.read_text(encoding="utf-8"))


# 训练面板路由（/api/train/*）。未装 python-multipart 时上传会失败，其余可用。
try:
    import train_api
    app.include_router(train_api.router)
except Exception as _e:      # 不让训练模块的问题拖垮合成功能
    print(f"[警告] 训练面板未启用: {type(_e).__name__}: {_e}", flush=True)


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description="VoiceLab 图形界面")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1",
                    help="默认只监听本机；局域网共享填 0.0.0.0（无鉴权，慎用）")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    runtime_paths.ensure_env()
    _load_history()

    import uvicorn
    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}"
    print(f"VoiceLab 界面: {url}", flush=True)
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
