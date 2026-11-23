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
    missing = []
    for key, label in (("gpt", "GPT 权重"), ("sovits", "SoVITS 权重"),
                       ("ref_audio", "参考音频")):
        v = p.get(key)
        if not v:
            missing.append(f"{label}未配置")
        elif runtime_paths.resolve(v) is None or not Path(runtime_paths.resolve(v)).is_file():
            missing.append(f"{label}缺失")
    rt = p.get("ref_text")
    if rt:
        rp = runtime_paths.resolve(rt)
        is_file = bool(rp and Path(rp).is_file())
        if not is_file and len(str(rt)) > 120:
            missing.append("参考文本文件缺失")   # 长路径才当文件看待
    if str(rt or "").strip().startswith("（"):
        missing.append("参考文本是占位符，需填逐字内容")
    reason = "、".join(missing)
    # 零样本缺素材时给可操作的引导，而不是干巴巴一句"参考音频缺失"
    if missing and "pretrained" in str(p.get("gpt", "")).lower():
        reason = ("零样本需要一段参考音频：把自己 3~10 秒的干声拖进"
                  "「高级 → 参考音频」，或放到 data/refs/my_ref.wav 并配同名 .txt")
    return {
        "ok": not missing,
        "reason": reason,
        "ref_lang": p.get("ref_lang", ""),
        "desc": p.get("desc", ""),
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
