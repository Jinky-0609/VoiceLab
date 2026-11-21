#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VoiceLab 本地推理引擎。

直接驱动 **本地上游仓库** `_src/GPT-SoVITS-main/GPT_SoVITS/TTS_infer_pack/TTS.py`
里的 `TTS` 类，不引用任何外部项目。依赖为本项目自己的 `.venv-tts`。

自包含约定（tools/runtime_paths.py）：
  * 上游需要 cwd 与代码同盘符 —— 本文件在 import / run 前 chdir 到仓库根；
  * 中文多音字模型 G2PW 放在 `LAB/GPT_SoVITS/text/G2PWModel`，上游按相对路径找它；
  * nltk 数据、HF 缓存都落在项目内 `runtime/`，不写用户主目录。

用法（供 voicelab.py 调用）：
    from tts_engine import TTSInferencer
    eng = TTSInferencer(gpt_path=..., sovits_path=...)
    sr, audio = eng.infer(text=..., ref_audio_path=..., prompt_text=...,
                          text_language="中文", prompt_language="中文")
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
REPO = LAB / "_src" / "GPT-SoVITS-main"
GSV = REPO / "GPT_SoVITS"
PRETRAINED = GSV / "pretrained_models"

# VoiceLab 的中文语言名 -> GPT-SoVITS 的 text_lang 代码
LANG_MAP = {
    "中文": "zh",
    "普通话": "zh",
    "日文": "ja",
    "日语": "ja",
    "英文": "en",
    "英语": "en",
    "粤语": "yue",
    "韩文": "ko",
    "韩语": "ko",
    "中英混合": "zh",
    "多语种": "auto",
}


def _to_lang_code(name: str) -> str:
    if not name:
        return "zh"
    n = name.strip()
    if n in LANG_MAP:
        return LANG_MAP[n]
    # 已经是代码形式就直接用
    if n in ("zh", "ja", "en", "yue", "ko", "auto", "all_zh", "all_ja", "all_yue", "all_ko"):
        return n
    return "zh"


class TTSInferencer:
    """驱动上游 GPT-SoVITS `TTS` 的薄封装：显式传权重路径，绕开上游一堆相对路径默认值。"""

    def __init__(self, gpt_path: str, sovits_path: str, device: str | None = None):
        gpt_path = str(Path(gpt_path).resolve())
        sovits_path = str(Path(sovits_path).resolve())
        for p, label in ((gpt_path, "GPT(s1)"), (sovits_path, "SoVITS(s2)")):
            if not Path(p).is_file():
                raise FileNotFoundError(f"{label} 权重不存在: {p}")

        self._check_env()

        # 上游 TTS.py 用 os.getcwd() 作为 now_dir 拼接 pretrained_models，
        # 所以必须在 import 之前把 cwd 切到仓库根。
        old_cwd = os.getcwd()
        os.chdir(REPO)
        try:
            # 让 `import tools.xxx` / `import feature_extractor.xxx` 等找到包
            for p in (str(GSV), str(REPO)):
                if p not in sys.path:
                    sys.path.insert(0, p)

            # 显式传入绝对路径，绕开上游那些「相对路径当 repo id」的默认值
            os.environ["bert_path"] = str(PRETRAINED / "chinese-roberta-wwm-ext-large")
            os.environ["cnhubert_base_path"] = str(PRETRAINED / "chinese-hubert-base")
            try:
                import runtime_paths
                runtime_paths.ensure_env()
            except ImportError:
                os.environ.setdefault("NLTK_DATA", str(LAB / "runtime" / "nltk_data"))
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            os.environ.setdefault("HF_ENDPOINT", os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))

            # 必须在 import TTS_infer_pack.TTS 之前打补丁（它在模块级 import torchaudio）
            self._patch_torchaudio_load()

            from TTS_infer_pack.TTS import TTS  # noqa: E402

            import torch

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            is_half = device == "cuda"

            version = self._detect_version(sovits_path)
            t0 = time.perf_counter()
            # 上游 TTS_Config.__init__ 里有这行：
            #   self.configs = configs_.get("custom", configs_["v2"])
            # 不包一层 "custom" 就直接取 v2 默认配置，我们传的路径全被丢掉，
            # 于是它去找 gsv-v2final-pretrained/... 而报 FileNotFoundError。
            cfg = {
                "custom": {
                    "device": device,
                    "is_half": is_half,
                    "version": version,
                    "t2s_weights_path": gpt_path,
                    "vits_weights_path": sovits_path,
                    "cnhuhbert_base_path": str(PRETRAINED / "chinese-hubert-base"),
                    "bert_base_path": str(PRETRAINED / "chinese-roberta-wwm-ext-large"),
                }
            }
            self.tts = TTS(cfg)
            self.model_version = version
            self.device = device
            self._load_secs = time.perf_counter() - t0

            # Windows 上没装 flash_attn 时，上游会回退到 torch_static_cuda_graph。
            # 那个后端的 CUDA graph 每次进程都要重建，实测单条 3 秒音频要 60~67 秒
            # （RTF 20~43），比普通推理慢一个数量级。这里直接禁用改为普通推理。
            self._disable_cuda_graph_backend()
        finally:
            os.chdir(old_cwd)

    @staticmethod
    def _disable_cuda_graph_backend() -> None:
        """让上游的 create_acceleration 永远返回 None（普通推理）。

        TTS.py:19 是 `from tools.acceleration import create_acceleration`，
        即模块级绑定。所以光改 tools.acceleration 里的函数没用，
        必须直接替换 TTS 模块里的那个名字。

        背景：没装 flash_attn 时 resolve_acceleration 会回退到
        torch_static_cuda_graph，而该后端的 CUDA graph 每个进程都要重建。
        实测单条 3 秒音频 67 秒（RTF 20.9），第二次 60 秒（RTF 43），
        比普通推理慢一个数量级，所以这里关掉。
        """
        mod = sys.modules.get("TTS_infer_pack.TTS")
        if mod is None:
            return
        if getattr(mod, "_vl_accel_patched", False):
            return

        def _no_accel(*args, **kwargs):
            return None

        mod.create_acceleration = _no_accel
        mod._vl_accel_patched = True

    @staticmethod
    def _patch_torchaudio_load() -> None:
        """torchaudio 2.9+ 的 torchaudio.load 默认走 torchcodec，而 torchcodec 需要
        FFmpeg 的【共享】运行时 DLL（系统里的 Gyan 版是静态编译，没有 avcodec-*.dll），
        于是报 "Could not find module libtorchcodec_core4.dll"。
        这里把 torchaudio.load 换成 soundfile 实现 —— 行为等价，且不依赖 torchcodec。
        """
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio

        if getattr(torchaudio.load, "_vl_patched", False):
            return

        _orig = torchaudio.load

        def _load(path, *a, **kw):
            # 先试 soundfile（wav/flac/ogg 都没问题）
            try:
                data, sr = sf.read(str(path), dtype="float32", always_2d=True)
                # soundfile 返回 (frames, channels)，torchaudio 要 (channels, frames)
                t = torch.from_numpy(np.ascontiguousarray(data.T))
                return t, int(sr)
            except Exception:
                return _orig(path, *a, **kw)

        _load._vl_patched = True
        torchaudio.load = _load

    # ------------------------------------------------------------------ 内部

    @staticmethod
    def _detect_version(sovits_path: str) -> str:
        """LoRA .pth 的头两字节是 GPT-SoVITS 的版本标记（v3 = b'03'）。"""
        vmap = {b"03": "v3", b"04": "v4", b"05": "v2Pro", b"06": "v2ProPlus", b"07": "v5dev", b"08": "v5turbo"}
        try:
            with open(sovits_path, "rb") as f:
                magic = f.read(2)
            if magic in vmap:
                return vmap[magic]
        except OSError:
            pass
        # 退回文件名判断
        name = Path(sovits_path).name.lower()
        for k in ("v4", "v2proplus", "v2pro", "v3"):
            if k in name:
                return {"v4": "v4", "v2proplus": "v2ProPlus", "v2pro": "v2Pro", "v3": "v3"}[k]
        return "v3"

    @staticmethod
    def _check_env() -> None:
        missing = [n for n in ("chinese-roberta-wwm-ext-large", "chinese-hubert-base")
                   if not (PRETRAINED / n).is_dir()]
        if missing:
            raise FileNotFoundError(
                "缺少预训练权重目录: " + ", ".join(missing)
                + f"\n应放在: {PRETRAINED}"
                + "\n可用 `python tools/fetch_models.py` 重新下载。"
            )
        bigv = PRETRAINED / "models--nvidia--bigvgan_v2_24khz_100band_256x"
        if not bigv.is_dir():
            raise FileNotFoundError(
                f"缺 BigVGAN 声码器目录: {bigv}\n可用 `python tools/fetch_models.py` 下载。"
            )

    # ------------------------------------------------------------------ 对外

    def infer(
        self,
        text: str,
        ref_audio_path: str,
        prompt_text: str,
        text_language: str = "中文",
        prompt_language: str = "中文",
        how_to_cut: str = "不切",
        top_k: int = 20,
        top_p: float = 0.6,
        temperature: float = 0.6,
        speed: float = 1.0,
        sample_steps: int = 16,
        pause_second: float = 0.3,
        if_sr: bool = False,
        repetition_penalty: float = 1.35,
        seed: int = -1,
        **kwargs,
    ):
        ref_audio_path = str(Path(ref_audio_path).resolve())
        if not Path(ref_audio_path).is_file():
            raise FileNotFoundError(f"参考音频不存在: {ref_audio_path}")

        inputs = {
            "text": text,
            "text_lang": _to_lang_code(text_language),
            "ref_audio_path": ref_audio_path,
            "prompt_text": prompt_text or "",
            "prompt_lang": _to_lang_code(prompt_language),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "temperature": float(temperature),
            "text_split_method": self._cut_method(how_to_cut),
            "batch_size": 1,
            "speed_factor": float(speed),
            "fragment_interval": float(pause_second),
            "seed": int(seed),
            "parallel_infer": True,
            "repetition_penalty": float(repetition_penalty),
            "sample_steps": int(sample_steps),
            "super_sampling": bool(if_sr),
        }

        # run() 内部也可能用到 cwd 相对路径
        import numpy as np

        old_cwd = os.getcwd()
        os.chdir(REPO)
        try:
            for item in self.tts.run(inputs):
                # 非流式时只 yield 一次 (sr, ndarray)
                sr, audio = item[0], item[1]
                # 上游 audio_postprocess 的契约是 int16（audio * 32768）。
                # 这里统一转成 float32 [-1, 1] 再交给调用方：
                # 直接拿 int16 当 float 写盘会被 soundfile 削顶成满量程噪音
                # （实测 mean|abs| 从正常的 ~0.04 变成 ~0.89）。
                a = np.asarray(audio)
                if a.dtype == np.int16 or (a.size and float(np.abs(a.astype(np.float64)).max()) > 4.0):
                    a = a.astype(np.float32) / 32768.0
                else:
                    a = a.astype(np.float32)
                return sr, a
        finally:
            os.chdir(old_cwd)

        raise RuntimeError("TTS.run 没有返回音频")

    @staticmethod
    def _cut_method(how_to_cut: str) -> str:
        return {
            "不切": "cut0",
            "凑四句一切": "cut1",
            "凑50字一切": "cut2",
            "按中文句号切": "cut3",
            "按英文句号切": "cut4",
            "按标点切": "cut5",
        }.get(how_to_cut, "cut0")
