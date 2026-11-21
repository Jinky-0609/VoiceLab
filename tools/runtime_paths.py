#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""集中管理 VoiceLab 的路径与环境变量 —— 全部落在项目内。

任何脚本 import 本模块即可获得：
  * PROJECT     项目根目录（由文件位置推导，不写死盘符）
  * resolve()   把配置里的相对路径解析成绝对路径（绝对路径原样返回）
  * ensure_env()把 NLTK_DATA / HF_HOME / TMP 等指到 PROJECT/runtime/
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RUNTIME = PROJECT / "runtime"
NLTK_DIR = RUNTIME / "nltk_data"
HF_DIR = RUNTIME / "hf"


def resolve(p: str | os.PathLike | None) -> Path | None:
    """相对路径 -> 相对 PROJECT；绝对路径原样。"""
    if p is None:
        return None
    q = Path(str(p))
    return q if q.is_absolute() else (PROJECT / q)


def resolve_str(p, default=None):
    r = resolve(p)
    return str(r) if r is not None else default


def project_python(name: str = ".venv-tts") -> str:
    """项目内虚拟环境的解释器绝对路径（不存在则退回当前解释器）。"""
    for cand in (
        PROJECT / name / "Scripts" / "python.exe",     # Windows
        PROJECT / name / "bin" / "python",             # POSIX
    ):
        if cand.is_file():
            return str(cand)
    return sys.executable


def ensure_env() -> None:
    """把一切可写缓存收进项目，保证：换机器/只读介质/多人共用都不互相污染。

    顺序很重要：必须在 import 上游 GPT-SoVITS 之前调用。
    """
    RUNTIME.mkdir(exist_ok=True)
    NLTK_DIR.mkdir(exist_ok=True)
    HF_DIR.mkdir(exist_ok=True)

    # nltk：cmudict + averaged_perceptron_tagger_eng（英文前端用）。
    # 项目内优先；若缺失且用户主目录有旧数据，也一并列入搜索路径。
    parts = [str(NLTK_DIR)]
    home_data = Path.home() / "AppData" / "Roaming" / "nltk_data"
    if not home_data.is_dir():
        home_data = Path.home() / "nltk_data"
    if home_data.is_dir():
        parts.append(str(home_data))
    os.environ["NLTK_DATA"] = os.pathsep.join(parts)

    os.environ.setdefault("HF_HOME", str(HF_DIR))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME / "cache"))
    # GPT-SoVITS 上游用相对路径找中文多音字模型
    os.environ.setdefault("NLTK_DOWNLOAD_PROGRESS", "")


def bootstrap_upstream_path() -> None:
    """让 `import tools.xxx` / `TTS_infer_pack` 能找到上游仓库。"""
    repo = PROJECT / "_src" / "GPT-SoVITS-main"
    for p in (repo / "GPT_SoVITS", repo):
        s = str(p)
        if p.is_dir() and s not in sys.path:
            sys.path.insert(0, s)
