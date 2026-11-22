# -*- coding: utf-8 -*-
"""nltk 数据自修复：把 zip 解压成目录，并建立上游需要的旧名别名。

## 为什么需要这个文件

实测（2026-11-23）三种语言出问题，根因各不相同，其中一个非常隐蔽：

* **英文生成空白音频**（0.84s，peak 0.0094 几乎全静音，但**不报错**）
  根因：nltk 3.10 不再认 `.zip` 形式的数据包，必须解压成目录；
  并且上游 `text/english.py` 的 `pos_tag()` 找的是**旧资源名**
  `averaged_perceptron_tagger`，而我们装的是新名 `_eng` 版本。
  两者对不上 -> 词性标注返回空 -> 前端拿不到音素 -> 输出一段静音。
  **这条最危险：整个链路零异常，只是结果是哑的。**

* 粤语缺 `ToJyutping`
* 韩文缺 `jamo` + `ko_pron` + `g2pk2` + `eunjeon`（Windows 上 mecab 的替代）

调用时机：必须在 import 上游 GPT-SoVITS 之前。
"""
from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path


def _unpack(zip_path: Path) -> None:
    """把 nltk 的 zip 就地解压成目录（nltk 3.10+ 只认目录）。"""
    if not zip_path.is_file():
        return
    out = zip_path.with_suffix("")
    if out.is_dir():
        return
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(zip_path.parent)
        print(f"[nltk] 解压 {zip_path.name} -> {out.name}/", flush=True)
    except Exception as e:
        print(f"[nltk] 解压失败 {zip_path.name}: {type(e).__name__}: {e}", flush=True)


def _alias_tagger(taggers: Path) -> None:
    """上游 pos_tag 要旧名 averaged_perceptron_tagger，nltk 3.10 只有 _eng。

    把 _eng 目录复制成旧名，并把文件名里的 _eng 去掉，使旧名的加载器能命中。
    """
    src = taggers / "averaged_perceptron_tagger_eng"
    dst = taggers / "averaged_perceptron_tagger"
    if not src.is_dir():
        return
    if dst.is_dir():
        return
    try:
        shutil.copytree(src, dst)
    except Exception as e:
        print(f"[nltk] 建别名失败: {e}", flush=True)
        return
    for f in list(dst.iterdir()):
        if f.name.endswith("_eng.weights.json"):
            f.rename(dst / f.name.replace("_eng.weights.json", ".weights.json"))
        elif f.name.endswith("_eng.classes.json"):
            f.rename(dst / f.name.replace("_eng.classes.json", ".classes.json"))
    print("[nltk] 已建别名 averaged_perceptron_tagger（上游 pos_tag 用旧名）", flush=True)


def repair(nltk_dir: Path) -> None:
    if not nltk_dir.is_dir():
        return
    for sub in ("taggers", "corpora", "tokenizers", "stemmers"):
        d = nltk_dir / sub
        if not d.is_dir():
            continue
        for z in d.glob("*.zip"):
            _unpack(z)
    _alias_tagger(nltk_dir / "taggers")


def verify(nltk_dir: Path) -> list[str]:
    """返回缺失的资源名列表（空 = 全齐）。只检查英文前端必需的两个。"""
    os.environ["NLTK_DATA"] = str(nltk_dir)
    need = ["taggers/averaged_perceptron_tagger", "corpora/cmudict"]
    missing = []
    try:
        from nltk.data import find
    except ImportError:
        return need
    for r in need:
        try:
            find(r)
        except LookupError:
            missing.append(r)
    return missing


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1] / "runtime" / "nltk_data"
    repair(base)
    miss = verify(base)
    print("nltk 数据:", "全部就绪" if not miss else f"仍缺 {miss}")
