#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""核心逻辑单元测试（不需要 GPU，秒级完成）。

    .venv-tts/Scripts/python.exe tools/test_core.py

延续 test_guard.py 的风格：零依赖、自断言、退出码可接 CI。
覆盖的是最近踩过坑的纯逻辑部分——这些地方出问题不会报错，只会静默出错。
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "tools"))

import config_io        # noqa: E402
import delete_voice     # noqa: E402
import emotions         # noqa: E402
import train_api        # noqa: E402

# _profile_status 在 webui 里（不在 train_api）。导入它会拉起 fastapi，
# 但那是纯依赖、不加载模型，秒级且不需要 GPU。
try:
    import webui            # noqa: E402
except Exception as _e:     # fastapi 缺失时容忍，相关用例跳过
    webui = None
    _WEBUI_ERR = f"{type(_e).__name__}: {_e}"

fails = 0
SECTION = ""


def sec(name: str) -> None:
    global SECTION
    SECTION = name
    print(f"\n[{name}]")


def expect(name: str, cond: bool, detail: str = "") -> None:
    global fails
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        fails += 1


# ---------------------------------------------------------------- config_io

sec("config_io：bootstrap / save / 原子写")
tmp = Path(tempfile.mkdtemp(prefix="vl_test_"))
try:
    cfg_path = config_io.CONFIG_PATH
    example = config_io.EXAMPLE_PATH
    backup = None
    if cfg_path.is_file():
        backup = cfg_path.read_bytes()

    # 模拟"没有 config.json"：应自动从 example 引导
    if cfg_path.is_file():
        cfg_path.unlink()
    cfg = config_io.load(verbose=False)
    expect("缺失时自动引导出 config.json", cfg_path.is_file() and "profiles" in cfg)

    # save 后能读回，且是合法 JSON
    cfg.setdefault("profiles", {})["__t"] = {"gpt": "x", "sovits": "y"}
    config_io.save(cfg)
    reread = config_io.load(verbose=False)
    expect("save/load 往返一致", reread["profiles"].get("__t", {}).get("gpt") == "x")
    expect("save 不留临时文件", not cfg_path.with_suffix(".json.tmp").exists())

    # 原子性：tmp 文件存在不应影响读取（模拟中途崩溃）
    tmpf = cfg_path.with_suffix(".json.tmp")
    tmpf.write_text("{broken", encoding="utf-8")
    ok = True
    try:
        config_io.load(verbose=False)
    except Exception:
        ok = False
    expect("残留 .tmp 不影响读取", ok)
    tmpf.unlink(missing_ok=True)

    if backup is not None:
        cfg_path.write_bytes(backup)
finally:
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- emotions

sec("emotions：增删查 / 非法名拒绝")
spk = "__test_voice__"
try:
    d = emotions._spk_dir(spk)
    shutil.rmtree(d, ignore_errors=True)

    expect("空库返回 {}", emotions.load(spk) == {})

    # 造一个假音频
    d.mkdir(parents=True, exist_ok=True)
    fake = d / "happy.wav"
    fake.write_bytes(b"RIFF" + b"\0" * 100)

    rel = str(fake.relative_to(LAB)).replace("\\", "/")
    emotions.add(spk, "开心", rel, "测试文本", "中文")
    lib = emotions.load(spk)
    expect("add 后可读到", "开心" in lib)
    expect("add 保留文本", lib["开心"]["text"] == "测试文本")

    # resolve 返回绝对路径
    got = emotions.resolve(spk, ["开心"])
    expect("resolve 找到条目", len(got) == 1 and Path(got[0]["audio"]).is_file())
    expect("resolve 忽略不存在", emotions.resolve(spk, ["不存在"]) == [])

    # 非法名
    bad_ok = False
    try:
        emotions.add(spk, "../evil", rel, "", "中文")
    except ValueError:
        bad_ok = True
    expect("拒绝路径穿越式情绪名", bad_ok)

    expect("remove 成功", emotions.remove(spk, "开心"))
    expect("remove 后为空", emotions.load(spk) == {})
finally:
    shutil.rmtree(emotions._spk_dir(spk), ignore_errors=True)


# ---------------------------------------------------------------- delete_voice

sec("delete_voice：资产收集与共享保护")
cfg = {
    "profiles": {
        "fake": {
            "gpt": "models/voices/fake/s1/x.ckpt",
            "sovits": "models/voices/fake/s2/x.pth",
            "ref_audio": "data/refs/fake_ref.wav",
        }
    }
}
fake_dir = LAB / "models" / "voices" / "fake"
try:
    (fake_dir / "s1").mkdir(parents=True, exist_ok=True)
    (fake_dir / "s1" / "x.ckpt").write_bytes(b"0" * 100)

    items = delete_voice.collect("fake", cfg, include_shared=False)
    labels = [t for t, _ in items]
    paths = [p for _, p in items]

    expect("收集到训练权重目录", any("训练权重" == t for t in labels))
    expect("默认不含预训练权重",
           not any("pretrained" in str(p) for p in paths))
    expect("默认不含 default 参考",
           not any("default" in p.name for p in paths))

    # 共享资产需显式放行
    shared = delete_voice.collect("fake", cfg, include_shared=True)
    expect("include_shared 才纳入更多", len(shared) >= len(items))
finally:
    shutil.rmtree(fake_dir, ignore_errors=True)


# ---------------------------------------------------------------- _find_list

sec("train_api._find_list：.list 路径自适应")
spk2 = "__test_list__"
base = LAB / "data" / "raw" / spk2
try:
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)

    expect("无文件时返回 None", train_api._find_list(spk2) is None)

    # 情形 A：直接放在 <sp>/
    a = base / f"{spk2}.list"
    a.write_text("a\tb\tc\td\n", encoding="utf-8")
    expect("可定位 <sp>/<sp>.list", train_api._find_list(spk2) == a)

    # 情形 B：asr_annotate 实际写到 <sp>/<sp>/<sp>.list
    a.unlink()
    nested = base / spk2
    nested.mkdir(exist_ok=True)
    b = nested / f"{spk2}.list"
    b.write_text("a\tb\tc\td\n" * 3, encoding="utf-8")
    expect("可定位嵌套 <sp>/<sp>/<sp>.list", train_api._find_list(spk2) == b)
finally:
    shutil.rmtree(base, ignore_errors=True)


# ---------------------------------------------------------------- profile 状态

sec("webui._profile_status：profile 可用性判定")
if webui is None:
    print("  SKIP  webui 未导入（缺 fastapi？）")
try:
    if webui is None:
        raise SystemExit
    _ps = webui._profile_status
    # 零样本缺参考音频 -> 仍可用，只标记 needs_ref（曾误判为不可用）
    st = _ps(
        {"profiles": {"zs": {"gpt": "models/pretrained/s1v3.ckpt",
                             "sovits": "models/pretrained/s2Gv3.pth",
                             "ref_audio": "", "ref_text": ""}}}, "zs")
    expect("零样本缺参考仍可用", st["ok"] is True)
    expect("零样本标记 needs_ref", st["needs_ref"] is True)
    expect("零样本被识别", st["is_zeroshot"] is True)

    # 权重缺失 -> 不可用（硬条件）
    st2 = _ps(
        {"profiles": {"bad": {"gpt": "models/nope.ckpt",
                              "sovits": "models/nope.pth",
                              "ref_audio": ""}}}, "bad")
    expect("权重缺失判为不可用", st2["ok"] is False)

    # 未知 profile
    st3 = _ps({"profiles": {}}, "ghost")
    expect("未知 profile 判为不可用", st3["ok"] is False)
except SystemExit:
    pass
except Exception as e:
    expect("profile 状态检查未抛异常", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------- 汇总

print("\n" + "=" * 50)
print(f"{'== 全部通过 ==' if fails == 0 else f'== {fails} 项失败 =='}")
sys.exit(1 if fails else 0)
