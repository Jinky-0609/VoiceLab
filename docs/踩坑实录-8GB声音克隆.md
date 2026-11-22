# 8GB 笔记本上把声音克隆跑通：我踩过的 10 个坑

> 这不是"跟着 README 装一遍"的记录，而是**每一步都已经失败过**之后的总结。
> 所有数字来自 RTX 5060 Laptop 8GB 上的实测。

## 为什么选 GPT-SoVITS v3

要做"任意人的声音克隆 + 专属音色微调"，可选项其实不多：

- **v2Pro** 音质好，但用不上 BigVGAN 声码器
- **v3** 既保留 BigVGAN v2 24kHz，又能做 LoRA 微调 → 两全
- **Bert-VITS2 / Fish-Speech** 生态也好，但 8GB 卡上训练资源更吃紧

结论：**v3 架构 + LoRA（s2）+ 全参微调（s1）**。s1 只有 148MB，8GB 卡吃得下；s2 才是显存大户，所以只给它上 LoRA（rank=32，产物 72MB）。

---

## 坑 1：`bert_path` 必须设成**绝对路径**

**现象**：`HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'`

**根因**：`text/chinese2.py:36` 把环境变量 `bert_path` 的默认值（一个相对路径）喂给 g2pw 的 ONNX 加载器，`transformers` 把它当成 HuggingFace 仓库名去解析。

**关键细节**：`bert_pretrained_dir` 和 `bert_path` 是**两个不同变量**，都要设。只设一个还会挂。

```python
os.environ["bert_path"] = str(PRETRAINED / "chinese-roberta-wwm-ext-large")
```

## 坑 2：数据准备"跑完"了，但没有 `2-name2text.txt`

**现象**：`prepare_datasets/*.py` 正常退出，下一步训练却报找不到文件。

**根因**：这些脚本只产出**带分片后缀**的中间文件（`2-name2text-0.txt`、`6-name2semantic-0.tsv`），合并动作是在上游 `webui.py:1105-1112` 和 `1214-1221` 里做的——命令行调用不经过那段代码。

**修法**：自己补合并，且注意两点：
- 语义文件要保留 `item_name\tsemantic_audio` 表头
- 分片必须**按数字排序**拼接，`0,1,10,2` 会毁掉顺序

## 坑 3：s2 训练 `AssertionError`，什么信息都没有

**现象**：断言失败，没有堆栈线索。

**根因**：`module/data_utils.py:288-293` 会在 `data.exp_dir` 下找 `2-name2text.txt` / `4-cnhubert` / `5-wav32k`。而 **`exp_dir` 必须是数据集目录本身**，不能自己开子目录。

**修法**（照抄 `webui.py:535`）：

```python
data.exp_dir = s2_ckpt_dir = 数据集目录
```

## 坑 4：s1 续训 `UnpicklingError: Unsupported global: pathlib.WindowsPath`

**根因**：torch ≥ 2.6 把 `torch.load` 的 `weights_only` 默认改成了 `True`，而旧 checkpoint 里存了 `WindowsPath` 对象。

**修法**：官方开关，恢复旧行为。

```powershell
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = "1"
```

## 坑 5：venv 里没有 pip，不能往里装包

**场景**：训练期需要 `tensorboard` / `absl`，但那个 venv 是 uv 建的、没装 pip。

**修法**：`--target` + `PYTHONPATH` 借道，不污染原环境。

```bash
uv pip install --target train/libs --no-deps tensorboard absl-py protobuf grpcio markdown werkzeug six packaging json5 sacremoses
```

## 坑 6：`ImportError: cannot import name 'setLangfilters'`

**根因**：PyPI 上的 `LangSegment` 和 GPT-SoVITS 要的**不是同一个项目**，只是同名。

**修法**：**不要装**。仓库自带 `GPT_SoVITS/text/LangSegmenter`，训练链路根本不需要 PyPI 那个。

## 坑 7：UVR5 分离报 `Format not recognised`

**根因**：`soundfile` / libsndfile 不支持 m4a/AAC。

**修法**：先转 WAV 再分离。

```bash
python tools/video2audio.py extract -i 素材.m4a -o 素材.wav --mono
```

---

## 坑 8（最阴险）：上游返回 **int16**，不归一化就是满量程噪音

**现象**：合成命令"成功"了，音频能生成、时长正常，但听起来是刺耳的嗡嗡声。

**根因**：`TTS.audio_postprocess` 的契约是 `audio * 32768 → np.int16`。如果直接把这个 int16 数组当 float 交给 `soundfile.write`，超出 [-1, 1] 的值会被**削顶**。

**如何发现**：看数值特征就很明显。

| | mean\|abs\| |
|---|---|
| 正常语音 | ≈ 0.04 |
| 削顶噪音 | **0.89** |

**修法**：

```python
if a.dtype == np.int16 or np.abs(a).max() > 4.0:
    a = a.astype(np.float32) / 32768.0
```

**这个坑的危险在于它不会报错**——我以为自己测试通过了，实际产物是垃圾。教训：**合成结果必须用 ASR 回环验证，不能只看"命令没报错"**。

## 坑 9：语义坍缩——28 个字生成了 52.9 秒乱音

**现象**：一段 28 字的文本，模型输出了 **52.9 秒**的循环乱音。

**触发条件**：参考音频语言与目标语言不一致（日文参考 → 中文目标）时**概率性**出现。注意是概率性，不是必然——我后来同样的配置成功过，ASR 逐字全对。

**如何判定**：在音频层看症状，比在 token 层拦截更省事。

| 规则 | 判据 | 校准数据 |
|---|---|---|
| 长度爆炸 | `dur > max(10, 0.8×字数)` | 正常 0.19 s/字，乱音 1.9 s/字 |
| 周期循环 | 频谱帧间自相似 ≥ 0.72 | 乱音 0.828，正常样本 p99 = 0.597 |
| 无声墙 | ≥12s 且静音帧 < 3% | 乱音 52.9s 仅 0.5% |

阈值是用 **81 个历史样本**校准出来的，不是拍脑袋。被拦下就换 seed 重试，乱音一律不落盘。

## 坑 10：显存与 batch size

| 阶段 | 上游默认 | 8GB 卡实际用 | 实测 |
|---|---|---|---|
| s1（GPT） | 8 | **4** | 15 轮 / 33 批次 ≈ 1.5 分钟，7.6 it/s |
| s2（SoVITS LoRA） | 32 | **3** | rank=32 |

---

## 顺带说：分离是"有损"的

UVR5 用 `Kim_Vocal_2` 分离 30 秒混音，实测：

| 指标 | 数值 |
|---|---|
| 人声保留 | 93.93% |
| 人声轨伴奏残留 | 12.81%（SIR 17.85 dB） |
| 伴奏轨人声残留 | 12.34%（SIR 18.18 dB） |

**约 18 dB 分离度 ≈ 每 8 份人声里仍混着 1 份伴奏。** 这个残留会被训练学进去，所以去混响和片段筛选不能省。

---

## 最后

上面这些坑，加上"合成出来的东西到底对不对"的验证方法（ASR 回环 + 拼音比对，避免同音字误判），我整理成了一个可跑的项目：

**VoiceLab** — 8GB Windows 笔记本上可复现的声音克隆与微调工作台，CLI + 图形界面，自带质量门禁。

> 如果你也在 8GB 卡上折腾 TTS，希望这份清单能帮你省掉几个通宵。

---

## 项目地址

**VoiceLab** — https://github.com/Jinky-0609/VoiceLab

- 图形界面：`tools/webui.py`（引擎常驻，首句之后每次几秒）
- 命令行：`tools/voicelab.py`
- 一键安装：releases 页面的 `install.ps1`
- 自包含：配置路径全相对项目根，可整体搬家；权重由脚本从官方源下载

> 所有数据均来自实机测试（RTX 5060 Laptop 8GB）。
