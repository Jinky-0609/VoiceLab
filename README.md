# VoiceLab —— 独立音色实验室

本地声音克隆与微调工作台：基于 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) v3，
支持**零样本克隆**（给一段参考音频即可）与 **LoRA 微调**（10 分钟以上干声，训出专属音色），
配 Qwen3-ASR 自动标注与 UVR5 人声分离，全流程 CLI 驱动。

> **自包含承诺**：推理、训练、运行时缓存全部落在本项目目录内。
> 配置里的路径一律**相对项目根**，整个目录可以任意改名、移动、拷到别的机器。

---

## 目录结构

```
VoiceLab/
├─ config.json              音色档案 + 语言表 + 默认参数（路径相对项目根）
├─ models/
│  ├─ pretrained/           方案2 预训练对（s1v3.ckpt / s2Gv3.pth）
│  ├─ voices/<说话人>/       方案3 微调产物（gpt + sovits LoRA）
│  └─ asr/                  Qwen3-ASR-0.6B（自动标注用）
├─ data/
│  ├─ refs/                 参考音频与对应文本
│  └─ <说话人>/              训练数据集 + 中间产物
├─ tools/
│  ├─ voicelab.py           ★ 合成 CLI（含语义坍缩守卫）
│  ├─ tts_engine.py         推理引擎（驱动上游 TTS 类）
│  ├─ semantic_guard.py     生成坍缩守卫（音频层判定）
│  ├─ runtime_paths.py      路径与环境集中管理（可移植性核心）
│  ├─ asr_annotate.py       ASR 自动标注 -> .list
│  ├─ asr_check.py          ASR 回环校验（按**拼音**判可懂度）
│  ├─ split_silence.py      按静音切分长音频
│  ├─ train_pipeline.py     方案3 微调流水线
│  ├─ uvr5.py               人声分离 / 去混响
│  ├─ video2audio.py        视频抽取音频
│  ├─ fetch_models.py       一键补齐预训练权重
│  └─ test_guard.py         守卫回归测试
├─ runtime/                 nltk_data / HF 缓存（不写用户主目录）
├─ out/                     所有合成结果
├─ _src/GPT-SoVITS-main/    上游推理与训练代码（vendored）
└─ logs/                    运行日志
```

---

## 快速开始

三个虚拟环境各司其职（互不干扰）：

| 环境 | 用途 | 解释器 |
|---|---|---|
| `.venv-tts` | 推理 + 训练（torch cu128） | `.venv-tts/Scripts/python.exe` |
| `.venv-asr` | Qwen3-ASR 自动标注 | `.venv-asr/Scripts/python.exe` |
| `.venv-uvr` | UVR5 人声分离（CPU 即可） | `.venv-uvr/Scripts/python.exe` |

从零部署见 `docs/环境重建.md`。

### 列出音色档案
```bash
.venv-tts/Scripts/python.exe tools/voicelab.py list
```

### 合成一句
```bash
# 零样本克隆（音色由参考音频决定）
.venv-tts/Scripts/python.exe tools/voicelab.py speak \
  -p zeroshot -t "你好，这是测试。" -l 中文 -o out/test.wav

# 换成自己的参考音频
.venv-tts/Scripts/python.exe tools/voicelab.py speak \
  -p zeroshot -r data/refs/我的参考.wav -R data/refs/我的参考.txt \
  --ref-lang 中文 -t "要合成的话" -l 中文 -o out/out.wav

# 已微调好的音色（profile 名换成你自己的）
.venv-tts/Scripts/python.exe tools/voicelab.py speak \
  -p myvoice -t "你好" -l 中文 -o out/x.wav
```

### 图形界面（推荐日常使用）

```bash
.venv-tts/Scripts/python.exe tools/webui.py            # 自动开浏览器，http://127.0.0.1:8765
.venv-tts/Scripts/python.exe tools/webui.py --port 9000 --no-browser
```

CLI 每次合成都要重新加载模型（25~60 秒），`webui.py` 把引擎常驻，**首句之后每次只要几秒**。
界面提供：音色/语言下拉（自动禁用权重缺失的档案）、跨语言参考提醒、
语速/采样步数/top_p 等高级参数、临时参考音频覆盖、历史列表（试听/保存/删除/失败重试）。
合成结果同样过语义坍缩守卫，被拦下的乱音不落盘。

只监听 `127.0.0.1`；要局域网共享用 `--host 0.0.0.0`（**无鉴权，慎用**）。
音频与历史落在 `out/web/`（不入库）。

### 批量合成
`lines.jsonl`：
```json
{"text":"第一句","lang":"中文","out":"0001.wav"}
{"text":"第二句","lang":"日文","out":"0002.wav"}
```
```bash
.venv-tts/Scripts/python.exe tools/voicelab.py batch -p myvoice -i lines.jsonl -d out/batch01
```

### 输出路径规则
`-o` / `-d` 相对**项目根**解析：`out/x.wav`、`x.wav`（自动进 `out/`）、绝对路径，三种写法都对，
不会出现 `out/out/` 那种双重拼接。

---

## 语义坍缩守卫

v3 模型在跨语言参考等场景下会概率性进入 token 坍缩，产出循环乱音
（实测 28 字文本生成 52.9 秒）。`tools/semantic_guard.py` 在**音频层**按症状判定，
乱音一律不落盘，并自动换 seed 重试（`--guard-attempts`，默认 2 次）。

| 规则 | 判据 | 校准依据 |
|---|---|---|
| `length_explosion` | `dur > max(10, 0.8·字数)` 秒 | 正常 0.19 s/字，乱音 1.9 s/字 |
| `periodic_repetition` | 频谱帧间自相似 ≥ 0.72 | 乱音 0.828，正常样本 p99 = 0.597 |
| `no_pause_drone` | ≥12s 且静音帧 < 3% | 乱音 52.9s 仅 0.5% |
| `bad_scale_audio` / `saturated_audio` | 量纲错误 / mean\|abs\| > 0.5 | 正常 ≈0.04，削顶 ≈0.89 |

阈值与背景见 `tools/semantic_guard.py` docstring；回归测试 `tools/test_guard.py`。

**注意**：跨语言参考**不是必然失败**——实测「日文参考→中文目标」多数时候正常
（ASR 回环逐字全对），坍缩是概率事件。守卫按症状拦截，而不是按语言配对一刀切。
想要稳定的目标语言音色，仍然应该用该语言的参考音频。

---

## ⚠️ 三条必须知道的技术规则（实测得出）

### 1. cwd 必须与代码同盘符，且要有 `GPT_SoVITS/text/G2PWModel`
上游 `tools/i18n/i18n.py` 用 `os.path.relpath(__file__)`，跨盘符会报
`path is on mount 'D:', start on mount 'C:'`；`chinese2.py` 用相对路径找中文多音字模型。
`voicelab.py` 已自动 `os.chdir(项目根)` + 项目内置 G2PW，无需手动干预。

### 2. 上游返回 int16，必须归一化再写盘
`TTS.audio_postprocess` 的契约是 `audio * 32768 → np.int16`。
直接当 float 交给 `soundfile.write` 会被削顶成满量程噪音（听感刺耳，
数值特征 mean|abs| ≈ 0.89，正常 ≈ 0.04）。`tts_engine.infer()` 已统一转
float32 [-1,1]，守卫的 `saturated_audio` 规则专门兜这类错误。

### 3. 英文合成需要 nltk 数据
`cmudict` + `averaged_perceptron_tagger_eng` 已随项目在 `runtime/nltk_data/`，
`runtime_paths.ensure_env()` 自动设置 `NLTK_DATA`（并兼容用户主目录的旧数据）。

---

## 方案 3：微调专属音色（v3 LoRA）

### 完整流程
```bash
# ① 准备原始音频（单人、干声、无 BGM），建议 10 分钟以上
# ②（可选）人声分离 + 去混响
.venv-uvr/Scripts/python.exe tools/uvr5.py make-dataset -i 素材.flac -o data/raw/小明

# ③ 按静音切分成片段
.venv-tts/Scripts/python.exe tools/split_silence.py -i data/raw/小明/长音频.wav -o data/raw/小明 --max-len 10

# ④ 用 Qwen3-ASR 自动标注（产出 .list）
.venv-asr/Scripts/python.exe tools/asr_annotate.py -i data/raw/小明 -s 小明 -l 中文

# ⑤ 数据准备（文本清洗 / HuBERT / 语义 token）
.venv-tts/Scripts/python.exe tools/train_pipeline.py prepare -s 小明 -l data/raw/小明/小明.list

# ⑥ 训练（GPT + SoVITS LoRA）
.venv-tts/Scripts/python.exe tools/train_pipeline.py train-only -s 小明 --epochs-s1 15 --epochs-s2 8

# ⑦ 把产物写进 config.json 的 profiles（路径写相对），然后试听
.venv-tts/Scripts/python.exe tools/voicelab.py speak -p 小明 -t "你好" -l 中文 -o x.wav
```

### 训练参数说明
| 项 | 值 | 说明 |
|---|---|---|
| GPT 起点 | `models/pretrained/s1v3.ckpt` | 预训练 s1，v3 架构 |
| SoVITS 起点 | `models/pretrained/s2Gv3.pth` | 预训练 s2 |
| LoRA rank | **32** | 产出 `.pth` < 100 MB，即上游 `if_lora_v3` 认的格式 |
| 训练脚本 | `s2_train_v3_lora.py` | 上游仓库内 |
| 声码器 | BigVGAN v2 24kHz | 微调后仍享有 v3 的高音质 |

**为什么走 v3 而不是 v2Pro**：微调作用在 v3 架构上，既拿到目标人音色，
又保留 BigVGAN 声码器（v2Pro 路线用不上它）。

### ⚠️ 训练踩坑记录（全部实测踩过，已修在 `tools/train_pipeline.py` 里）

| 现象 | 根因 | 处理 |
|---|---|---|
| `HFValidationError: Repo id must be in the form ...` | `text/chinese2.py:36` 把 `bert_path` 环境变量默认值（相对路径）喂给 g2pw 的 ONNX 加载器，被 transformers 当成 HF 仓库名 | 设成**绝对路径**。注意 `bert_pretrained_dir` 和 `bert_path` 是两个变量，都要设 |
| 数据准备"跑完"却没有 `2-name2text.txt` | `prepare_datasets/*.py` 只产出带分片后缀的 `2-name2text-0.txt` / `6-name2semantic-0.tsv`，合并是上游 `webui.py:1105-1112`、`1214-1221` 做的 | 流水线里补了合并。语义文件要保留 `item_name\tsemantic_audio` 表头，且分片需按数字排序 |
| s2 训练 `AssertionError`（无任何信息） | `module/data_utils.py:288-293` 会在 `data.exp_dir` 下找 `2-name2text.txt` / `4-cnhubert` / `5-wav32k`。**`exp_dir` 必须是数据集目录本身**，不能另开子目录 | 照上游 `webui.py:535`：`data.exp_dir = s2_ckpt_dir = 数据集目录` |
| s1 续训 `UnpicklingError: Unsupported global: pathlib.WindowsPath` | torch ≥ 2.6 把 `torch.load` 的 `weights_only` 默认改为 `True`，而旧 checkpoint 里存了 `WindowsPath` | 训练时设 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`（PyTorch 官方开关，恢复旧行为） |
| `No module named 'tensorboard'` / `'absl'` | venv 缺训练期依赖 | `uv pip install --target train/libs --no-deps ...`，再用 `PYTHONPATH` 借进来（见 `train/README.txt`） |
| `ImportError: cannot import name 'setLangfilters'` | PyPI 上的 `LangSegment` 与 GPT-SoVITS 要的不是同一个项目 | **不要装**。仓库自带 `GPT_SoVITS/text/LangSegmenter`，训练链路也不需要它 |
| UVR5 分离报 `Format not recognised` | `soundfile`/libsndfile 不支持 m4a/AAC | 先用 `video2audio.py extract` 转成 WAV 再分离 |

### 显存与 batch size（RTX 5060 Laptop 8 GB 实测）

| 阶段 | 上游默认 | 本机可用 ~6 GB 时用 | 实测 |
|---|---|---|---|
| s1（GPT） | 8 | **4** | 15 轮 / 33 批次 ≈ 1.5 分钟，7.6 it/s |
| s2（SoVITS LoRA） | 32 | **3** | rank=32 |

### 上游备选界面
训练仓库自带官方 Gradio 界面，需要手动点击时用：
```bash
cd _src/GPT-SoVITS-main && python webui.py
```

---

## UVR5 人声分离

独立环境：`.venv-uvr`（Python 3.12 + audio-separator 0.47.1 + CPU torch）
模型目录：`uvr5/weights/`

```bash
$uvr = ".venv-uvr/Scripts/python.exe"   # bash 下同理换路径

$uvr tools/uvr5.py models                                   # 看已装模型
$uvr tools/uvr5.py separate -i 歌.wav                        # 分离人声/伴奏
$uvr tools/uvr5.py dereverb -i 人声.wav                      # 去混响
$uvr tools/uvr5.py make-dataset -i 歌.flac -o data/raw/某人   # 分离+去混响+切分
```

已装模型：

| 文件 | 用途 |
|---|---|
| `Kim_Vocal_2.onnx` | 人声分离（默认） |
| `UVR-MDX-NET-Voc_FT.onnx` | 人声分离（备选） |
| `UVR-MDX-NET-Inst_HQ_3.onnx` | 伴奏提取 |
| `Reverb_HQ_By_FoxJoy.onnx` | 去混响（输出 `(No Reverb)` = 干声） |

**网络受限环境的绕过**：audio-separator 会去 `raw.githubusercontent.com` 拉模型元数据。
把 3 个 JSON 预取到权重目录即可，存在就跳过下载：
`download_checks.json` / `vr_model_data.json` / `mdx_model_data.json`。

实测性能（CPU）：30 秒音频约 28 秒（含模型加载）。GPU 需另装 CUDA 版 torch。

### 分离质量实测（30 秒合成混音，Kim_Vocal_2）

| 指标 | 数值 |
|---|---|
| 人声保留 | 93.93% |
| 人声轨中伴奏残留 | 12.81%（SIR 17.85 dB） |
| 伴奏轨中人声残留 | 12.34%（SIR 18.18 dB） |
| ASR 回环（人声轨） | 逐字正确 |

**结论：分离是"有损"的，约 18 dB 分离度——每 8 份人声里仍混着 1 份伴奏。**
这个残留会被训练学进去，所以去混响和片段筛选不可省略。

---

## 支持语言

```
中文 / 英文 / 日文 / 粤语 / 韩文
中英混合 / 日英混合 / 粤英混合 / 韩英混合
多语种混合 / 多语种混合(粤语)
```

---

## 磁盘占用（首次部署需下载的部分）

| 部分 | 体积 | 获取方式 |
|---|---|---|
| `s1v3.ckpt` | 148 MB | `tools/fetch_models.py` |
| `s2Gv3.pth` | 733 MB | `tools/fetch_models.py` |
| G2PW 中文多音字模型 | 608 MB | `tools/fetch_models.py` |
| BERT + CNHuBERT + BigVGAN | ~1.4 GB | `tools/fetch_models.py` |
| fastText 语种检测 `lid.176.bin` | 125 MB | `tools/fetch_models.py`（MD5 校验） |
| Qwen3-ASR-0.6B | 1.8 GB | `tools/fetch_qwen_asr.py` |
| UVR5 权重（4 个 onnx） | ~120 MB | 首次使用自动下载 |
| 上游仓库 `_src/` | 32 MB + 权重 | git clone |

---

## 延伸阅读

- [**踩坑实录：8GB 笔记本上把声音克隆跑通**](docs/踩坑实录-8GB声音克隆.md) ——
  10 个坑的现象 / 根因 / 修法，含 int16 削顶与语义坍缩这两个"不报错但产物是垃圾"的陷阱。

## 许可与伦理

- **代码** MIT（见 `LICENSE`）—— 仅覆盖本项目的 `tools/` 脚本、配置模板与文档。
- **模型权重与音频一律不入库**，由你自行下载，各自遵守原始许可（见 `NOTICE`）：
  GPT-SoVITS 为 MIT；Qwen3-ASR、UVR5 权重请以其上游仓库声明为准。
- **克隆声音前请读 `docs/音色伦理.md`**：只处理你自己的声音，或已取得明确知情同意的声音。
  参考音频、数据集、训练出的音色与 `config.json` 全部被 `.gitignore` 排除，
  既为隐私，也为许可。

---

## 已知限制

- 训练需要自备目标人音频；没有数据集时只能走零样本
- 零样本对参考音频质量敏感：必须是单人、干声、无混响
- 跨语言参考会**概率性**触发 token 坍缩（守卫会拦，但会浪费一次生成）
- `train_pipeline.py` 的训练阶段参数是按上游 `webui.py` 逻辑复刻的，
  首次实际训练前建议先跑 `prepare` 验证数据链路，训练超参按显存自行调整
  （8 GB 显存建议 `--batch-s2 2~4`）
- 合成质量请用 `tools/asr_check.py` 回环校验，不要只信"跑通了"。
  它按**无声调拼音**比对（字形编辑距离会把「解耦→截偶」这类同音误识别
  错算成合成失败：字形相似度 0.778，拼音 1.000），阈值 0.85，退出码可直接卡流水线。
