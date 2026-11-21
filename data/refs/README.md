# data/refs/ —— 参考音频

零样本克隆需要「参考音频 + 该音频的逐字文本」，命名成对即可，例如：

```
data/refs/
├─ myvoice.wav        3~10 秒，单人、干声、无 BGM 无混响
└─ myvoice.txt        这段音频里说的每一个字（逐字，不要 paraphrase）
```

推荐参数：采样率 ≥ 24 kHz、单声道、峰值不过载、内容语速自然。
参考音频的**语言要与目标语言一致**——跨语言会概率性触发语义坍缩（守卫会拦，
但白跑一次生成）。

配好之后在 `config.json` 的 `profiles` 里注册：

```json
"myvoice": {
  "desc": "我的音色（零样本参考）",
  "gpt": "models/pretrained/s1v3.ckpt",
  "sovits": "models/pretrained/s2Gv3.pth",
  "ref_audio": "data/refs/myvoice.wav",
  "ref_text": "data/refs/myvoice.txt",
  "ref_lang": "中文"
}
```

路径相对项目根。用 `--ref-audio / --ref-text / --ref-lang` 也可以在命令行临时覆盖，
不必改配置。

> 本目录内容**不入库**：参考音频是声音身份数据，属于隐私。
> 只提交自己有权使用的音频，并对被克隆者取得明确同意。
