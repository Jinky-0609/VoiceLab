train/libs/ —— 训练期额外依赖（不入库）

这里的包是给 GPT-SoVITS 训练脚本用的（tensorboard / absl-py / protobuf / grpcio 等），
通过 train_pipeline.py 注入 PYTHONPATH 借给子进程，**不装进 .venv-tts**，
这样推理环境保持干净、装坏了也不会影响合成。

重建：

    uv pip install --target train/libs --no-deps \
      tensorboard absl-py protobuf grpcio markdown werkzeug six packaging json5 sacremoses

（没有 uv 时：python -m pip install --target train/libs --no-deps <同样的包名>）

train/ 下另有从上游拷来的 libs 副本时，只在训练阶段使用；推理完全不需要。
