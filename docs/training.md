# 服务器训练

唯一主模型是 `Qwen/Qwen3-8B`，固定提交见 `configs/text/qwen.json`。采用 NF4 冻结底座、LoRA 微调及独立的程度/证据评分头；不让模型自由生成数字。MacBERT-base 配置仅供对照，两者使用同一真实语料和划分。

## 准备和启动

先按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/) 安装服务器驱动适配的 CUDA 环境，再安装 `.[research,text,quantized,io,plots]`。执行 `python -m rhythm_dnb hardware` 查看实际设备。以下为 Linux 命令，路径替换为服务器路径，输出目录必须尚不存在。

```sh
# 显式下载固定提交的权重、分词器，并生成哈希凭据
python -m rhythm_dnb download-text --config configs/text/qwen.json \
  --output-dir /models/qwen --cache-dir /cache/huggingface

# 单卡；训练、校准、测试均读取本地文件
CUDA_VISIBLE_DEVICES=0 python -m rhythm_dnb train-text \
  --corpus /data/corpus.json --base /models/qwen \
  --config configs/text/qwen.json --output-dir /runs/text-training

# 同一台服务器两卡：每卡一个进程
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  --module rhythm_dnb train-text \
  --corpus /data/corpus.json --base /models/qwen \
  --config configs/text/qwen.json --output-dir /runs/text-multigpu
```

多卡使用 Linux/WSL NCCL；Windows 支持单卡。每张卡都装入底座，多卡显存不会合并。建议优先在 24 GB 卡上部署；8 GB 卡是否能运行须实测，不承诺仅靠 NF4 就能容纳全部状态。NF4 依赖 bitsandbytes 和 CUDA；CPU 实验需显式改为 `quantization=none` 并准备足够内存。

## 调参和产物

主模型起点：每卡批量 1、累积 16 次、最大 512 token、梯度检查点开启。有效批量为「每卡批量 × 累积次数 × 卡数」；两卡若保持 16，应把累积次数改为 8。超长文本报错，不静默截断。自动精度在所有参与卡支持时使用 BF16，否则 GPU 使用 FP16；溢出跳过更新会记录，整轮无有效更新则失败。

多卡等待主进程验证和保存时，`process_timeout_minutes` 默认 120 分钟；超大验证集或慢共享磁盘需按实际耗时调整。运行记录同时保存 PyTorch、Transformers、PEFT 等库版本。

参数的理由和调整范围见 [参数表](parameters.csv)。它们是待验证起点，不是已经寻优的最优值。先在验证集比较学习率、LoRA 秩、长度和损失权重；校准集只确定每个指标的证据接收门槛，测试集只在选定模型后使用一次。证据目标精度 0.95 是经验接收目标，不是总体保证；每类至少 20 条只是运行门槛。

`execution.json` 记录设备和精度；`history.json` 记录每轮损失及更新；`model/` 保存适配器、评分头、分词器和文件哈希；`result.json` 与 `test-predictions.json` 保存测试结果（后者不含原文）。Qwen 推理必须同时提供原底座与适配器，不能随便换一个 `.pt/.pth` 文件。检查点不含完整优化器续训状态。

```sh
python -m rhythm_dnb score-text --checkpoint /runs/text-training/model \
  --base /models/qwen --category stress --text '实际待分析文本' --output /runs/score.json
python -m rhythm_dnb plot-text --result /runs/text-training/result.json \
  --predictions /runs/text-training/test-predictions.json --output-dir /runs/text-figures
```

图表显示各项误差、证据覆盖、评分波动幅度、误差相关和人工/模型散点。需要 `observed_at` 的同人重复标注才能评价个人变化；横断面成绩不能替代这一项。小模型测试只验证软件，真实训练和服务器多卡性能须另行报告。

实现依据：[Qwen3 模型卡](https://huggingface.co/Qwen/Qwen3-8B)、[QLoRA](https://arxiv.org/abs/2305.14314)、[PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)。
