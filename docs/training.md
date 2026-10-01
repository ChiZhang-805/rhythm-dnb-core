# 服务器训练

GPU 以服务器实际可见设备为准，不写死型号、数量或显存。先按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/) 安装匹配服务器驱动的 CUDA 运行环境，再安装本项目。训练和推理都从本地文件读取，不自动联网下载模型。

## 训练前

1. 在服务器执行 `python -m rhythm_dnb hardware`，确认设备名称、显存和 CUDA 是否可用。
2. 准备按人隔离、审核通过的真实语料，格式见 [数据接入](data.md)。
3. 准备 `configs/text/macbert.json` 指定上游提交的 MacBERT 编码器、分词器文件，以及 `download.json` 下载凭据。凭据包含 `model_id, revision, files`，其中 `files` 是“相对文件名 → SHA-256”；必须覆盖除凭据自身外的全部文件。程序校验清单和文件哈希，不能只填模型名称。
4. 选择一个尚不存在的输出目录。数据和权重目录可放在服务器共享磁盘。

## 启动

以下为 Linux shell 命令，路径替换成服务器真实路径。

```sh
# 单卡：只让程序看到指定 GPU
CUDA_VISIBLE_DEVICES=0 python -m rhythm_dnb train-text \
  --corpus /data/corpus.json --base /models/macbert \
  --config configs/text/macbert.json --output-dir /runs/text-training

# 同一台服务器的两张卡：每卡一个进程
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  --module rhythm_dnb train-text \
  --corpus /data/corpus.json --base /models/macbert \
  --config configs/text/macbert.json --output-dir /runs/text-training-multigpu
```

多卡采用 Linux/WSL 的 NCCL；Windows 支持单设备训练。此处准备的是单服务器多 GPU 流程，未提供跨服务器调度。

## 资源参数

| 配置 | 当前起点与调整方法 |
| --- | --- |
| `device` | `auto` 自动选择 CUDA，否则 CPU；正式 GPU 任务可设 `cuda`，无 GPU 时直接报错。 |
| `precision` | `auto` 在所有参与卡支持时用 BF16，否则 GPU 用 FP16，CPU 用 FP32；也可手动指定。 |
| `batch_size` | 每卡每次 2 条，给显存较小设备留出空间；在服务器测量峰值显存后调整。 |
| `gradient_accumulation` | 累积 8 次再更新。有效批量 = 每卡批量 × 累积次数 × GPU 数量。 |
| `gradient_checkpointing` | 默认开启，用额外计算降低激活值占用。 |
| `num_workers` | 默认 0；CPU 和内存充足时再增加，避免每张卡重复启动大量加载进程。 |

默认单卡有效批量为 16。若改为两卡并希望仍为 16，可将累积次数设为 4；不能只增加卡数却忽略训练设置变化。显存不足时先减小每卡批量，再调整累积次数。最大文本长度 512，超长文本报错，不静默截断。每卡都保存完整模型，多卡显存不直接相加；混合型号的速度和可用批量受较慢、较小显存设备限制。

`execution.json` 记录实际设备、精度和有效批量；`history.json` 记录损失、实际样本数、更新次数和溢出跳过次数；`result.json` 保存选定模型及最终测试结果。验证集负责选模型，测试集只在最后评价。检查点是已选模型权重，不包含完整优化器续训状态。

训练过程按真实样本数加权，不用重复样本填满最后一批；只有主进程写模型。CPU 双进程检查验证梯度合并；有 GPU 的检查还会实际训练并重新加载小模型。这些是软件验证，服务器多 GPU 性能和真实语料效果仍需在目标环境运行后报告。

实现依据：[PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) 和 [混合精度与梯度累积](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)。
