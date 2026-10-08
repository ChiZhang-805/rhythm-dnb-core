# 已训练模型怎么用

当前保留 `rhythm-text-expanded`：按修正后的来源，在 3,390 条既有评估文本上，四类指标的平均绝对误差约 **4.97 分**（自建实验参考评分）。复杂人物、转述和反话仍可能判断错误；这不是经过真实人群验证的准确率。

后续训练改善了部分复杂语境，但未全面胜出，因此程度主模型不变。另附 `rhythm-text-evidence-policy` 判断每项是否有依据；它仍是实验附件，不能保证分数正确。比较见 [修缮结果](research.md#修缮结果与验证边界)。

## 文件放哪里

进入服务器上的项目目录，保持下面的结构。模型文件夹要完整保留，不能只拿出一个权重文件。

```text
rhythm-dnb-core/
├── models/qwen3-8b/               # 原始基模，约 16.4 GB
├── models/rhythm-text-expanded/   # 已训练的评分参数，约 190 MB
├── models/rhythm-text-evidence-policy/ # 实验依据判断器，可选
└── runs/                         # 保存输出结果
```

本机模型已保存于 `Q:/NAVA-Workspace/rhythm-dnb-core/models/`，当前 Runpod 为 `/workspace/rhythm-dnb-core/models/`，两端已核验文件哈希。权重不在 GitHub 源码仓库里。依据附件的新门槛只改清单，权重与已完成本地回传的附件相同。

## 输入一句话

使用已配置 CUDA/PyTorch 的 NVIDIA GPU 环境；本次在 A40 上验证。原来约 4 GB 内存的 CPU 服务器可以保存文件，不能直接运行这份 NF4 模型。

当前 Runpod 已配置好，直接执行：

```sh
cd /workspace/rhythm-dnb-core
bash runs/precision-delivery/score-text.sh emotion '今天和朋友见面很开心，心情比昨天放松。'
```

命令会打印结果文件路径；它使用已核验的源码和模型，程度推理固定为 FP32，GPU 忙时会排队。

自行部署到其他服务器时，在最新源码根目录和已激活的 Python 环境中执行（首次使用才需要安装）：

```sh
python -m pip install -e ".[text,quantized]" -c requirements-text.lock
python -m rhythm_dnb score-text-experiment \
  --checkpoint models/rhythm-text-expanded --base models/qwen3-8b \
  --evidence-checkpoint models/rhythm-text-evidence-policy \
  --category emotion --text '今天和朋友见面很开心，心情比昨天放松。' \
  --device cuda --precision fp32 --output runs/text-score.json
```

更换 `--text` 后的内容即可分析自己的中文文本；更换输出文件名可保留上一次结果。长文本应先整理为同一人、同一时段的描述，超过模型长度限制会报错。

`--precision fp32` 显式固定程度模型的计算精度；不改权重，也不改独立依据附件的原设置。重现旧实验可选 `auto`，也是未指定时的默认行为。输出的 `inference_profile` 和 `evidence_inference_profile` 分别记录两者的精度、环境及身份。研究中应固定设置，不把不同计算方式的分数直接混成个人时间序列；改变后须重新检查。

`--category` 可选：`emotion`（情绪）、`sleep`（睡眠）、`diet`（饮食）、`social`（社交）、`stress`（压力，辅助实验项）。

## 输出怎么看

打开 `runs/text-score.json`，查看 `estimates`：每项是 **0～100 分的原始程度估计，不是概率**。总体心情 `mood_valence` 越高表示越好；悲伤 `sadness_intensity`、焦虑 `anxiety_intensity` 越高表示越强，所以不能把所有高分都理解为好。

`evidence_estimates` 是依据判断器的实验输出；`experimental_acceptance` 表示是否超过当前研究门槛，**不保证程度分数正确**。没有添加附件时不输出这两项。

当前权重尚未完成独立依据校准，`acceptance_certified=false`，可用于下游的 `scores` 和 `normalized` 保留 `null`。查看实验分数可读 `estimates`，不能把 `null` 填成零或回退到原始估计后送入 DNB。

这一步完成“文本 → 数值”。输出中的 `eligible_for_primary_dnb: false` 表示它目前是实验文本模型，不能直接当作已经验证的节律紊乱 0/1 预警结果。

需要只运行程度主模型时，去掉 `--evidence-checkpoint` 一行即可；原始程度分数不变。保留的其他实验文件夹用于对照，不会自动替换当前模型。
