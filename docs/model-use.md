# 已训练模型怎么用

当前保留 `rhythm-text-expanded`：在 3,420 条普通测试文本上，四类指标的平均绝对误差约 **4.91 分**（自建实验参考评分）。复杂人物、转述和反话仍可能判断错误；这不是经过真实人群验证的准确率。

六次后续实验改善了部分复杂语境，但未全面胜出，因此主模型不变；完整比较见 [修缮结果](research.md#修缮结果与验证边界)。本地另存 `models/rhythm-text-context-candidate/` 供研究对照，不应直接当作升级替换。

## 文件放哪里

进入服务器上的项目目录，保持下面的结构。两个模型文件夹都要完整保留，不能只拿出一个权重文件。

```text
rhythm-dnb-core/
├── models/qwen3-8b/               # 原始基模，约 16.4 GB
├── models/rhythm-text-expanded/   # 已训练的评分参数，约 190 MB
└── runs/                         # 保存输出结果
```

本机模型已保存于 `Q:/NAVA-Workspace/rhythm-dnb-core/models/`。模型权重不在 GitHub 源码仓库里。

## 输入一句话

使用已配置 CUDA/PyTorch 的 NVIDIA GPU 环境；本次在 A40 上验证。原来约 4 GB 内存的 CPU 服务器可以保存文件，不能直接运行这份 NF4 模型。

在项目根目录、已激活的 Python 环境中执行（首次使用才需要安装）：

```sh
python -m pip install -e ".[text,quantized]" -c requirements-text.lock
python -m rhythm_dnb score-text-experiment \
  --checkpoint models/rhythm-text-expanded --base models/qwen3-8b \
  --category emotion --text '今天和朋友见面很开心，心情比昨天放松。' \
  --device cuda --output runs/text-score.json
```

更换 `--text` 后的内容即可分析自己的中文文本；更换输出文件名可保留上一次结果。长文本应先整理为同一人、同一时段的描述，超过模型长度限制会报错。

`--category` 可选：`emotion`（情绪）、`sleep`（睡眠）、`diet`（饮食）、`social`（社交）、`stress`（压力，辅助实验项）。

## 输出怎么看

打开 `runs/text-score.json`，查看 `estimates`：每项是 **0～100 分的原始程度估计，不是概率**。总体心情 `mood_valence` 越高表示越好；悲伤 `sadness_intensity`、焦虑 `anxiety_intensity` 越高表示越强，所以不能把所有高分都理解为好。

当前实验权重尚未校准“文本是否有足够依据”，所以可用于下游的 `scores` 和 `normalized` 保留 `null`，`reasons` 说明原因。旧版调用者应改读 `estimates` 查看实验分数，不能把 `null` 填成零或偷偷回退到原始估计。

这一步完成“文本 → 数值”。输出中的 `eligible_for_primary_dnb: false` 表示它目前是实验文本模型，不能直接当作已经验证的节律紊乱 0/1 预警结果。
