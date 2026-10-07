# 服务器训练

唯一主模型是 `Qwen/Qwen3-8B`，固定提交见 `configs/text/qwen.json`。采用 NF4 冻结底座、LoRA 微调及独立的程度/证据评分头；不让模型自由生成数字。MacBERT-base 配置仅供对照，两者使用同一真实语料和划分。

## 准备和启动

进入服务器上的 `rhythm-dnb-core` 目录后，先设置路径。每次打开新终端都执行这两行；换服务器只需先进入新的项目目录。不要沿用本机 `Q:` 路径，也不需要在系统根目录创建 `/data` 或 `/models`。

```sh
export DNB_ROOT="$(pwd -P)"
mkdir -p "$DNB_ROOT/models" "$DNB_ROOT/runs" "$DNB_ROOT/cache"
```

代码通过 GitHub 下载；数据库和原始数据单独通过 SFTP 上传。已准备的 `dnb-data.zip` 和同名 `.sha256` 放在项目根目录，执行：

```sh
sha256sum -c dnb-data.zip.sha256
python3 tools/transfer_data.py unpack --archive "$DNB_ROOT/dnb-data.zip" --project-root "$DNB_ROOT"
```

数据恢复到 `data/legacy/`、`data/sources/` 和 `data/public_library/`，文件哈希及数据库表记录数会复核。目标 `data/` 必须为空或不存在。打包使用 SQLite 一致快照；不要直接拖动正在写入的数据库。源库中的历史路径仅用于追溯，读取器使用本机显式传入的路径。底座整个文件夹另传到 `models/qwen3-8b/`。

确认 GPU 后，按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/) 安装服务器驱动适配的运行环境，再安装 `.[research,text,quantized,io,plots]`。执行 `python -m rhythm_dnb hardware` 检查设备。下列 `corpus.json` 指已按当前契约审核、导出的语料，不是把 SQLite 改后缀；训练输出目录必须尚不存在。

```sh
# 显式下载固定提交的权重、分词器，并生成哈希凭据
python -m rhythm_dnb download-text --config configs/text/qwen.json \
  --output-dir "$DNB_ROOT/models/qwen3-8b" --cache-dir "$DNB_ROOT/cache/huggingface"

# 检查语料划分、权重哈希和实际 token 长度，不加载神经网络
python -m rhythm_dnb check-text --corpus "$DNB_ROOT/data/corpus.json" --base "$DNB_ROOT/models/qwen3-8b" \
  --config configs/text/qwen.json --output "$DNB_ROOT/runs/text-input-check.json"

# 单卡；训练、校准、测试均读取本地文件
CUDA_VISIBLE_DEVICES=0 python -m rhythm_dnb train-text \
  --corpus "$DNB_ROOT/data/corpus.json" --base "$DNB_ROOT/models/qwen3-8b" \
  --config configs/text/qwen.json --output-dir "$DNB_ROOT/runs/text-training"

# 同一台服务器两卡：每卡一个进程
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  --module rhythm_dnb train-text \
  --corpus "$DNB_ROOT/data/corpus.json" --base "$DNB_ROOT/models/qwen3-8b" \
  --config configs/text/qwen.json --output-dir "$DNB_ROOT/runs/text-multigpu"
```

多卡使用 Linux/WSL NCCL；Windows 支持单卡。每张卡都装入底座，多卡显存不会合并。建议优先在 24 GB 卡上部署；8 GB 卡是否能运行须实测，不承诺仅靠 NF4 就能容纳全部状态。NF4 依赖 bitsandbytes 和 CUDA；CPU 实验需显式改为 `quantization=none` 并准备足够内存。

## 调参和产物

主模型起点：每卡批量 1、累积 16 次、最大 512 token、梯度检查点开启。有效批量为「每卡批量 × 累积次数 × 卡数」；两卡若保持 16，应把累积次数改为 8。超长文本报错并给出记录 ID，不静默截断。`check-text` 通过仅说明输入检查通过，显存能否承受反向传播仍须在服务器实测。自动精度在所有参与卡支持时使用 BF16，否则 GPU 使用 FP16；溢出跳过更新会记录，整轮无有效更新则失败。

多卡等待主进程验证和保存时，`process_timeout_minutes` 默认 120 分钟；超大验证集或慢共享磁盘需按实际耗时调整。运行记录同时保存 PyTorch、Transformers、PEFT 等库版本。

参数的理由、依据类别和调整范围见 [参数表](parameters.csv)。QLoRA 原 7B 实验采用秩 64、alpha 16、学习率 2e-4；本项目是另一种回归任务，不能照搬并称为最优。目前秩 16 和学习率 1e-4 是资源较保守的待验证起点。

调参时给 `check-text` 和 `train-text` 加 `--development-only`，输入文件只能含训练集和验证集；程序拒收校准、测试行。比较学习率、秩、长度和损失权重后锁定配置，再执行一次完整训练。不要反复运行完整流程挑测试成绩。

每项证据门槛先在验证组选定，再在独立校准组检验，失败不重新搜索门槛。每人每项按记录身份的固定哈希选一条，避免重复文本虚增人数。接收条件是精度的单侧 [Clopper–Pearson 下界](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html)达到 0.95，17 项 Bonferroni 分配总 alpha=0.05；全正确也至少需 114 位被接收者。0.95 是预登记使用目标，保证依赖参与者独立、校准样本代表实际输入且未被人为正负平衡等条件；不是临床有效性保证。证据不足输出 `null`。

`execution.json` 记录设备和精度；`history.json` 记录每轮损失及更新；`model/` 保存适配器、评分头、分词器和文件哈希；`result.json` 与 `test-predictions.json` 保存测试结果（后者不含原文）。Qwen 推理必须同时提供原底座与适配器，不能随便换一个 `.pt/.pth` 文件。

训练时加 `--save-resume-state` 可在每轮结束后保存优化器、学习率和随机状态。中断后使用相同数据、配置、代码和运行环境，加 `--resume-state 原运行目录/resume/epoch-N.json`，并指定新的输出目录，即可从最近完整的一轮继续。应备份整个运行目录；只保存推理模型不能无损续训，尚未完成的那一轮需要重做。

```sh
python -m rhythm_dnb score-text --checkpoint "$DNB_ROOT/runs/text-training/model" \
  --base "$DNB_ROOT/models/qwen3-8b" --category stress --text '实际待分析文本' --output "$DNB_ROOT/runs/score.json"
python -m rhythm_dnb plot-text --result "$DNB_ROOT/runs/text-training/result.json" \
  --predictions "$DNB_ROOT/runs/text-training/test-predictions.json" --output-dir "$DNB_ROOT/runs/text-figures"
```

图表显示各项误差、证据覆盖、评分波动幅度、误差相关和人工/模型散点。需要 `observed_at` 的同人重复标注才能评价个人变化；横断面成绩不能替代这一项。小模型测试只验证软件，真实训练和服务器多卡性能须另行报告。

下载支持分段续传，核对固定提交的官方文件大小、SHA-256/Git 摘要及全部权重分片，再生成 `download.json`；该文件存在且校验通过才代表底座完整。完整底座不等于已完成本项目微调。

## 冻结评分模型，另训依据判断

`extract-text-features` 缓存固定模型的句子表示，不修改基模、LoRA 或程度头。`fit-text-guard` 只读取训练与验证缓存，按各指标验证 log loss 选择逻辑回归正则强度；接收门槛按显式给定的验证精度目标选择。`evaluate-text-guard` 再读取独立存放的测试缓存；看过的场景必须加 `--regression-only`。缓存与判断器都绑定原模型哈希，不能换底座后混用。

本次正则候选为 `0.0001 0.001 0.01 0.1 1`，实验目标为 `--target-precision 0.95`；目标不是已经达到的总体保证。自编正负均衡语料上的表现不能代替正式独立校准。训练需要 `research` 依赖；提取表示使用 GPU，线性判断器拟合使用 CPU。

双人复核可用 `export-review-sheet --packet rater-A.json --output rater-A.xlsx` 导出空表，填完后用 `import-review-sheet --packet rater-A.json --workbook rater-A.xlsx --output completed-A.json` 导入。两人分别填写各自的 A/B 表，再交给 `compare-text-reviews`；工具检查身份声明和原文依据，不替人完成复核。

当冻结表示上的依据分类器无法区分复杂对话时，可用 `train-evidence-adapter --corpus development.json --checkpoint 原评分模型 --base 基模 --config 配置.json --output-dir 新训练目录` 单独学习上下文。它只接收显式依据标签，全部程度分数必须为空，17 项在训练和验证中均须有正负例；配置要求 `evidence_loss_weight=1`、`scope_loss_weight=0`、`train_experimental_evidence=true`。使用 `research,text,quantized` 依赖。

该流程按验证集各指标平均 log loss 选轮次，封存检查用 `evaluate-evidence-adapter`。接收门槛取验证集中相邻接收/拒绝概率的中点，在保持验证决策不变的前提下留出最大边界余量；不是统一设为 0.5，也不是独立校准。旧依据模型可用 `refine-evidence-thresholds` 对原验证语料重算该边界，权重不变，新文件另存。

独立依据模型只能通过 `score-text-experiment --evidence-checkpoint 依据模型` 与绑定的原评分模型配合，不能直接打程度分。原评分不变，未独立校准时仍不放行正式分数。覆盖报告会检查同一句话中“部分指标有依据、部分没有依据”的组合；“明确没有症状”算有依据。还须覆盖多项同时出现、同时缺席及强弱混合，不能只让每项单独凑齐正负例。新训练目录保留每次改善的检查点；此入口尚不支持优化器断点续训。

## 文本训练之后怎么走

自建模拟文本和模型参考评分使用独立的实验入口：`export-text-experiment` 从 `master.sqlite` 导出 `development.json` 与 `test.json`；`train-text-experiment` 只读取前者，用验证误差选轮次。锁定模型后才运行 `evaluate-text-experiment`；已查看的测试只能加 `--regression-only` 作开发检查。`score-text-experiment` 把原始 0–100 分放在 `estimates`，未校准的 `scores` 保留为空；实验权重不能默认接入正式 DNB。

标注用 `label_states` 区分 `supported`（有依据与分数）、`supported_unscored`（有依据但程度未定）、`explicit_absence`（症状明确不存在）、`insufficient_evidence`（信息不足）、`unreviewed`（待标注）和 `disputed`（有分歧）。只有第一、第三种填写数值，其余保持空值；“有依据但未定分”可训练证据判断，不编造程度。`train_experimental_evidence=true` 只从显式标注学习证据头，仍须独立校准才能接收分数。

`scope_targets` 按指标记录本人、时段、证据原文和应排除的片段；`scope_loss_weight>0` 启用逐 token 的辅助监督，只有标过的片段参与损失。它可与旧适配器一起续训并完整保存；辅助头不是已验证的解释器。用相同数据、预算和种子比较权重为零的对照，不预设某个权重最优。

修缮入口：`audit-text` 查全部 17 项覆盖；`prepare-text-review` 生成隐藏旧答案的双人标注包；`compare-text-reviews` 列出分歧而不自动平均；`refresh-text-source` 从最新只读库恢复来源时间和文本；`seal-text-experiment` 拒绝旧暴露记录进入新留出集。时间只是自编文本对应的源记录时间，不能冒充真实文本提交时间。

主配置启用 `require_all_validation_metrics=true`，训练或验证缺少压力等任一指标的参考就先补数据；局部实验可显式关闭并保留警告。`study-text` 接收完整配置组成的 `arms`、至少两个预登记 `seeds` 和 `rationale`，比较验证均值；不读取测试，也不自动替换主模型。复核分歧的容许差须明确提供，程序不暗设“差 10 分也算一致”。

扩充节律文字用 `expand-text-experiment --database 节律库 --legacy-development 旧development.json --legacy-test 旧test.json --checkpoint 旧模型 --output-dir 新语料目录`，可加 `--supplement 复杂场景.jsonl`。按人及来源分层留出新验证/测试，去除相同文本的冲突评分和数字替换模板；没有明确指标线索的参考分数保留未知，不自动造分。关键词筛选仅是保守的数据过滤，不能证明语义标注正确。原库不修改，排除原因、来源和过滤规则留在导出审计中。

大库中的“没有孤独感”等否定或正常状态描述存在冲突参考分数，相关目标也先屏蔽，不能自动改成猜测的零分。旧冻结导出可用 `review-text-experiment --corpus-dir 旧导出 --output-dir 新导出` 应用同一筛查，保留原评分和原因。经单独编写核对的复杂场景保留自己的参考标注；这些仍是实验语义参考，不是独立人工金标准。

新数据继续微调用 `train-text-experiment ... --initialize-from 旧模型 --save-resume-state`：继承适配器和评分头，重新建立优化器；`--resume-state` 则仅用于同一实验中断恢复。保留初始模型参与验证比较，续训变差时不强行替换。类别平衡通过配置 `balance_categories` 控制，使压力等小类仍能参与学习。新四类测试不用于声称压力能力提高。

模型读取完整文字；悲伤词并不直接决定分数。参考 [CheckList](https://aclanthology.org/2020.acl-main.442/) 和[对照样本测试](https://aclanthology.org/2020.findings-emnlp.117/)，分别检查否定、转折、当前/过去、自己/他人、反话和隐含表达。自编场景分数属于实验参考；训练场景与最终行为检查用不同文本，不把这些检查的通过率称作真实人群准确率。信息不足的反话不能仅靠几个词确定含义；[iSarcasmEval](https://aclanthology.org/2022.semeval-1.111/)专门区分作者意图与外部判断。

`text/behavior.py` 同时检查语义不变时的评分偏移：对照清楚的本人描述、加入干扰信息的原句和固定范围提示。保留绝对分数，避免高低排序正确掩盖明显误判。提示实验不修改权重；还要检查普通文本是否退步，通过小样本检查也不直接替换正式输入流程。

配对标注由 `text/annotations.py` 检查主体、时段、原文证据位置、排除原因、指标方向和等义句引用，并按整个场景隔离数据。格式通过不代表语义正确或已有人复核。纠偏试验可让等义句共享冻结模型对清楚陈述的参考值；明确无症状才使用量表零端点，其他分数保留模型来源。训练前保存回滚包，对照组保持相同步数和普通数据回放，分别报告新场景与旧错误的变化。

| 步骤 | 入口 | 做什么、保存什么 |
| --- | --- | --- |
| 训练文本模型 | `train-text` · `text/train.py` | 主要使用 GPU；从人工标注学习 17 项程度与证据，保存适配器、评分头及评价。 |
| 生成每日指标 | `quantify` / `prepare` · `workflows/prepare.py` | 用冻结文本模型和客观规则量化，再选出固定 DNB 面板；不是再次训练。 |
| 生成独立真值 | `fit-endpoint` → `endpoints` → `label` · `workflows/endpoints.py` | 先拟合稳定界限，再确认事件，最后给每次预测生成 0/1/未知的离线标签。 |
| 开发预警模型 | `develop` · `workflows/develop.py` | 主要使用 CPU；拟合参考人群、发现共同波动的指标群组、校准报警门槛，保存冻结 bundle。 |
| 最终测试与使用 | `validate` → `score` | 在独立人群评估，通过后逐日预测；保持模型与报警状态连续。 |

`compare-warnings` 接收各冻结方法的校准/测试逐日分数，要求完全相同的人、预测时点、标签和监测日历，在同一误报预算下单独校准门槛，再报告灵敏度、误报、提前量和覆盖率。输入分数必须由各方法按当时可见信息产生；该比较器不替调用方认证特征可用性。`plan-window-sensitivity` 只改变预测窗口，禁止同时改变结局定义；改变结局窗口属于另一个研究问题，不能按测试结果挑选。

标签来自独立睡眠、饮食和活动观测，不能用文本预测值或 DNB 分数反过来造标签。各阶段按人隔离，真实输入字段见 [数据接入](data.md)。批量量化时通过 Python 接口复用同一个 `TextPredictor`，避免逐日重新加载底座。

```sh
python -m rhythm_dnb develop --input "$DNB_ROOT/data/development.json" \
  --study configs/study.json --output-dir "$DNB_ROOT/runs/dnb-bundle" > "$DNB_ROOT/runs/development-receipt.json"
BUNDLE_PATH=$(python -c "import json,os; print(json.load(open(os.path.join(os.environ['DNB_ROOT'],'runs/development-receipt.json')))['bundle'])")
# EVALUATION_CUTOFF 是预先确定、带时区偏移的评价截止时间
python -m rhythm_dnb validate --bundle "$BUNDLE_PATH" \
  --cases "$DNB_ROOT/data/test-cases.json" --events "$DNB_ROOT/data/test-events.json" \
  --monitoring "$DNB_ROOT/data/test-monitoring.json" --as-of "$EVALUATION_CUTOFF" --output "$DNB_ROOT/runs/evaluation.json"
python -m rhythm_dnb score --bundle "$BUNDLE_PATH" \
  --request "$DNB_ROOT/data/today-request.json" --state "$DNB_ROOT/runs/previous-score.json" --output "$DNB_ROOT/runs/today-score.json"
```

首次 `score` 省略 `--state`；以后直接传上次完整输出。`validate` 默认沿用冻结配置的重采样次数，显式 `--bootstrap` 可覆盖并会记录。拿到服务器后依次核对系统/驱动、GPU 型号与显存、数据路径，再用实际训练记录确定批量和长度。

实现依据：[Qwen3 模型卡](https://huggingface.co/Qwen/Qwen3-8B)、[QLoRA 官方 7B 配置](https://github.com/artidoro/qlora/blob/main/scripts/finetune_guanaco_7b.sh)、[PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)。
