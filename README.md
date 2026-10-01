# Rhythm DNB Core · 节律预警科研核心

研究一个人是否即将出现**行为节律失稳**：将文本、睡眠、饮食和活动转为数值，发现共同波动的 DNB 群组，再用独立数据校准预警。项目独立于 NAVA APP。

合格数据输出 `warning=0/1`；数据不足或策略未校准时输出 `null` 和原因。DNB 分数不是概率，当前没有随库提供经过真实人群验证的预警模型。

文本主模型固定为 **Qwen3-8B**：本地权重 + LoRA 微调 + 程度/证据评分头。MacBERT-base 仅作可选对照。选择依据见研究方案；尚未证明哪个模型在本项目的真实人群上效果最好。

## 安装

需要 Python ≥ 3.11。服务器先按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装与驱动兼容的 GPU 运行环境，再安装本项目。

```sh
git clone https://github.com/ChiZhang-805/rhythm-dnb-core.git
cd rhythm-dnb-core
python -m pip install -e ".[research,text,quantized,io,plots]"
python -m rhythm_dnb hardware
python -m rhythm_dnb --help
```

仅使用已经冻结的数值预警模型时，安装 `python -m pip install .` 即可。真实数据、模型权重和训练产物由研究者另行提供，不上传到源码仓库。

## 使用顺序

1. 按人划分数据，核对来源、时间、标注和随访。
2. `check-text` 核对输入，再用人工审核的真实文本训练程度模型；`quantify` 输出 8 项客观指标和 17 项文本指标，无依据则保留缺失。
3. 独立定义失稳事件，拟合稳定参考，发现 DNB 群组，校准报警策略。
4. 冻结模型，用未参与开发的人群测试；通过后才接入实际预警。各阶段入口和服务器命令见 [训练与使用](docs/training.md)。

入口是 `src/rhythm_dnb/workflows/`；数学计算在 `dnb/`，量化在 `measures/` 和 `text/`，输入输出在 `io/`。配置集中在 `configs/`。

本机正式源码位于 `Q:/NAVA-Workspace/rhythm-dnb-core`；`models/` 放本地底座，`runs/` 放运行结果，二者不进 Git。共享下载缓存放在 `Q:/NAVA-Workspace/Caches/`，旧材料保留在归档或原交付目录，不作为当前源码。服务器路径通过命令行显式指定。

## 只保留四份说明

- [研究方案](docs/research.md)：预测目标、DNB 计算和验证方法。
- [数据接入](docs/data.md)：真实数据需要提供什么。
- [服务器训练](docs/training.md)：GPU、启动命令和运行记录。
- [参数表](docs/parameters.csv)：默认值、依据和调整建议。

软件检查：`python -m unittest discover -s tests -q`；逐文件扫描：`python tools/audit_project.py --output-dir runs/code-audit`。测试中的人工小数据仅用于核对程序，不是训练语料，也不产生科研结论。

发布包用 `python tools/build_wheel.py --output-dir dist`（目录须尚不存在）；从新建的源码副本构建，并逐一核对包内代码，防止旧 `build/` 文件混入。新增源文件先加入 Git 跟踪。

注释约定：模块开头一句说明职责；函数内用一条 `PSEUDOCODE` 注释交代计算顺序，额外注释只解释容易误解的依据。修改沿用原文件，计算规则的兼容性由内容指纹自动检查。
