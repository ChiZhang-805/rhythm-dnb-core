# Rhythm DNB Core · 节律 DNB 科研核心

独立 Python 科研后端，用于研究**行为节律失稳的提前预警**。当前版本 `0.2.0`。实现文本连续程度量化、时间和传感器测量、群体 DNB 模块发现、固定参考 sDNB、个人滚动 DNB、独立结局、阈值校准与冻结验证。

这是一个待真实数据验证的研究方案。DNB 分数不是概率；只有数据合格且策略已校准时才输出 `warning=0/1`，否则为 `null` 并给出原因。不能将模拟测试结果当成真实人群准确率或临床诊断。

## 克隆与安装

Python ≥ 3.11；发布检查使用 Python 3.13。仓库包含源码、合成测试夹具、配置和科研文档，不需要 NAVA APP 或其他私有仓库。

```sh
git clone https://github.com/ChiZhang-805/rhythm-dnb-core.git
cd rhythm-dnb-core
python -m venv .venv
```

Windows PowerShell 激活：`.venv\Scripts\Activate.ps1`；Linux/macOS：`source .venv/bin/activate`。

```sh
python -m pip install --upgrade pip
python -m pip install -e ".[research,text,io,plots]"
python -m rhythm_dnb --help
python -m unittest discover -s tests -q
```

只需要冻结数值推理时安装 `python -m pip install -e .`，仅依赖 NumPy 和时区库。完整测试包含一次本地微型 BERT 的训练、保存、加载，需安装上述可选依赖；测试不下载预训练模型。GPU 不是运行测试的必要条件。

## 先跑合成示例

```sh
python -m rhythm_dnb simulate --seeds 30 --output runs/mechanism.json
python -m rhythm_dnb demo --panel objective8 --output-dir runs/demo-01
```

`demo` 通过生产流程拟合、发现和校准，保存 bundle、报告和请求示例。默认合成开发样本较少，可能得到 `no_stable_modules`；这是保留统计门槛的有效结果。程序仅在存在可用校准策略时计算独立测试指标。每次演示使用新的输出目录。

生成网络热力图、分量柱状图、负相关散点图和参数敏感性图：

```sh
python tools/review_research.py --output-dir runs/review-01
```

默认只使用合成数据。检查自己的历史数据库时可显式提供 `--database /path/to/rhythm.sqlite`，以及可选的 `--text-database /path/to/master.sqlite`。数据库只读；数据分布和来源覆盖图仅在提供相应输入时生成。这些参数针对项目历史表结构，任意数据库请先使用数据适配器。

## 目录

```text
src/rhythm_dnb/
  contracts.py, config.py, timebase.py, provenance.py
  measures/     睡眠、饮食、活动、生理测量与固定尺度
  text/         语料审查、分类回归头、训练及冻结推理
  dnb/          经典 DNB、sDNB、参考与候选模块
  outcomes/     独立结局阈值、事件确认、未来标签
  warning/      因果窗口、评分、连续报警与冷却
  research/     发现、校准、评估、对照、模拟、科学图
  workflows/    数据准备、研究拟合、终点、锁定验证
  io/           输入校验、来源适配、存储与分区
  api.py, bundles.py, cli.py
configs/        v2 研究配置、面板、文本训练设置
tests/          数值性质、边界、合成全流程和文本训练测试
tools/          可复现的逐文件扫描与科学诊断
examples/       同一 API/CLI 的调用示例
docs/core/      协议、参数依据、逐文件职责、接口和验证记录
```

详细说明：

- [研究协议与论文到代码的对应](docs/core/research_protocol.md)
- [全面复核与科学限制](docs/core/review_20261001.md)
- [可调参数及依据](docs/core/parameters_v2.csv)
- [每个代码文件的职责](docs/core/file_map.md)
- [API、输入格式与工作流](docs/core/api_contract.md)
- [实现与注释规范](docs/core/implementation.md)
- [独立仓库验证记录](docs/core/verification_v2.md)
- [开发协作](CONTRIBUTING.md) · [数据与模型接入](docs/core/data_access.md)

## 使用真实数据

按 [数据契约](docs/core/api_contract.md) 准备同一人的纵向观测、来源证据、实际可得时间、独立终点及按人隔离的研究分区。数据、模型权重和拟合 bundle 不随源码发布；当前没有随库提供经过真实前瞻性验证的预警模型。

`develop` 使用参考、开发和校准数据生成冻结 bundle；`validate` 使用独立测试个体、事件登记与监测日历验证；`score` 只加载冻结产物进行个体评分。完整参数见各命令 `--help`。

所有默认研究参数都是可检验的初始方案，不是已证明最优值。量化、bundle 和核心存储采用 v2；历史 v1 产物必须从来源重建并重新校准。
