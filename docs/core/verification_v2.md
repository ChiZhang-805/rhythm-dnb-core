# 独立仓库发布验证

验证日期：2026-10-01。版本：`rhythm-dnb-core 0.2.0`。

## 本地结果

| 检查 | 结果 |
| --- | --- |
| 核心源码 | 70 个 Python 文件与提取前核验版本逐字节一致 |
| 独立安装 | 构建 wheel 后安装于单独环境，运行时从其 site-packages 导入 |
| 旧工程依赖 | `rhythm_pipeline` 与 `text_distillation` 均不可导入，核心测试仍通过 |
| 全套核心测试 | 85 项通过，35.607 秒；包含合成完整研究及本地微型 BERT 实际训练/加载 |
| 静态诊断扫描 | 88 个 Python/JSON 文件、7,013 行；核心函数均有 PSEUDOCODE，未见 NotImplemented 占位 |
| CLI | `demo` 与 `simulate --seeds 2` 完成，示例产物明确标注 simulation_only |
| 图表 | 生成网络相关热力图、DNB 分量柱状图、负相关散点图、参数敏感性图，PNG/SVG 可导出 |
| 数据访问 | 默认诊断不读取任何外部数据库；真实数据库只能由用户显式指定 |

本地环境为 Windows / Python 3.13.0。发布隔离环境复用已安装的科学依赖目录；核心包自身来自新构建 wheel，未借用旧源码。关键依赖版本见 `requirements-core.lock`。此文件不是跨平台完整锁文件。

wheel 的 SHA256：`2bd1122352a033db256a71517e8a5ba34b6b77654da88dfb88adf72976aba7dc`。该哈希对应本地验证构建，不要求不同机器重新构建产生相同 ZIP 字节。安装包中全部 70 个源文件已与工作树核对。

## 统计失败结果也保留

合成模块发现检查使用同一配置，开发人数 40/80/200 时，最小最大统计量校正 p 值分别为 0.101/0.041/0.037，选中模块数为 0/1/1。它们说明程序能保留空结果，不构成人数推荐或真实效能证据。

默认 `demo` 此次报告 `insufficient_outcomes`，未生成可用预警阈值；程序没有为演示而降低校准门槛。完整研究的正向路径由另外的明确合成测试队列覆盖。研究数据不足或没有稳定模块时允许 `warning=null`。

## 协作者复现

```sh
python -m pip install -e ".[research,text,io,plots]"
python -m unittest discover -s tests -q
python tools/review_research.py --output-dir runs/release-review
```

诊断输出包含逐文件哈希、默认配置、三个 DNB 分量和合成敏感性结果。原始数据审计、用户路径和本机运行日志不随仓库发布。

GitHub Actions 配置为从 wheel 安装，在 Linux / Windows、Python 3.13 上检查完整测试与 CLI。每个提交的实际远端结果以仓库 Actions 页面为准；本地通过不自动代表所有平台均已验证。

所有验证都属于实现与合成机制验证。目前仓库没有附带合格的真实前瞻性研究数据，也没有真实人群预警有效性的结论。
