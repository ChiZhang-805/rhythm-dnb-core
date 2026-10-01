# 数据与模型接入

本仓库只发布实现、配置、文档和合成测试夹具。原始数据、参与者文本、历史 SQLite、训练权重以及包含参考个体矩阵的 bundle 均由研究环境管理，不存入 Git。

## 数据入口

| 输入 | 入口 | 使用条件 |
| --- | --- | --- |
| 新采集的结构化观测 | `io/sources/hospital.py`、`workflows/prepare.py` | 显式字段映射、单位、时区、测量起止、可得时间及来源哈希 |
| PMData、LifeSnaps、Sleep Diary、REST、DELSOM、Manchester 历史源 | `io/sources/` 对应模块 | 从原始项目取得相应文件；遵守数据访问条件，保留发布哈希和原有时间语义 |
| 文本监督语料 | `text/corpus.py`、`train-text` | 分类程度标签、人工复核证据、来源、按人和模板隔离的分区 |
| 终点评估输入 | `workflows/endpoints.py` | 独立稳定参考、完整基线及观测资格，冻结相同协议身份 |
| 冻结研究产物 | `develop`、`validate`、`score` | 不同阶段独立个体，有序截止时间，校准/测试事件登记与监测日历 |

来源解析器顶部的 URL、文件清单和哈希说明各自支持的版本。历史回顾性记录即使来源可追溯，也不自动具备真实前瞻性可得时间或 SEA-v2 事件标签。缺少同一人同步多域记录时保留缺失，不能拼接不同人的数据。

## 本地目录

可将输入和产物放在仓库之外，所有 CLI 路径都显式指定。若使用 `config.resolve_paths`，通过 `RHYTHM_DNB_DATA`、`RHYTHM_DNB_MODELS`、`RHYTHM_DNB_BUNDLES`、`RHYTHM_DNB_RUNS` 提供各自的绝对路径，或传入不提交 Git 的 `configs/paths.local.json`。这些设置不会触发自动下载。

项目原 Windows 工作区遵循 `Q:\NAVA-Workspace` 的所属目录规则；其他开发环境可使用自己的目录，不需要该盘符。

## 文本模型

`configs/text/macbert.json` 记录编码器 ID 和固定 revision。自行取得对应本地模型文件后，在模型目录保存 `download.json`：

```json
{
  "model_id": "hfl/chinese-macbert-base",
  "revision": "与研究配置相同的完整提交 ID",
  "files": {
    "config.json": "该文件的 SHA256",
    "model.safetensors": "该文件的 SHA256",
    "tokenizer_config.json": "该文件的 SHA256",
    "vocab.txt": "该文件的 SHA256"
  }
}
```

`files` 必须覆盖实际使用的配置、权重和 tokenizer 文件；如有其他 tokenizer 文件，一并列入。程序按收据校验本地内容，不凭模型名称信任权重。下载及数据来源应另外保留独立的取得记录，内容哈希本身不能证明来源真实性。

完整语料契约、训练和推理命令见 [API 契约](api_contract.md)。仓库不包含已完成真实效能验证的 checkpoint，开发者需要合格语料后才能开展相应研究。
