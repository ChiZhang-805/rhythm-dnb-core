# 数据接入

需要同一人的连续观测、真实到达时间、独立事件和足够随访。历史公开数据的诊断、名义日期或人工补全值，不能直接作为“即将失稳”的训练真值。

## 两类输入

| 数据 | 必须提供 |
| --- | --- |
| 文本语料 JSON 列表 | `example_id, participant_id, group_id, split, category, text, scores, origin, review_status, annotation_evidence_id`。 |
| 原始观测 | `observation_id, participant_id, variable, value, unit, start, end, available_at, timezone, provenance`。 |

文本的 `split` 为 `train/validation/test`，每组都包含情绪、压力、饮食、睡眠、社交五类。分数按 [schema.py](../src/rhythm_dnb/text/schema.py) 对应类别填写全部输出，范围 0–100；必须人工审核通过。只接受真实原文或真实原文翻译；构造文本不进入主训练和测试。同一人、同源组和重复文本不能跨分区。

观测的时间必须带偏移量，另附实际 IANA 时区。来源记录 `kind, source_id, source_hash, parent_ids, method, independent`；推导值保留上游记录。未知值用 `null`，不能猜测设备未记录的时间、佩戴覆盖或热量摄入。具体校验见 [validation.py](../src/rhythm_dnb/io/validation.py)。

## 工作流接口

| 命令 | 输入与输出 |
| --- | --- |
| `prepare` | 输入单人的 `participant_id, day, zone, issued_at, observations`；另须声明 `sleep_complete/eating_complete`，可附 `text_checkpoint`。输出每日面板，保留来源、覆盖和缺失原因。 |
| `fit-endpoint` / `endpoints` | 前者从稳定人群拟合界限；后者生成独立结局。字段由 [endpoints.py](../src/rhythm_dnb/workflows/endpoints.py) 定义。 |
| `develop` | 输入参考记录、稳定/事件前配对、校准请求、三个拟合截止时间、校准事件与监测日历；输出冻结 bundle。完整参数见 [develop.py](../src/rhythm_dnb/workflows/develop.py)。 |
| `validate` | 提供冻结 bundle、独立测试 cases、events、monitoring 和评价截止时间；输出性能与覆盖。 |
| `score` | 提供 bundle 与 `participant_id, issued_at, timezone, history` 请求；后续调用传回上次状态，输出预警和新状态。 |

每条离线 case 将 `request` 和 `label, label_available_at, outcome_protocol_id` 分开。事件记录 `participant_id, onset, confirmed_at, definition`；监测日历记录 `participant_id, first_issue_day, last_issue_day, timezone`。具体字段以 [contracts.py](../src/rhythm_dnb/contracts.py) 为准，各命令参数用 `--help` 查看。

研究日从当地 04:00 起，下一天 12:00 发布；跨日区间按实际时间切分。测量内容指纹随面板保存，外部输入不能省略。旧产物缺少指纹或指纹不同，应从来源重算，不能手工改标识。

## 存储与图表

`io/repository.py` 创建核心 SQLite 数据库，追加保存观测、每日指标、预测和审计记录；结局由独立工作流输出，库内保留结局表。已有不兼容数据库会拒绝写入，原始数据不覆盖、不自动迁移。

`io/sources/` 保留不同真实数据集的读取器。多数历史读取器输出回顾性记录；补齐并验证真实时间及来源后，才可通过规范化入口接入预警流程。

`tools/audit_project.py --database /绝对路径/历史库.sqlite --output-dir /绝对路径/检查结果` 可只读检查历史库，并输出真实字段分布和来源覆盖图。该参数面向原项目的历史表结构；核心库通过 repository 接口读取。网络相关热力图、分量柱状图和散点图由 `research/plots.py` 接收实际测量矩阵生成，不自造数据。
