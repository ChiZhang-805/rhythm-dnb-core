# 数据接入

需要同一人的连续观测、真实到达时间、独立事件和足够随访。历史公开数据的诊断、名义日期或人工补全值，不能直接作为“即将失稳”的训练真值。

## 两类输入

| 数据 | 必须提供 |
| --- | --- |
| 文本语料 JSON 列表 | `example_id, participant_id, group_id, split, category, text, scores, origin, review_status, annotation_evidence_id`。 |
| 原始观测 | `observation_id, participant_id, variable, value, unit, start, end, available_at, timezone, provenance`。 |

主模型的 `split` 为 `train/validation/calibration/test`，分别负责拟合、选模型、校准证据门槛和最终评价。每组包含情绪、压力、饮食、睡眠、社交五类。分数按 [schema.py](../src/rhythm_dnb/text/schema.py) 填写全部字段：有依据时为 0–100，无法判断时为 `null`。明确“没有焦虑”可标 0；根本没提到且无法推断不能标 0。训练集每项都需有可评分和不可评分的真实例子，标注经过人工审核与分歧仲裁。

正式验证入口只接受真实原文或其翻译；自建文本继续通过 `*-text-experiment` 入口开展实验，明确保留作者和参考分来源。同一人、同源组和重复文本不能跨分区；文本拟合、选模和证据校准使用过的人，不能再进入 DNB 报警校准或最终测试。跨数据集沿用统一人员标识。`observed_at` 须带时区偏移，并用 `temporal_basis` 区分真实提交时间与自建文本对齐的源记录时间；后者不能证明真实个人波动。外部 0–3 等级标签不能伪装成精确的 0–100 程度标签。

`label_states` 明确区分有分数、有依据但未定分、明确无症状、信息不足、待标注和有分歧，字段值见 [训练说明](training.md)。旧数据中空白分数只表示未知，不自动作为“信息不足”的负例。双人盲审包隐藏旧答案；比较后保留分歧，不自动平均或改写源库。

观测的时间必须带偏移量，另附实际 IANA 时区。来源记录 `kind, source_id, source_hash, parent_ids, method, independent`；推导值保留上游记录。未知值用 `null`，不能猜测设备未记录的时间、佩戴覆盖或热量摄入。具体校验见 [validation.py](../src/rhythm_dnb/io/validation.py)。

## 工作流接口

| 命令 | 输入与输出 |
| --- | --- |
| `quantify` / `prepare` | 输入单人的 `participant_id, day, zone, issued_at, observations`，声明 `sleep_complete/eating_complete`；文本附 `text_checkpoint, text_base, text_device`。前者输出全部 25 项，后者选取固定的 12/8 项 DNB 面板；均保留单位、来源、覆盖和缺失原因。 |
| `fit-endpoint` / `endpoints` | 前者从稳定人群拟合界限；后者生成独立结局。字段由 [endpoints.py](../src/rhythm_dnb/workflows/endpoints.py) 定义。 |
| `label` | 输入 `--timeline`、`--request`、`--study`、`--followup-end` 和 `--as-of`，输出完整离线 case；检查人员、时区、事件规则和预测期限一致。 |
| `develop` | 输入参考记录、稳定/事件前配对、校准请求、三个拟合截止时间、校准事件与监测日历；联合面板还需 `text_checkpoint`，核验人员隔离。输出冻结 bundle。字段见 [develop.py](../src/rhythm_dnb/workflows/develop.py)。 |
| `validate` | 提供冻结 bundle、独立测试 cases、events、monitoring 和评价截止时间；输出性能与覆盖。 |
| `score` | 提供 bundle 与 `participant_id, issued_at, timezone, history` 请求；后续 `--state` 可直接传上次完整输出或其中的状态对象。 |

每条离线 case 将 `request` 和 `label, label_available_at, outcome_protocol_id` 分开。事件记录 `participant_id, onset, confirmed_at, definition`；监测日历记录 `participant_id, first_issue_day, last_issue_day, timezone`。具体字段以 [contracts.py](../src/rhythm_dnb/contracts.py) 为准，各命令参数用 `--help` 查看。

`label` 生成的 case 按事先划分的人员整理成校准/测试列表。时间线保存生成截止时间和完整协议；更改事件窗口、持续天数或预测期限后，须从原始观测重建，不能只改标签。用较晚快照生成较早截止时间的标签也会被拒绝。

研究日从当地 04:00 起，下一天 12:00 发布；跨日区间按实际时间切分。测量内容指纹随面板保存，外部输入不能省略。旧产物缺少指纹或指纹不同，应从来源重算，不能手工改标识。

文本输出中的 `scores` 才是通过证据门槛的分数；`estimates` 是未经接收检查的原始估计，不能代替缺失值送入 DNB。分数反映文字所表达的程度，不是情绪发生概率，也不是设备或临床测量值。

## 存储与图表

上传流程见 [服务器训练](training.md)。`tools/transfer_data.py` 将现有 SQLite 库作一致快照，与完整来源文件打包；服务端校验后放入项目 `data/`，不覆盖已有数据。`data/legacy/master.sqlite` 和 `rhythm.sqlite` 是保留原始标记的来源库，包含待审核或构造内容，不能直接作为主模型的合格训练输入。训练前仍须按当前语料与来源契约筛选、审核和导出。

`io/repository.py` 创建核心 SQLite 数据库，追加保存观测、每日指标、预测和审计记录；结局由独立工作流输出，库内保留结局表。已有不兼容数据库会拒绝写入，原始数据不覆盖、不自动迁移。

`io/sources/` 保留不同真实数据集的读取器。多数历史读取器输出回顾性记录；补齐并验证真实时间及来源后，才可通过规范化入口接入预警流程。

`tools/audit_project.py --database /绝对路径/历史库.sqlite --output-dir /绝对路径/检查结果` 可只读检查历史库，并输出真实字段分布和来源覆盖图。该参数面向原项目的历史表结构；核心库通过 repository 接口读取。网络相关热力图、分量柱状图和散点图由 `research/plots.py` 接收实际测量矩阵生成，不自造数据。

扩大实验前运行 `tools/audit_expansion.py --database /来源库.sqlite --text-development /文本开发集.json --output /新检查目录`。它逐条清点来源、连续日期、原始测量覆盖、结局缺项和文本开发用过的人，输出 CSV 与哈希清单。日期足够只说明可能形成窗口，不代表已有正确答案；旧库中填满的数值也不等于全部实测。先按用途利用全部记录，再以有独立结果的部分评估预警，不能把训练记录直接并入测试正确率。
