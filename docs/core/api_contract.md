# API 与数据契约

## 在线入口

```python
from rhythm_dnb.api import RhythmPredictor
from rhythm_dnb.contracts import parse_request, AlarmState

predictor = RhythmPredictor.from_bundle(bundle_path)
response = predictor.predict(parse_request(request_json), AlarmState())
# 保存 response 与 response.state，下一天传入上次 state。
```

`PredictionRequest` 只有 `participant_id, issued_at, timezone, history`。`issued_at` 必须为时区一致的本地正午。`history` 是 DailyPanel 列表：`participant_id, day, timezone, features`。不接受标签、未来事件或 split 字段。未来日的数据不会参与当前计算。

每个 FeatureValue：`name, value, unit, available_at, provenance, reason, coverage, version, measured_until, model_id`。日期/时间在 JSON 中使用 ISO8601，时刻必须带 UTC offset；`day` 为研究日起始日期。`measured_until` 必须落在该研究日范围，且不得晚于 `available_at`。缺失使用 JSON null 及原因；禁止以 0 占位。版本固定为 `"2"`。

单位由 `measures/panel.py` 固定：时间/时长 hour，活动量 count，心率 bpm，RA ratio，文本 score_0_100。文本特征携带冻结 checkpoint manifest 的 SHA256 `model_id`。

Provenance 字段：`kind, source_id, source_hash, parent_ids, method, independent`。kind 取 observed/derived/constructed/synthetic/unknown。observed/derived 还需要可追溯哈希和独立性才能进入真实主分析；根据数值改写的描述为 constructed。`independent` 表示未由目标数值/标签人为构造，并非宣称所有生理变量相互统计独立。simulation 模式仅接受明确标注的 synthetic，不能混入真实验证。

响应保留 `score`、每个模块的三个分量、`reasons`、`status`、`warning` 和新的 AlarmState。可用、已校准时 warning 为 0/1；资料不足或未校准为 null。分数不是概率。相同人、相同 bundle 必须按日递增调用；重复请求应返回业务层已保存的响应，不能再次推进状态。不同 bundle、评分方法或时区的 state 不可复用。

## 测量与终点

原始 `Observation` 字段为 `observation_id, participant_id, variable, value, unit, start, end, available_at, timezone, provenance`。`workflows.prepare.prepare_day` 支持明确的睡眠区间、含能量进食、24 格活动/佩戴分钟、设备静息心率和分类文本。只有明确完整的睡眠/饮食记录才能产生相应日测量；未佩戴不能算静止。

`sleep_episode` 表示实际睡着的非重叠区间（unit=state）；`main_sleep_period` 是经过明确识别的主睡眠入睡至最终觉醒区间（unit=interval）。存在碎片化或午睡时，必须提供主睡眠区间才计算中点，不能把最长的一段连续睡眠误当整晚中点。v2 主睡眠中点按完成日归属；总时长只累加研究日内部的 asleep=1 区间部分，清醒和前一研究日部分不计入，不会重复计数。

终点分支依次调用 `rolling_outcome_measures` → `personal_anchor / fit_criteria / assess_day` → `detect_events` → `future_label`。终点阈值来自独立稳定参考，所有终点输出保存在离线结局分支。`future_label.observed_days` 为本地研究日起始日期的完整结局评估覆盖清单，必须传入对应 timezone，调用者不得用设备有记录的日期替代结局完整日期。

## 研究拟合 JSON

`develop --input ...` 的顶层键：

- `reference`：ReferenceCandidate 列表，每项含 `panel, stable, stability_evidence_id, stability_available_at, baseline`。
- `pairs`：DiscoveryPair 列表，每项含 `participant_id, stable, pre_event, evidence_available_at, evidence_id`；stable/pre_event 为固定面板顺序的原始值；真实研究还必须提供 `stable_panel, pre_event_panel, event_onset, outcome_protocol_id`，代码核对数值、来源资格、模型身份及事件前时间窗。选取规则必须事先冻结。
- `calibration`：LabeledCase 列表，每项含 `request`（在线结构）、`label`（`value, reason, onset`）、`label_available_at`。真实研究还必须包含与开发阶段一致的 `outcome_protocol_id`。标签只在预测完成后参与校准。
- `calibration_events`：真实研究必需的独立事件登记列表，每项 `participant_id, onset, confirmed_at, version`。
- `calibration_monitoring`：真实研究必需的监测区间，每项 `participant_id, first_issue_day, last_issue_day, timezone`，日期是计划发报的本地日（两端包含），不是只有成功取得数据的日期。
- `reference_cutoff, discovery_cutoff, calibration_cutoff`：有序拟合截止时刻。
- `text_model_id`：joint12 必需；objective8 留空。

`validate --cases ...` 输入同一 LabeledCase 结构；`--events` 为独立事件登记；`--monitoring` 为独立监测日历；`--as-of` 必须晚于所使用结局的可得时间。若未提供事件登记，只有明确 simulation 模式允许执行，报告会明确分母仅包含可评估标签对应的事件。

Bundle 包含 study、reference（完整参考及 scaler）、discovery、calibration、text_model_id。每个拟合产物与整体分别校验哈希，检查人群分离、时序顺序和尺度一致性。改动任何配置或文本权重都应重新产生 bundle；历史文件不覆盖。哈希用于完整性检测，不构成来源真实性的数字签名。

## 文本训练

```powershell
python -m rhythm_dnb train-text --corpus reviewed.json --base models/pinned-model --config configs/text/macbert.json --output-dir runs/text-01
python -m rhythm_dnb score-text --checkpoint models/checkpoint --category stress --text "今天一直担心工作上的事情" --output runs/text-result.json
```

语料每项包含 `example_id, category, text, scores, split, group_id, participant_id, origin, review_status, annotation_evidence_id`。默认要求所有类别、组/人隔离、人工复核证据，原始/忠实翻译的观测文本；构造文本仅能通过 Python 显式参数加入辅助训练，绝不进入真实验证/测试。既有历史语料必须先提供这些复核信息，不会因行数很多就自动通过。

训练从本地下载收据校验模型版本与文件，按验证集选择最佳轮次，最终才计算冻结测试误差。梯度累计按实际样本数加权，包括最后不足一组的批次。独立类别回归头使用 Huber loss，样本先按输出数平均，避免类别输出数量不同导致权重偏差。

## v2 结局编排

`fit-endpoint --input stable-people.json --study configs/study_v2.json --cutoff ... --output criteria.json` 冻结稳定人群结局阈值；输入使用 fit_criteria 文档中的字段。

`endpoints --input person-endpoint-days.json --study configs/study_v2.json --as-of ... --output timeline.json` 构造个人基线、评估与事件。顶层为 `days, baseline_start, criteria`；days 每项为 `participant_id, day, timezone, values, available_at, provenance`。values 含 sleep_midpoint_h、first_caloric_h、last_caloric_h 及已经按暴露量归一化的24格 activity_hours。

Python 的 `workflows.endpoints.label_timeline` 用同一StudyConfig生成后续标签。timeline 的 protocol_id 应写入 DiscoveryPair 和 LabeledCase；冻结准则与窗口设置改变后身份也改变。

`prepare` 的JSON可额外给出 `text_checkpoint` 本地路径，CLI会构造冻结文本预测器。连续文本时间/单位和来源仍须真实。
