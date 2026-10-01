# 实际代码文件与职责

以下清单由实际源文件生成；不是尚未实现的占位目录。

| 文件 | 职责 | 主要定义 |
| --- | --- | --- |
| `src/rhythm_dnb/__init__.py` | 包标识及少量明确导出 |  |
| `src/rhythm_dnb/__main__.py` | python -m rhythm_dnb入口，仅转发CLI |  |
| `src/rhythm_dnb/api.py` | 唯一稳定的Python调用入口；加载固定包并调用准备、评分与报警 | `RhythmPredictor` |
| `src/rhythm_dnb/bundles.py` | 版本化研究包/推理包的保存、加载、完整性和兼容性检查 | `make_bundle`, `check_compatibility`, `save_bundle`, `load_bundle` |
| `src/rhythm_dnb/cli.py` | 按命令组解析参数并调用工作流 | `main` |
| `src/rhythm_dnb/config.py` | 校验研究配置与显式路径；解析配置优先级 | `StudyConfig`, `load_study`, `resolve_paths` |
| `src/rhythm_dnb/contracts.py` | 共享类型与结果状态；观测、特征、预测和结局使用不同类型 | `Provenance`, `Observation`, `FeatureValue`, `DailyPanel`, `PredictionRequest`, `DNBResult`, `AlarmState`, `PredictionResponse`, `OutcomeAssessment`, `OutcomeEvent`, `OutcomeLabel`, `EvaluationDay`, `parse_panel`, `parse_request`, `parse_observation` |
| `src/rhythm_dnb/dnb/__init__.py` | 包标识及少量明确导出 |  |
| `src/rhythm_dnb/dnb/classic.py` | 经典DNB三个分量及比值；群体/个人窗口共用公式 | `dnb_components` |
| `src/rhythm_dnb/dnb/modules.py` | 模块定义、候选枚举、补集及预算检查 | `candidate_modules` |
| `src/rhythm_dnb/dnb/reference.py` | 独立稳定参考资格与抽样；固定尺度及参考相关缓存 | `ReferenceCandidate`, `fit_reference` |
| `src/rhythm_dnb/dnb/single_sample.py` | 固定参考加一个样本的差分关联和sDNB | `sdnb_components`, `sample_network`, `discover_sample_modules` |
| `src/rhythm_dnb/dnb/statistics.py` | DNB共享数值检查、完整行、相关矩阵、成对均值和分母保护 |  |
| `src/rhythm_dnb/io/__init__.py` | 包标识及少量明确导出 |  |
| `src/rhythm_dnb/io/fetch.py` | 显式下载公开源，校验哈希和断点结果；不执行数据解析 | `fetch_source` |
| `src/rhythm_dnb/io/repository.py` | SQLite事务、表与审计；分别管理节律库及文本库连接 | `RhythmRepository`, `TextRepository` |
| `src/rhythm_dnb/io/sources/__init__.py` | 包标识及少量明确导出 |  |
| `src/rhythm_dnb/io/sources/delsom.py` | 只解析delsom源的格式、时间语义、测量和原始标签证据 | `build_records`, `build_treatment_records` |
| `src/rhythm_dnb/io/sources/hospital.py` | 医院分段表和约定宽表适配；不猜缺失日期 | `read_observations` |
| `src/rhythm_dnb/io/sources/legacy_schema.py` | 隔离历史宽表契约；仅由历史源解析器使用，不进入在线预测 | `validate_record` |
| `src/rhythm_dnb/io/sources/lifesnaps.py` | 只解析lifesnaps源的格式、时间语义、测量和原始标签证据 | `build_records` |
| `src/rhythm_dnb/io/sources/manchester.py` | 只解析manchester源的格式、时间语义、测量和原始标签证据 | `assemble_records`, `build_records` |
| `src/rhythm_dnb/io/sources/pmdata.py` | 只解析pmdata源的格式、时间语义、测量和原始标签证据 | `build_records` |
| `src/rhythm_dnb/io/sources/rest.py` | 只解析rest源的格式、时间语义、测量和原始标签证据 | `build_records` |
| `src/rhythm_dnb/io/sources/sleep_diary.py` | 只解析sleep_diary源的格式、时间语义、测量和原始标签证据 | `build_records` |
| `src/rhythm_dnb/io/splits.py` | 按人/来源/模板群及时间生成和检查固定分区 | `make_splits`, `validate_splits` |
| `src/rhythm_dnb/io/tabular.py` | CSV/Excel输入输出、显示单位转换与单表视图 | `read_table`, `export_view` |
| `src/rhythm_dnb/io/validation.py` | 原始字段字典、单位与格式检查，保留缺失原因 | `validate_observations` |
| `src/rhythm_dnb/measures/__init__.py` | 包标识及少量明确导出 |  |
| `src/rhythm_dnb/measures/activity.py` | 活动日特征、M10/L5/RA、IS与IV及佩戴覆盖 | `hourly_profile`, `daily_activity`, `activity_regularity`, `intradaily_variability` |
| `src/rhythm_dnb/measures/eating.py` | 首末含能量事件、饮食日特征及E1 | `daily_eating`, `eating_regularity` |
| `src/rhythm_dnb/measures/panel.py` | 特征字典、固定面板注册与文本/数值对齐 | `get_panel` |
| `src/rhythm_dnb/measures/physiology.py` | 心率、静息心率等真实生理摘要 | `daily_physiology` |
| `src/rhythm_dnb/measures/scaling.py` | 固定参考标准化和确定性变换；fit与apply分开 | `circular_summary`, `fit_scaler`, `transform` |
| `src/rhythm_dnb/measures/sleep.py` | 睡眠日特征、S1及SRI测量；午睡和跨午夜一致处理 | `daily_sleep`, `sleep_regularity`, `sleep_regularity_index` |
| `src/rhythm_dnb/outcomes/__init__.py` | 包标识及少量明确导出 |  |
| `src/rhythm_dnb/outcomes/criteria.py` | 拟合独立端点参考阈值；按S1/E1/A1算日候选状态 | `personal_anchor`, `weighted_quantile`, `fit_criteria`, `assess_day` |
| `src/rhythm_dnb/outcomes/events.py` | 持续三次确认事件，区分起点、确认时间和未知区段 | `detect_events` |
| `src/rhythm_dnb/outcomes/labels.py` | 未来24小时至7日标签、t+9确认随访、删失 | `future_label` |
| `src/rhythm_dnb/provenance.py` | 数据血缘、内容哈希、真实/衍生/构造/模拟资格 | `json_default`, `canonical_json`, `fingerprint`, `file_hash`, `eligible_measurement`, `build_lineage` |
| `src/rhythm_dnb/research/__init__.py` | 包标识及少量明确导出 |  |
| `src/rhythm_dnb/research/baselines.py` | 偏离、近期趋势逻辑回归和晚融合对照 | `fit_baselines`, `deviation_and_trend` |
| `src/rhythm_dnb/research/calibrate.py` | 按完整报警策略选择阈值；校准独立于公式 | `calibrate` |
| `src/rhythm_dnb/research/discover.py` | 开发集模块稳定性选择、重采样和冻结 | `DiscoveryPair`, `discover` |
| `src/rhythm_dnb/research/evaluate.py` | 事件匹配、提前量/误报率/覆盖、固定时点指标及聚类区间 | `replay`, `event_metrics`, `cluster_intervals` |
| `src/rhythm_dnb/research/report.py` | 从已冻结数值结果生成表、科学图与报告 | `audit_legacy_store`, `save_report` |
| `src/rhythm_dnb/research/simulate.py` | 机制和应用模拟两个显式入口，共享种子与场景契约 | `simulate_mechanism`, `simulate_cohort` |
| `src/rhythm_dnb/text/__init__.py` | 包标识及少量明确导出 |  |
| `src/rhythm_dnb/text/annotations.py` | 人工标注包、复核、分歧及质量审核 | `compare_annotations` |
| `src/rhythm_dnb/text/checkpoint.py` | 模型权重及tokenizer加载、保存、设备选择和固定版本校验 | `device_for`, `inspect_checkpoint`, `load_checkpoint`, `save_checkpoint` |
| `src/rhythm_dnb/text/corpus.py` | 唯一文本监督语料视图、资格选择和语料快照 | `prepare_corpus` |
| `src/rhythm_dnb/text/dataset.py` | 文本token化、训练Dataset和Collator | `encode`, `ScoreDataset`, `Collator` |
| `src/rhythm_dnb/text/evaluate.py` | 固定语料上的文本误差、均值对照、类别与来源报告 | `average_ranks`, `errors`, `mean_baseline`, `report` |
| `src/rhythm_dnb/text/model.py` | MacBERT输出头与回归损失 | `ScoringModel`, `regression_loss` |
| `src/rhythm_dnb/text/predict.py` | 单条及批量推理，连续值供科学计算、取整值仅展示 | `TextPredictor` |
| `src/rhythm_dnb/text/schema.py` | 文本类别、固定输出维度及锚点契约 | `normalize_category`, `validate_input`, `schema` |
| `src/rhythm_dnb/text/train.py` | 优化、早停与训练产物；可选显式基础权重准备 | `validate_config`, `predict_rows`, `train` |
| `src/rhythm_dnb/timebase.py` | 时区、研究日04:00边界、12:00发布时点和as-of筛选 | `instant`, `local_boundary`, `research_day`, `forecast_bounds`, `available_as_of`, `clock_delta` |
| `src/rhythm_dnb/warning/__init__.py` | 包标识及少量明确导出 |  |
| `src/rhythm_dnb/warning/policy.py` | 连续两日、7日抑制与幂等报警状态机 | `advance` |
| `src/rhythm_dnb/warning/score.py` | 个人sDNB与滚动评分、固定模块汇总及模式资格 | `score_modules`, `aggregate_score` |
| `src/rhythm_dnb/warning/windows.py` | 按真实时间切过去窗口、完整日数和缺口检查 | `panel_vector`, `select_window` |
| `src/rhythm_dnb/workflows/__init__.py` | 包标识及少量明确导出 |  |
| `src/rhythm_dnb/workflows/develop.py` | 参考、端点拟合、模块发现、阈值校准按顺序编排 | `LabeledCase`, `score_cases`, `develop` |
| `src/rhythm_dnb/workflows/ingest.py` | 显式取数、适配、验证、存储和交换的编排 | `ingest_table` |
| `src/rhythm_dnb/workflows/prepare.py` | 生成指定as-of的测量、文本特征和固定面板 | `prepare_day`, `rolling_outcome_measures` |
| `src/rhythm_dnb/workflows/validate.py` | 已冻结预测与独立结局的正式评估、群体探索和对照编排 | `validate` |

## 外围文件

`configs/` 保存手工研究选择，拟合产物存放在外部 bundles/runs 中。`tests/core/` 验证已安装的新包；`tools/review_research.py` 扫描依赖与注释边界。`examples/` 直接调用同一 API/CLI。

科研核心已经提取为独立仓库；DNB 与测量逻辑只有一个生产实现。HTTP/APP 接入属于后续应用工作。

## v0.2 新增文件

| 文件 | 职责 |
| --- | --- |
| `src/rhythm_dnb/workflows/endpoints.py` | 贯通配置、个人基线、独立结局阈值、确认和标签；原始终点输入与DNB分开 |
| `src/rhythm_dnb/io/sources/lineage.py` | 解释六类历史来源的不同证据格式；不升级为实时合格数据 |
| `src/rhythm_dnb/research/plots.py` | 可导出的指标分布、来源覆盖、有符号网络与分量图 |
| `tools/review_research.py` | 全文件静态扫描、来源审计和合成敏感性证据生成；不修改数据库 |
| `tests/core/test_review_regressions.py` | 本轮科学边界和数据资格缺陷的回归用例 |

运行诊断脚本后，逐文件状态与 SHA256 保存在输出目录的 `file_scan.json`；发布验证见 [verification_v2.md](verification_v2.md)。
