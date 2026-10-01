# 开发协作

从 `main` 创建功能分支，通过 Pull Request 合并。此仓库只维护 DNB 科研核心；新增功能放入既有分层，避免在脚本或服务中复制一套公式。

## 本地检查

```sh
python -m pip install -e ".[research,text,io,plots]"
python -m unittest discover -s tests -q
python -m rhythm_dnb demo --output-dir runs/my-demo
python tools/review_research.py --output-dir runs/my-review
```

测试使用合成数据和临时目录。`tests/fixtures/migration.json` 是从迁移前 DNB 实现记录的合成回归结果，不包含参与者数据。测试针对已安装包执行，不通过修改 `sys.path` 静默替换包。发布时还应构建 wheel 并在独立环境安装验证。

GitHub Actions 在 Linux 和 Windows 上使用 Python 3.13 检查 wheel 安装、完整测试和命令行模拟。`requirements-core.lock` 仅记录本次本机验证的关键版本，不是跨平台完整传递依赖锁；便携安装范围以 `pyproject.toml` 为准。

## 代码与科研变更

- 生产函数保留英文 `# PSEUDOCODE:`，写明处理步骤；docstring 说明单位、形状、缺失处理和科学假设。中文使用说明写在文档。
- 测量、群体发现、终点和在线预警各自独立。推理不得读取未来标签或重新拟合参考、尺度与阈值。
- 修改公式、缺失策略、时间语义或参数时，更新协议、参数表、版本和针对实际风险的回归测试。保留失败和拒判状态。
- 数据质量修复必须保留来源、原值与修订证据；不能把构造值升级为实测值。禁止把不同数据集的人按列拼成一个虚拟人。
- 数据库、原始文本、模型权重、个人路径、访问凭据和研究输出留在 Git 之外。提交前检查 `git diff --cached`。

运行中的数据库与模型由各研究环境自行管理。公开源码不代表第三方数据或预训练模型的再分发权限；按原始来源提供的访问与使用条款取得这些资源。
