# 网页版与本地版

输入中文，点击“情绪、睡眠、饮食、社交、压力”中的一个类别，再点“开始分析”。类别是当前模型的输入之一，不需要填写真实分数。结果为 0～100 的实验估计；未提及的指标仍可能被打分，不代表疾病诊断或 DNB 预警。

## 本地使用：没有云端算力费

从 [本地版下载页](https://github.com/ChiZhang-805/rhythm-dnb-core/releases/tag/local-web) 下载 `rhythm-text-local.zip` 并解压。

1. 安装 Python 3.13 和 NVIDIA 驱动。需要 NVIDIA 显卡；已在 Windows RTX 4060 Laptop 8 GB 上实测，建议 16 GB 以上系统内存、约 50 GB 空余磁盘用于依赖、基模与下载缓存。12 GB 以上显存余量更宽裕。Mac、无独显电脑和 4 GB 内存服务器目前不支持这份模型。
2. Windows 双击 `install-web.cmd`，首次安装环境并下载约 16.4 GB 基模。这个过程需要联网，下载中断后可重试。解压路径尽量简短。
3. 完成后双击 `start-web.cmd`，打开 <http://127.0.0.1:7860>。等待页面显示模型就绪，再输入文本。关闭运行窗口即可释放本机显存。

Linux 或命令行分别执行 `python tools/local_web.py --prepare` 和 `python tools/local_web.py`。本地运行时文本不离开电脑；应用不保存输入，不自动用于训练。准备好模型和依赖后，可断网运行。

下载包含已训练 LoRA、评分头、网页和源码，不含训练数据。为便于公开分发，清单移除了训练人群等元数据，因此清单标识不同；权重文件未改。它不是独立 EXE，也不是仅靠浏览器计算。Qwen 基模遵循随包附带的 Apache 2.0 许可。

## 已有开发环境

在源码根目录执行：

```sh
python -m pip install -e ".[text,quantized,web]"
python -m rhythm_dnb.web --checkpoint models/rhythm-text-expanded --base models/qwen3-8b
```

默认只监听本机，单次最多 500 字符；加上提示词超过模型 512 token 上限时会要求缩短，不会偷偷截断。一次只处理一个请求，繁忙时返回提示。默认每分钟最多 30 次是服务负载上限，不是科研参数，可用 `--requests-per-minute` 修改。

如部署到有 GPU 的服务器，显式设置 `--host 0.0.0.0 --allow-host 你的服务域名 --allow-origin https://chizhang-805.github.io`，再配置 HTTPS。浏览器页面的“连接设置”填写该服务地址即可；不要填写云平台密钥。跨域来源名单不是身份认证，公开服务仍应配置入口限流与费用上限。

## 托管选择（2026-10-08 核查）

| 方式 | 费用与限制 | 本项目建议 |
|---|---|---|
| 本地网页 | 无云端算力费，使用自己电脑的显卡与电力 | 已实测，可先发给有 NVIDIA 显卡的使用者 |
| GitHub Pages | 公共静态页面免费，不能运行 Python/GPU 模型 | 用作网页入口与下载页，评分需另外连接模型服务 |
| Render 免费服务 | 512 MB 内存；闲置 15 分钟后休眠 | 装不下本模型，换成动态网页也不能解决算力问题 |
| Hugging Face ZeroGPU | 合格免费个人账号可建最多 2 个；需邮箱验证且账号超过 30 天。访客有队列与每日时长额度 | 免费在线首选候选；需验证账号资格和本项目 NF4/LoRA 加载兼容性，尚未部署验证 |
| Modal Starter | 每月提供 30 美元计算额度；超额有费用，受账号条件约束 | 低访问量的备选；可承载自定义评分代码，需设置用量限制并实测冷启动 |
| Runpod Serverless | 按工作进程运行秒数计费；启动、计算、空闲等待和存储均可能计费 | 访问量增加后的候选；不能只用单句推理时间计算费用 |
| 常驻 Runpod Pod | 无请求也占用租用时间；还可能有存储费用 | 不适合当前追求低成本的公众试用 |

参考：按当前截图的 0.49 美元/小时连续开 30 天，计算费约 352.80 美元，另加存储等费用。因此免费静态网页加常驻 GPU 并不等于免费服务。免费云平台也不能保证无限用户、无限调用。

依据：[GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)、[Render 价格](https://render.com/pricing)、[Render 免费限制](https://render.com/docs/free)、[ZeroGPU 条件与额度](https://huggingface.co/docs/hub/spaces-zerogpu)、[Modal 价格](https://modal.com/pricing)、[Runpod Serverless 计费](https://docs.runpod.io/serverless/pricing)。价格和资格可能调整，以创建服务时控制台为准。
