# LumiStageBaseline

在同一段真实视频上对比两种灯具编辑方法。每个请求绑定一段 49 帧源片、一个指定灯具和该操作的目标。Qwen-Image-2.1 编辑首帧，PropFly 接收实际生成且经过图审的 PNG 并传播 49 帧。VideoCoF 在同段视频上执行文字指令。适合查看某个编辑是否沿用了指定的灯具、位置和相机要求，以及失败发生在哪一阶段。

| 方法 | 直接送入模型的条件 | 本实验的原生输出 |
|---|---|---|
| Qwen-Image-2.1 | 实际提示词与按顺序固定的多张图，角色见请求的 `condition_images` | 832×480 PNG |
| PropFly | 源 49 帧、真实 Qwen PNG、传播提示词 | 49 帧，832×480，10 fps |
| VideoCoF + FusionX | 源视频与一条编辑文字 | 49 帧，10 fps，尺寸随已冻结的视频输入；首轮常见 640×352 |

**从这里开始。** 在 StarReal 仓库根目录执行下面的命令，可用 Python 标准库检查脱敏示例的七项任务协议和模型参数。输出明确标记是否启动 GPU。示例中的零摘要仅说明格式。

```sh
cd baselines/LumiStageBaseline
python lumistage.py dry-run examples/request.json
python lumistage.py --help
```

运行真实场景所需的 StarReal 视频、灯具资产、模型权重和上游代码由使用者另行提供。仓库记录了生产版本和文件指纹。按顺序阅读 [环境与版本](docs/environment.md)、[数据流与几何](docs/architecture.md)、[从输入到完整视频](docs/reproduction.md) 和 [参数、限制与验收](docs/evaluation.md)。[验证记录](docs/validation.md) 列出本次交接实际做过的检查。

| 文件 | 职责 |
|---|---|
| `prepare.py`、`geometry.py`、`guides.py` | 从实际请求和 CoF 样本绑定源片、条件、GLB 与两条相机轨迹，导出逐帧投影和图审板 |
| `lumistage.py` | 参数与输入校验、原生入口、单设备互斥、有限队列与断点续接 |
| `native/` | Qwen 官方加载调用、逐值加载观察器、PropFly 完整 49 帧写出、VideoCoF 完整 FusionX 适配 |
| `versions.json` | 三套实测 Python 环境、模型 revision、仓库 commit、关键源码／权重摘要 |
| `docs/production_sources.json` | 与生产脚本逐项对应的原始 SHA256 |
| `examples/` | 可检查的请求结构、环境配置和人工审核空白表 |

实验的[公开结果网页](https://frontrear-real-video-baseline-lab-20261113.static.hf.space/)可选择案例、操作与方法。原始成片和下载索引见 [StarReal 公开 Dataset](https://huggingface.co/datasets/FrontRear/starreal-video-baselines/tree/main/baseline)。
