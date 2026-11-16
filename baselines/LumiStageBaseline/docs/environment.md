# 环境与版本

复用已经验证的 Linux CUDA 环境。三套 Python 路径分别填入 `examples/config.json` 的副本。示例相对路径相对于配置文件所在目录。入口使用离线模型目录，上游仓库需要完整的工作树。请从模型官方渠道取得授权与权重，配置文件中只写本地目录。

| 组件 | 实际来源 | 固定版本 |
|---|---|---|
| Qwen 软件 | [QwenLM/Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1) | ab96caa944157ca28a4f36b695c3047549f48f98 |
| Qwen 模型 | [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) | d26bb61231c349cf6b7896fa83353113880e1ba3 |
| Diffusers 的 Qwen 2.1 实现 | [huggingface/diffusers](https://github.com/huggingface/diffusers) | 843f1743e259e510f22065c0cb917d417b98d9f5 |
| PropFly | [pmjames16/PropFly](https://github.com/pmjames16/PropFly) | 1c50638aaf62c27634852be5e8689970b835d788 |
| VideoCoF | [knightyxp/VideoCoF](https://github.com/knightyxp/VideoCoF) | f45b8009616b3ef8e2024a7da585249b4c74fc2b |

生产环境查询得到 Python 3.12.16、PyTorch 2.8.0+cu128、Diffusers 0.42.1、Transformers 5.22.1。仅安装同名包可能得到不同的 Qwen 实现；实际 `QwenImage21Pipeline` 文件 SHA256 为 `e75e4ce38d38e194edaac818bba99b2115b4bb2bfdbd15ce03a2813b0fe8856c`，图像阶段会校验。每个环境与已记录依赖的精确版本见 [versions.json](../versions.json)。上游项目各自的依赖、GPU 支持和使用条款随相应固定版本。

PropFly 使用 Wan2.1-VACE-1.3B-Diffusers 加作者 PropFly-1.3B 方法权重。VideoCoF 使用 Wan2.1-T2V-14B、VideoCoF 方法权重和 FusionX 单文件。已记录的仓库、revision、相对文件名、大小与已有 SHA 随版本清单提供。

```sh
python lumistage.py weights workspace/config.json
```

权重校验会实读有摘要的文件。Qwen 的小文件支持 Git blob 校验。CPU 检查需要 NumPy、Pillow，以及能用的 imageio/ffmpeg 后端或 PATH 中的 ffmpeg 和 ffprobe。真实模型推理分别使用配置中三个上游兼容的解释器。Qwen 采用 Linux 的 CPU offload 与内存回收逻辑；VideoCoF 的 GPU 常驻配置需要支持整模加载的显存。这里沿用经过测量的参数，保留完整 49 帧。
