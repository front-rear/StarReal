# 本次交付验证

**真实数据的 CPU 验证。** 在已有计算环境选取 StarReal 83 的独立纯替换请求，通过仓库的 `prepare.py → lumistage.py check → guides.py` 正常入口。实际源片为 1280×704/10 fps，解码 49 帧，逐帧 RGB 与冻结源帧对应。另将 49 帧按现行制备方式缩放至 640×352，与实际 VideoCoF 的 FFV1 输入逐帧校验。Qwen 使用四张条件图，原包中的 P 和新灯 Q 同中心。相机 A、B 分别投影 49 帧，审核板在验证目录实际生成。此验证未启动 GPU。

| 实际绑定 | SHA256 |
|---|---|
| 纯替换请求 | 608ab72d563f14e359ed5580b1d1a91f24f8d019688f24eb549c88e0ba95fe20 |
| 原生 VideoCoF 批次 | 0d6cfba22b1c5122d864099381931f4731a553eeaebe04b19dac89956d53cc7d |
| 原始源片 | e7bc822a31e1d23a9c02a30385ef86e49fc818c83d96f333c5a7a0d651e95567 |
| 完整灯具 GLB | f2d37ea6414858489fe415d54898c0d7d19d36e2e3ce6ede28439b995040341b |

**源码和原方法。** 两个上游 git 工作树的已跟踪文件干净，关键入口与固定 HEAD 对应。对已整理生产代码逐个校验摘要。保留冻结的 Qwen 官方模型 revision、PropFly 作者采样/写帧入口和 VideoCoF 完整 FusionX 适配。可移植版本本次只做 CPU 检查。上面真实源片的既有成片属于原生产运行，网页保留当时的实际输入和判语。

独立的本机 CPU 测试另外生成 49 帧临时合成源，验证正常预检和对照板、错误 SHA 拒绝、纯替换错误 Q 拒绝、CLI dry-run 与全部 Python 语法；合成媒体只存在于临时目录。六个逐字节保留的生产模块全部通过 SHA 比对。机器可读结果见 [validation.json](validation.json)。
