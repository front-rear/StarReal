# 从一个 StarReal 请求开始

1. 选定案例、操作及该操作标注的变体。保存实际 manifest、VideoCoF 批次和 sample_id、相机/几何 NPZ、指定 GLB、源视频和原始帧。不同资产/颜色变体各用自己的不可变请求，同一份比较使用同一对象、位置与时序。历史修订可让两种方法绑定不同祖先；这种情况要保留各自真实请求，将新对照另立请求，避免改写旧成片。
2. 读取操作的 specification，拿到绑定的相机文件及 SHA。清单含独立几何包时传入 `--geometry` 和 `--geometry-sha`；字段名有差异时用 `--geometry-fields` 指定实际 NPZ 字段。选择 GLB 时，校验其 SHA 和请求记录一致。
3. 生成可移植请求，核对源视频逐帧像素、CoF 冻结输入、Qwen 实际条件和 A/B 投影。示例使用自备数据的环境变量。

```sh
python prepare.py --manifest "$MANIFEST" --cof-batch "$COF_BATCH" --sample-id "$SAMPLE" \
  --camera "$CAMERA_NPZ" --camera-sha "$CAMERA_SHA256" --asset "$ASSIGNED_GLB" \
  --output workspace/case
python lumistage.py check workspace/case/request.json --report workspace/source_check.json
python guides.py workspace/case/request.json workspace/input_review
```

`prepare.py` 直接读取原 Qwen 的图像顺序与提示词、PropFly 的实际传播文字、所选 VideoCoF 样本的实际编辑文字。独立制备的 640×352 FFV1 输入会与源 49 帧按生产用 Pillow LANCZOS 缩放的 RGB 逐帧核对。若原制备使用另一种已验证的变换，可通过 `--cof-pair-proof` 绑定真实对照记录，格式见脚本。缺少资产或位置等必要配置时保留缺项。若修改 `condition_images`、输入角色或提示词，需要新建请求、重新核实体和重新生成首帧。

**实际运行。** 拷贝环境配置并填写自有路径。看过完整源首帧、目标位置和灯具参考后，填写 `examples/admission.template.json` 的审核人、观察及按批次顺序列出的请求 SHA256，保存为 `workspace/admission.json`。Qwen 跨请求共用一次已审计的模型加载，每条单独设置同一初始种子。VideoCoF 跨请求保持真实输入顺序。

```sh
python lumistage.py qwen workspace/case/request.json --config workspace/config.json \
  --admission workspace/admission.json --output workspace/run-qwen
python lumistage.py videocof workspace/case/request.json --config workspace/config.json \
  --admission workspace/admission.json --output workspace/run-cof
```

Qwen 成功后，`run-qwen/complete.json` 列出真实 PNG 的相对路径。打开该 PNG 与所有条件图。复制 `examples/firstframe_review.template.json`，填写实际观察、是否达到首帧可见编辑要求、`decision` 以及请求、PNG、同目录 `qwen_record.json` 和 `native/propfly_native_observed_runner_v1.py` 的 SHA256。允许为基线记录传播真实失败首帧，仍需保留失败原因和 `visible_firstframe_pass=false`。

```sh
python lumistage.py propfly workspace/case/request.json --config workspace/config.json \
  --admission workspace/admission.json --output workspace/run-prop \
  --png "$ACTUAL_QWEN_PNG" --review workspace/firstframe_review.json
python lumistage.py decode "$NATIVE_MP4"
```

**有限队列和恢复。** 参照 `examples/queue.template.json` 填写任务列表。所有待使用同一 GPU 的进程使用同一 `gpu_lock`。调用 `python lumistage.py queue workspace/queue.json` 一次处理列出的任务；首帧待审时可先运行能独立处理的 VideoCoF 列表。各次尝试使用新的目录。失败保留 `failure.json`、日志或进程中断时留下的现场。重新执行同一命令会校验已完成结果的请求、审核与文件 SHA；尚未完成的尝试使用新目录。已有运行的服务器应沿用该服务器原调度器与同一把 GPU 锁。此仓库的运行器不会修改外部生产注册表。

**旧试验的边界。** 生产曾在仅修改传播文字时复用已经人工审过的父请求 PNG。有效祖先链应同时校验 parent/current 请求、原始生成记录、原生调用、条件、GLB、PNG 和人工审核摘要。本仓库命令默认跑 fresh 首帧并要求完全匹配当前请求。复查历史时按其真实生成父记录追溯，计数时将首帧与两条 49 帧视频分开。
