# StarReal real-case construction code

这是一份从实际生产目录提取的、可追溯的 real case 条件构造代码。它包含普通 PCF+bbox、固定世界坐标的 depth-aware PCF+bbox，以及灯具轮廓 / bbox 两类 Color 和 signed Intensity 条件。私有仓库不包含真实视频、灯具 GLB、模型权重、训练包或登录凭据。

本次导出对应 2026-11-16 的生产状态：**54/59 个原始数字 ID 已完成当前条件包的青海/HF核验**。99、100、101、103、7578553 尚未完成。54 个 ID 不等于 54 个不同语义场景，也不代表新增了有监督的 edited-B RGB 真值。

## 从哪里读

| 内容 | 主入口 |
|---|---|
| 哪些地方使用 AI，以及哪些判断发生在代码外 | [docs/AI_USAGE.md](docs/AI_USAGE.md) |
| 两种 PCF+bbox 的准确区别与黄色重叠规则 | [docs/PCF_AND_BBOX.md](docs/PCF_AND_BBOX.md) |
| 两类颜色/强度 mask、Remove 补充、相对强度 | [docs/MASKS_AND_CONTROLS.md](docs/MASKS_AND_CONTROLS.md) |
| 几何、相机、遮挡和原版参数边界 | [docs/PIPELINE_AND_PARAMETERS.md](docs/PIPELINE_AND_PARAMETERS.md) |
| 安装、复现、当前生产脚本调用顺序 | [docs/REPRODUCE.md](docs/REPRODUCE.md) |
| 已经犯过的错误及其处理方式 | [docs/FAILURES_AND_LIMITS.md](docs/FAILURES_AND_LIMITS.md) |
| 每份原始源码的路径、SHA、来源 | [provenance/SOURCE_ORIGINS.json](provenance/SOURCE_ORIGINS.json) |
| 文件级 AI/程序化计算清单 | [provenance/AI_CODE_INVENTORY.json](provenance/AI_CODE_INVENTORY.json) |
| 当前数据 ID / HF archive / SHA 入口 | [provenance/CURRENT_DATASET_SNAPSHOT.json](provenance/CURRENT_DATASET_SNAPSHOT.json) |

## 当前主要代码

- `frozen/render/depth_fusion_fixed_world_yellow_overlap_v2.py`：已使用的固定世界坐标 / 共享射线深度 / 黄色投影重叠版本。SHA256 为 `a6c6c71bb8ce48f07c0e04e1fbfedd1af3d51ddf5a707f01de6c08de6c231f3f`。
- `frozen/ordinary/bbox.py`、`frozen/render/original_joint_frame_stream_v1.py`：普通框投影与原始 joint 合成数学。
- `production_snapshot/runtime83/build_native_case_two_pcf49_QH_cpu_v5.py`：实际 49 帧、5 操作、两版 PCF 与 15 个视频的生产入口。
- `production_snapshot/runtime83/build_native_five_role_geometry49_solidmask_QH_cpu_v3.py`：真实网格 / 相机 / 场景深度产生轮廓、bbox 和独立遮挡/未知区。
- `production_snapshot/runtime83/build_native_cached_conditions49_QH_v3.py`：成组生成两类 Color / 原始 signed gamma。
- `production_snapshot/remove_completion/build_remove_source_conditions_QH_v2.py` 和 `add_remove_observed_A_conditions_QH_v1.py`：补充 Remove 原灯具的 A/B 两视图、两类颜色和负相对强度。
- `production_snapshot/solid_completion/audit_and_export_solid_all35_v1.py`：纯色 Color 导出及核验；文件名中的 all35 是历史命名，不是当前范围。
- `frozen/readers/starreal_complete_reader.py`：当前完整入口，组合 core、Remove、solid Color 与 reference 绑定。
- `src/starreal/`：方便复用的低层接口；不自动选位置、不自动批准、不自动上传。

`production_snapshot/` 保留生产路径和历史命名以便核对原始 SHA。它同时包含生产、支援和历史失败实验，**不能将目录中所有脚本当作“都应运行”的流水线**。已批准版本和主入口见文档与文件清单。

## 快速运行

CPU-only 默认不申请 GPU。真实数据及大产物继续放青海。

```bash
python -m pip install -e .
python tools/build_native.py
python -m starreal render --scene examples/fixed_world_scene.json --out outputs/example
python tools/audit_repository.py
```

例子是诊断用的显式三维盒子，不是一个真实场景，不计成组。真实生产仍需有效的 49 帧 RGB / 相机 / 深度 / 已审核网格及支撑位置。完整组、两视图 Remove 和 HF 发布必须按 [复现说明](docs/REPRODUCE.md) 的完整流程完成。

## 必须保留的约束

正式 RGB 与空间条件为 1280×704、49 帧、10FPS。禁止把 640 RGB 放大当正式高分辨率输入。相机 K 与 crop/resize 使用同一变换；世界位姿不能通过复制或改名冒充新的轨迹。

相对强度为 target-source，模型入口仅除 1.1 一次；环境 alpha 仅除 0.9 一次。Remove 也必须有原灯具 Color、signed Intensity、轮廓、bbox 及 source reference。普通 PCF 与 depth-aware PCF 两版均保留。AI 图审与真实文件 reader 检查完成后才批准具体 revision；唯一 HF writer 上传并核服务端 size/SHA。

数据是实拍视频的几何/控制条件。没有生成或宣称观测到“移灯后的真实 RGB”。隐藏背面、相对深度、支撑面和新视角可见性含有估计，必须和观测 mask、未知区域分开记录。

Historical snapshots include one preserved failed candidate with a known indentation error. See [provenance/HISTORICAL_NONRUNNABLE.json](provenance/HISTORICAL_NONRUNNABLE.json); it is not a current runnable entrypoint. Frozen core and current adapters passed separate validation.
