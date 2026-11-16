# 参数、控制范围与验收

| 阶段 | 原生设置 | 能从输出核实的范围 |
|---|---|---|
| Qwen-Image-2.1 | 832×480，output_resolution 1024，40 步，true CFG 1.0，每条 seed 42，KV cache，BF16 + CPU offload | 核实官方 pipeline 源码；独立校验 transformer、VAE、text encoder 加载值和 offload 后一致性。模型只接收列入 `condition_images` 的图与文本 |
| PropFly | 832×480，完整 49 帧/10 fps，50 步；CLI guidance_scale 3，batch_frames 25、stride 2、overlap 6、max_frame_num 240 | 固定作者源码的循环内两次预测混合权重前 25 步为 0、后 25 步为 1。命令行 3 仅记录实传值。作者入口沿用原生随机状态，旧输出没有 RNG 快照；Qwen 的 seed 42 只属于首帧。保留作者 GIF 与全部输出 PNG，完整 MP4 由所有 49 帧编码 |
| VideoCoF + FusionX | 4 步、guidance 1、reasoning_frames 4、repeat_rope、source_frames=49、num_frames=49、10 fps | 初始 seed 0，同一批内采样状态延续。固定批次顺序、重试位置与权重放置方式。FusionX 必须实际加载 405 组 LoRA、447 组 diff_b、200 组 diff 共 1052 项。前后两种权重放置在一个两样本对照中共 98 帧和两份 MP4 逐字节一致，其他配置需各自评测 |

**操作。** 逐条使用 StarReal 当次所选 GLB、位置、尺寸、颜色和 PCF。画面内和遮挡处的灯具应分别说明。灯罩、灯头、灯臂、支架、杆、底座、夹座及标注附属件都计入整灯；指定的侧面玻璃/金属组件随灯一并编辑。全程保留人物、身份动作和无关物体。

| 操作 | 逐帧检查 |
|---|---|
| insert | 原灯保留，在蓝 Q 正确范围与支撑位置新增一盏指定灯 |
| remove | 红 P 的整灯及其指定附属件消失，观察残座、残线、残罩或后段重新出现 |
| move | P 清空，Q 只出现一盏同身份灯，完整部件随时间保留 |
| replace | 在原 P 完整替换，检查新灯身份、尺寸与朝向，其他灯保持 |
| relight | 改变实际照明的指定颜色/强度，关注受光物与灯壳/材质染色的区别 |
| camera_change | 对照 PCF 的 B 轨迹逐帧查看人物与结构，记录可用于几何判定的证据和失配情况 |
| move_replace | 在 P 删除整盏原灯，在指定蓝 Q 插入完整新灯 |

**单独记录五种状态。** 请求绑定、真实 PNG、完整 49 帧原生视频、可见灯具效果判定、公开发布各有独立证据。完整视频结果应附原文件 SHA、49 帧解码、源片对照、实际参数与输入、评审者和失败现象。逐帧看完整成片和源视频，至少列明首中末帧位置、人物状态、灯具完整性与时序。首帧通过只覆盖该图。轨迹需要单独保存相机或特征重投影证据，注明匹配数、覆盖片段、残差、不可观察部分；没有可靠匹配时记作无法量化。文字描述“约 10°”和单目估计均不能作为实际 6DoF 真值。公开失败样本应带其具体失败判语。

**变更映射。** [生产源码指纹](production_sources.json) 的 copied_exactly 表列出原样保留的逐值审计、原生 49 帧写出和 FusionX 数值适配代码。Qwen 的单次加载及每行 sampler 参数和 PropFly 参数沿用固定生产入口。目录、队列、历史多级 manifest 与主机解释器改由配置／请求传入。VideoCoF 在执行时从锁定且无修改的官方工作树读取 fast_infer.py，仅复现生产版的 `sequential_cpu_offload → model_full_load`，完整脚本摘要必须与生产版一致。FusionX 参数路径来自同一次实际 CLI。权重和上游仓库由使用者另备，发布目录里保留指针与指纹。
