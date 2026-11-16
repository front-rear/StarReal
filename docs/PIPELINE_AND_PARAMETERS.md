# 实际流水线、输入和参数

## 入口索引

这里先列当前 Native49 分支。production_snapshot 中许多旧案例脚本保留了旧绝对路径、旧阶段或有限试验，不应按文件名排序全部执行。SOURCE_ORIGINS.json 逐文件给来源路径、SHA、角色；AI_CODE_INVENTORY.json 还标出模型调用位置。

| 阶段 | 当前实现或参考 |
|---|---|
| 从已有输入构建 Native49 输入证明 | production_snapshot/runtime83/prepare_native_from_readonly_SFTP_QH_v1.py |
| Native 原始 RGB/深度/K/E/置信度的读取 | production_snapshot/runtime83/read_native_history_cameras_and_source49_QH_v2.py |
| 实际 SAM2 CPU 及逐部件可见轮廓修订 | production_snapshot/all59 下 build_next_observed_parts_SAM49_QH_CPU_v1.py、probe_case83_*SAM*；AI_CODE_INVENTORY 列实际匹配文件 |
| 固定世界 source/move/三款真实 GLB 几何与可见性 | production_snapshot/all59 的 build_*geometry49* 与对应实际 case_profiles；frozen/geometry/real_geometry.py |
| 有限条件链 | production_snapshot/runtime83/run_native_reviewed_finite_chain_QH_v3.py |
| 两类 Color / signed Intensity 条件 | production_snapshot/runtime83/build_native_cached_conditions49_QH_v3.py |
| source_hint 颜色参考 | production_snapshot/runtime83/prepare_native_source_appearance_reference_QH_v2.py |
| 两版 PCF | production_snapshot/runtime83/build_native_case_two_pcf49_QH_cpu_v4.py、encode_native_pcf_variant_stream49_v1.py |
| 原灯 Remove 条件补齐 | production_snapshot/remove_completion/build_remove_source_conditions_QH_v2.py 及该目录的后继/双视角脚本 |
| 后续纯色条件修复、完整 reader | production_snapshot/solid_completion/；frozen/readers/starreal_complete_reader.py |
| 实际图审后的打包、HF 与回读 | production_snapshot/all59 与 shared_project 的 stage / publish / verify 脚本；禁止跳过具体 revision 的图审 |

项目通用 source scripts 与案例派生脚本在生产期以独立冻结副本运行。因此 runtime83 的硬编码 R 路径并不是跨案例配置接口。复现时先校验 SOURCE_ORIGINS 和实际案例 config，再逐项映射数据目录；不得修改冻结代码后冒称同 SHA。

## 数据流

1. 核原始视频 SHA、解码帧序列、裁剪、时间戳，核对应同一视频/时间线的 Native depth、K、E、点置信度和 SAM。仅文件名或同一个 Pexels ID 不足以证明同一片段。
2. 对 source 灯具和目标落点做实际 RGB 图审。旧位置合理就保留；不合理才调整。桌面需真正可用且无书、人、边沿穿插，壁灯需真实墙面支撑。不能用几何数值通过替代语义审图。
3. 原灯 source / move 保留身份，用可见表面和明确标记的未知补全部件构成静态世界几何。插入、替换、移动+替换分别绑定一款真实新 GLB。必须核资产 SHA、朝向、类型、底座/背板接触、全49帧投影。
4. 同一 K、w2c_A、w2c_B、固定 source / target 几何生成两类 mask。新 B 视角的不可见面是 estimate，不叫 SAM 实拍或 RGB 真值。
5. 生成普通和 depth-aware 两版 PCF，使用同一固定世界归一化、同一相机。P/Q重叠仍黄色；格点/格线有双向遮挡。
6. 核每帧条件的支持区/颜色常量/带符号强度及对应几何。Remove 补 source 的 A/B 双视角两类 Color/Intensity，reference 补真实原灯和三款 GLB。
7. 独立审具体 revision，完整 reader 实际读回，再打包、上传、核 HF commit/path/size/LFS SHA 与公共读回；写 current/readme 索引。生成、预览、已上传和已核验为不同状态。

## 分辨率、时间与相机

正式当前输出是1280×704、49帧、10FPS。已是704的 Native 输入不能重复裁剪。1080/720或4K源必须根据真实头信息用 Original cover-resize 到1280×720后上下各裁8，并同步修改K。禁止640放大冒充高分辨率源。条件与 RGB 的 crop、像素中心、帧索引必须一致。

VGGT 的 E 是 OpenCV world-to-camera；世界尺度通常只有相对单位。先取 A 的第0帧固定归一化原点，再对所有49帧保持不变。B相机是条件视角，不意味着已生成其真实RGB。不同案例B轨迹可能不同，以包内矩阵为准，不把“10度”等历史字样当统一授权。

发现原视频49帧实际上为15FPS时，不能 resample 到10FPS后把越界帧 clamp 为第48帧：这样会出现重复尾帧，且旧深度/SAM缓存不再对应。99、100、101、103、7578553在本次54/59快照尚有各自具体缺口，不能为了计数忽略时间线/可见性问题。参见 FAILURES_AND_LIMITS.md。

## 和 Original 的关系

原版 UE 脚本来源：
`/home/fangrui/LampEdit/source_snapshot/ROOMSDAY/origin_UE_python_scripts`。

合成训练的 Original 约束为24mm、10FPS、4000lm按原组件/socket分配，环境和目标分离，PT128 spatial×1 temporal /8 bounces/1000截断/native denoise/曝光-5，EXR逆补偿32，随机8套alpha/gamma/color原分布、逐轨固定曝光和白点。室内600/室外900距离基数可按人类授权±100cm；world/camera不得造假。

**这些是合成UE的原版参数，不应声称真实视频也按24mm/4000lm/PT128拍摄。** 本 real-case pipeline 对输入真实视频主要构造几何与编辑条件，并未重渲染实拍背景或产生监督编辑后的 B RGB。可对齐的是 Original crop、字段语义、PCF/box约定、reader和相对alpha/gamma归一化。原版训练使用 (target-source)/0.9 与 (target-source)/1.1，各只做一次；打包/上传不能悄悄修改系数。

Original Wan/Base 的未改源码置于 frozen/original，只作为训练入口语义与裁剪依据。不能因为代码导出里有模型 pipeline 引用，就声称本次实际运行了神经编辑视频模型。
