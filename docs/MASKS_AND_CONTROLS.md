# 两类 Color 与 signed Intensity

## 四种字段组合

| 空间 support | Color | Intensity |
|---|---|---|
| lamp_silhouette | 实际整灯 mesh / 可见原灯具支持区域上的恒定线性RGB | 同一 support 上的 signed gamma |
| bbox_silhouette | 三维 OBB 完整投影填充轮廓上的恒定线性RGB | 同一 bbox support 上的 signed gamma |

bbox_silhouette 是投影盒子的填充轮廓，不是灯具轮廓，也不只是8条线。空间条件和 RGB/相机都为1280×704，49帧。正式完整组还有相机、参考、各类独立可见性证据、两版PCF与视频等，单独一个 low-level NPZ 不等于完整组。

## mask 如何得到

`frozen/geometry/real_geometry.py` 与 `mesh_zbuffer.cpp` 将真实完整资产网格或明确标记的原灯具估计几何，经实际 K/w2c 投影。场景深度从相应帧的实际深度估计构面得到；移除原灯时须区分原场景与移除后的源区域。相机变化时，目标 mesh 固定在选定的世界位置，通过新相机重投影。

生产输出分别保存：
- amodal；
- scene_occluded_estimate；
- visible_known_estimate；
- unobserved_scene_region；
- visible_with_unknown_estimate。

未知背景区域可以作为明确标记的估计完整 support，但不能写成“观测到的可见性真值”。原灯具的观测A mask、估计B mask、隐藏/补全部件也应分开保存。

当前 bbox 主字段使用完整投影 support；另存 bbox_scene_visible_estimate。不能声称主 bbox support 已经自动等同“受真实场景遮挡后的可见 bbox”。

## Color 必须纯色

同一个对象、同一条颜色控制的非背景像素应是同一个RGB向量，背景为0。不能将 P/Q 标签的紫/青混在一盏灯上；不能把PCF的红/蓝/黄可视化当 Color mask；不能把置信度或可见性渐变偷偷编码到颜色里。未知区/遮挡区独立保存。

`build_case1_masks49.fields` 是保留的原始 ROI 数学。当前使用的缓存版本是 `production_snapshot/runtime83/build_native_cached_conditions49_QH_v3.py`。纯色导出与逐member检查见 `production_snapshot/solid_completion/audit_and_export_solid_all35_v1.py`；完整绑定读者见 `frozen/readers/starreal_complete_reader.py`。

## 相对强度

存储的 raw_gamma = gamma_target - gamma_source。进入模型时：
- relative_gamma = raw_gamma / 1.1，只做一次；
- relative_alpha = (alpha_target - alpha_source) / 0.9，只做一次；
- model Intensity 背景为 -1，前景按 signed gamma；不能因 gamma=0 就误认为没有灯。

控制档位是显式接口参考，**不是对真实视频发光强度的测量**：

| 操作 | source→target |
|---|---|
| insert | 0→0.15 / 0.5 / 1.05 |
| remove | 0.5→0 |
| move / replace / move_replace | 0.5→0.15 / 0.5 / 1.05 |

例如 Remove 的 raw_gamma=-0.5，模型前景=-0.5/1.1，而不是正数亮度，也不是提前归一化后再除一次。负强度的红色诊断图是可视化编码，不是原灯具颜色。

这些不是 Original UE 随机8套参数分布的重新抽样。实拍条件没有重新合成真实RGB；8个显式请求颜色与原有合成训练的随机采样不能混称同一件事。

## 纯移动与 source appearance hint

纯 move 保持源灯具身份、大小、朝向与外观意图；默认 unchanged gamma 的差值为0。单独的 recolor / brightness 请求仍可额外提供。

当前 source hint 使用 all49 观测RGB和保留mask：每帧转线性RGB，选亮度最高20%的支持像素，取帧均值，再取时序中位数并以最大通道归一化。它受反射、曝光、间接光和截断影响，不是发射光谱或绝对功率。代码为 `prepare_native_source_appearance_reference_QH_v2.py`。

## Remove 最新要求

旧 core 中“目标不存在”的空 Color 不能直接当最终 Remove 条件。最新数据必须另有：
- 原灯具 source reference；
- 观测 A 与估计 B；
- 两个视图中 lamp_silhouette 和 bbox_silhouette；
- 各49帧 source Color 与 signed negative Intensity。

即196个NPZ、392个Color/support PNG，并由实际完整 reader 绑定到原 core archive SHA。代码：
`build_remove_source_conditions_QH_v2.py`、
`add_remove_observed_A_conditions_QH_v1.py`、
`remove_condition_reader.py`、
`starreal_complete_reader.py`。

Remove没有新的target reference，但必须有原灯具source reference。完整reader按operation找到正确参考，不能套旧的“remove参考数量=0”规则。

## reference

三个新灯具各绑定实际raw GLB SHA与四张既有材质参考图。Source move/Remove参考来自真实原灯具裁剪/透明图。Cycles CPU材质参考渲染是确定的三维渲染，不是生成式AI相似图。参考缺失不应伪造；旧回执、材质渲染与实际模型要逐项核对。
