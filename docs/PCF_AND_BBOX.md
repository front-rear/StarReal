# 普通版与 depth-aware 版

## 共同输入

固定三维 Source P / Target Q OBB、同一相机 K 和 w2c、同一 PCF 格架。盒子中心/旋转始终定义在同一个世界坐标系，不能随相机逐帧挪动盒子来追画面。生产代码用第0帧 A 相机定义一次规范 PCF 坐标原点，同时变换盒子与所有相机；这不改变相对空间关系。

`Box.size` 在便携接口中是完整尺寸；冻结 renderer 的 `half_size` 是半尺寸。rotation 列向量为 local-to-world 轴。w2c 是 OpenCV +X右、+Y下、+Z前。传给 PCF 的 c2w 是相应逆矩阵。

## 普通 PCF+bbox

主数学参考：`frozen/ordinary/bbox.py` 和 `frozen/ordinary/CASE18_bbox_dataset_original_UNMODIFIED.py` 的 joint_frames；实际704入口为 `frozen/render/original_joint_frame_stream_v1.py` / 生产49帧 driver。

先得到 PCF 背景与 P/Q 投影 mask/线，再合成：
- P 红、Q 蓝、可见投影相交标签3为黄；
- 占据区域使用 `rint(0.42 * background + 0.58 * palette[label])`；
- 轮廓线按 Original wire precedence 覆盖。

这一版不根据 PCF 格架深度挡住二维叠加框。保留它是为了与旧版同相机对照。参考源码里历史640/352尺寸和旧 Remove 空字段不代表当前数据接口；不要直接用历史 load_row 读取当前数据。

## 固定世界 / depth-aware PCF+bbox

实际文件：`frozen/render/depth_fusion_fixed_world_yellow_overlap_v2.py`。冻结 SHA：
`a6c6c71bb8ce48f07c0e04e1fbfedd1af3d51ddf5a707f01de6c08de6c231f3f`。

PCF 球/杆格架和 OBB 在同一世界坐标系里，使用同一归一化子像素射线：

1. PCF 的 sphere tracing 得到沿射线距离。
2. OBB 以 slab 精确求交得到沿同一射线的距离。
3. 近格架挡住远盒子，近盒子挡住后格架。
4. 相机移动时 P/Q 坐标不变，仅投影发生变化。

**不能把 camera-z 深度直接拿去和归一化射线距离比较**。real_geometry 的 mesh z-buffer 是 camera-z；PCF renderer 是射线距离。相关换算在 `camera_z_to_ray_distance`，不能把两个变量因为都叫 depth 就混用。

## 黄色规则是可见投影重叠

用户后来明确要求“叠加部分仍然黄色”。当前版本把 P/Q 当编辑标记：分别通过 PCF 射线深度测试后，只要同一子像素射线同时看到 P 和 Q 标记，label=3、颜色为黄色。两盒子不必在真实三维体积上相交。前面的 PCF 格架仍会挡住标记。

这与早期“仅三维体积交集黄色”版本不同。已批准 v2 不再使用那个旧规则。黄色受面光照/抗锯齿影响可有亮度差，但完整黄色像素满足 R=G、B=0；它不是 Color mask 的调色板字段。

## 格架与实拍场景深度不同

这里的 PCF 是人工的三维周期坐标格架，不是把实拍房间的三角网格直接当 PCF 背景。格架-盒子的前后关系由共享射线处理。

真实场景估计深度还用于“灯具网格可见性 mask”的场景遮挡。那是另一条几何链，见 mask 文档。两者都使用相机，但不能混为一个深度缓冲。

## 固定参数与 CPU 加速

生产：1280×704，AA2，spacing0.5，far_cells64，max_steps4096。新版盒面不透明，PCF 背景混合系数0；普通版 fill0.58。实拍世界单位通常是单目相对单位。`*_m` 和源码 metres 注释是历史命名，**不能据此宣称已标定为米**。

`pcf_trace_cpu.cpp`/`pcf_cpu_accelerator.py` 是 CPU 加速；NumPy reference 保留。正式使用前核对 trace、完整 RGB 与 aux 的相等性。背景缓存只可在 K、相机矩阵、分辨率和所有参数完全相同时重用。局部 sparse 模式在框外复用背景，在所有候选框像素的全部 AA 子样本上执行同一求交，不能把它变成低分辨率近似。

当前83的完整生产记录含 PCF在框前901693个AA样本、框在PCF前26056175个、黄色重叠2768435个，unresolved=0。这是该 revision 的证据，不是后续场景的验收阈值。
