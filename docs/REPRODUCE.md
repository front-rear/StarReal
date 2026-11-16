# 复现与验证

## 小型、可移植 CPU 核心

Linux/Python3.9+、g++、OpenMP；Python依赖固定在pyproject.toml。源码树中：
```bash
python -m pip install -e .
python tools/build_native.py
python -m starreal render --scene examples/fixed_world_scene.json --out /tmp/starreal_example
python tools/audit_repository.py
```

输出普通PCF、depth-aware PCF及同射线深度辅助字段。示例是明确的程序几何，不是某个真实场景的完成证明。所有示例/测试临时输出应写远端临时目录，绝不写回正式case。

`mesh-masks` 是单帧几何/请求颜色调色板组件入口，必须传真实mesh顶点/面、相机K/E与明确操作。没有提供scene_depth时遮挡是未知估计。这个小入口并非完整case打包器，也不自动完成Remove双视角/source_hint/reference/图审/HF；这些必须由生产脚本执行。Remove的完整生产入口见MASKS_AND_CONTROLS。没有用假的edited RGB填补缺失真值。

源码依赖通过src/starreal/_frozen原样副本随包安装；它与顶层frozen逐文件SHA相等。C++内核按本机CPU编译，不提交.so。生产可追溯版本仍以顶层frozen和provenance记录为准。没有自动下载权重、跑GPU或上传的入口。

## 完整真实case所需外部输入

- 实拍49帧及视频/帧SHA、Original crop和时间索引证明。
- 同时间线Native depth、点置信度、K、w2c；source observed SAM及所有补充可见部件标记。
- 真实500+灯具库中按实际SHA绑定的三款GLB；reference应从同一资产生成。
- 具体case的 source / move / insert / replace / move_replace fixed_world_geometry、候选/接触/朝向/落点审图结论。
- 生产环境运行依赖（trimesh/scipy/imageio-ffmpeg/huggingface_hub、SAM2等），见 MODELS_AND_EXTERNAL_DEPENDENCIES.json。

这些大输入不会塞进Git。当前实际规范数据在青海：
`/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal`
以及HF数据仓库 FrontRear/VLMdataset 的 StarReal 目录。Source6000D仍只读；处理和输出在青海，不在Mac建立大镜像。

case_profiles/83是实际生产配置示例，含历史绝对路径，需明确映射到所持有的数据。生产快照脚本保留原路径与assert防重入；不得直接在正式目录全批跑一遍。复制到新有限revision/新输出目录，按原owner和锁顺序执行。未通过具体几何/49帧图审时，代码成功退出也不能进入正式发布。

## 本次导出验证

validation/EXPORT_VALIDATION.json记录：
1. 源文件逐SHA与原生产副本一致、冻结数学版本未变。
2. 可移植核心的语法/导入与C++重新编译。
3. C++逐射线结果对NumPy参考实现的精确比较。
4. 实际case83相机和固定世界几何下，1280×704/AA2的一帧两版PCF与正式已生成PNG逐像素比较。
5. 从实际case83既有条件恢复support，验证两类纯色/强度raw数组与冻结生产fields逐项一致。
6. Remove原灯非空support、signed negative gamma、只在reader归一化、纯色foreground的组件核验。
7. 凭据/大媒体/权重排除扫描。

这是代码导出与有限真实组件的验证，不是重新验收59组，也不是证明所有历史试验脚本都可脱离原输入直接运行。
