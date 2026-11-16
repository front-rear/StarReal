# 必须保留的错误经验和当前边界

| 已出现的问题 | 正确处理 |
|---|---|
| move落在书、人物、桌沿或悬空，mask随bbox同样错误 | 看实际RGB和全49帧，确认空支撑面；只改有问题的操作/版本，合理旧位置保留 |
| 把PCF和bbox按二维区域硬叠 | 固定共享世界坐标和同一相机射线，网格/框按距离双向遮挡；普通版仍单独保留 |
| depth-aware只按谁最近着色，P/Q重叠丢黄色 | 对P/Q分别做PCF遮挡后取可见投影标签并集，重叠黄色，固定已批准renderer SHA |
| move Color把source/target两套不同颜色拼成青/品红 | 同一条Color条件前景必须是一个指定线性RGB常量；source_hint和请求颜色变体分开，不把诊断双色图当训练mask |
| Remove用空target推导空Color、无原灯条件 | target灯不存在正确，但另补source A/B的轮廓+bbox Color和负Intensity、原灯reference |
| reference丢失或图片与实际GLB不是同一资产 | 精确SHA绑定，按实际资产渲染，不用相似图片或AI图代替 |
| 相机移动时用2D bbox每帧硬贴、大小手调 | 固定世界几何用逐帧K/E重新投影，保持静态world位置 |
| 把单目相对尺度当米或把几何估计叫真实GT | 明确scale unknown、source hidden completion、novel-B unknown/estimated；保留uncertainty |
| confidence激活下限也当可靠场景遮挡 | 记录Native点头conf的实际来源/有效性，未知区单独输出，不能伪称测量可见 |
| 49个15FPS输入强转10FPS并clamp最后一帧 | 明确时间线，保留真实49帧/有证据的时序适配后重新核depth/K/E/SAM；不重复尾帧过验收 |
| 同视频ID或别名就复用另一段depth/SAM | 核真实RGB序列/源SHA/crop/timestamps；不同红光相位、照明、视角不可按索引混用 |
| 先包、先传、后补图审 | 具体revision先实际审图，再reader、发布；未审状态不能改accepted |
| 磁盘新代码就声称旧Pythonworker已采用 | 核实际进程、argv/claim和输出，不凭磁盘SHA推断内存版本 |
| 原灯在画外/被遮挡，只移动可见碎片到中央 | 保留可见/未知边界，不能把残缺片段当整灯；需要可靠source几何再推进 |

本次快照54/59，其中余下99、100、101、103、7578553没有虚报完成：有原灯不可见部分、原帧时序不一致、不同别名片段、原4K输入与估计几何一致性等尚需逐项闭环。仓库导出不替这些问题盖“已修复”章。

各组的颜色、强度是编辑请求条件，不是从视频定量测量出的灯具瓦数或光度。source_hint是可复现的观察颜色参考，仍受白平衡、曝光和反射影响。没有真实edited RGB或novel-B视角时不能补一张生成图冒充监督真值。

生产快照保留旧路径/历史阶段是为了审计。新的可移植入口只整理IO，不修改冻结renderer/Original训练语义；任何未来算法改变都要新revision并对实际数据重新比较，禁止静默替换HF现有包。单元/组件验证也不能替代实际位置合理性的视觉审查。
