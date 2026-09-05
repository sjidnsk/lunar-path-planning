# 面向车体包络与跟踪误差的亚栅格地形风险校准与巡视器路径规划

**英文题目：** Calibrated Sub-Grid Terrain Risk for Footprint- and Tracking-Aware Rover Path Planning<br>
**文稿性质：** 理论方法与预注册验证方案稿<br>
**当前证据状态：** 尚未执行本文提出的新实验，因此本文可以作为期刊论文的方法学主稿和实验实施规范，不能作为已经完成实证验证的投稿终稿。

## 摘要

栅格高程图把连续地形经过有限分辨率、固定栅格相位与方向、传感噪声及配准误差压缩为离散表示。现有可通行性方法通常直接在该表示上计算坡度、台阶、粗糙度或障碍掩膜，并据此规划车辆中心或膨胀后的车体路径。这一做法没有回答同一栅格值可能对应多种亚栅格真实地形的问题，也不能把单格置信度自然解释为整条路径的安全概率。本文提出一种面向 2.5D 几何可通行性的分层建模方法：首先将栅格化写成显式依赖分辨率、平移相位与栅格方向的 change-of-support 观测过程；继而从配准后的高精度连续地形参考中构建按地貌、观测质量、分辨率、相位和方向条件化的整场残差库，并分离扫描级配准共模误差；随后将地形残差与位置、航向及闭环跟踪误差共同传播到方向相关的车体扫掠极限状态。决策采用四层合同：确定性已知危险预筛、经场景级校准包络上的鲁棒扫掠、通过路径间的经验条件风险价值排序，以及冻结规划算法后的独立任务级统计审计。本文证明单幅**线性格元平均图**对亚栅格危险不可识别，给出带 Lipschitz 裕量的有限采样连续证书条件，并给出独立任务验证下的二项失效率上界。最后，本文提出多分辨率、多相位、多方向、高精度真值、地貌分布外和闭环车辆试验协议。该方法的目标不是恢复唯一“真实亚栅格地形”，而是建立从栅格表示误差到车辆包络路径风险之间可校准、可拒绝和可审计的桥梁。

**关键词：** 栅格化误差；亚栅格地形；可通行性；车体包络；风险校准；CVaR；路径规划；行星巡视器

## 1. 研究问题与项目现状

### 1.1 专家问题的严格表述

专家指出的不是单纯“分辨率太低”，而是以下推理链存在断点：

\[
\text{真实连续地形}
\longrightarrow
\text{传感与制图}
\longrightarrow
\text{栅格高程图}
\longrightarrow
\text{可通行指标}
\longrightarrow
\text{车辆路径是否安全}.
\]

若不显式描述前两个箭头造成的信息损失，那么在栅格上计算得再精确，也只是“栅格模型内部的精确”，不等于真实场景中的车辆可通行性。特别地，**栅格尺寸小于车体尺寸并不能消除该问题**：车辆足迹虽覆盖多个格元，但窄沟、尖石、坑缘、格内正负起伏相消、栅格边界相位变化以及跨格相关误差仍可能造成假安全。

### 1.2 当前项目：部分处理，但尚未解决

当前项目已经具备较好的几何安全基础，但没有完成本文所定义的栅格化不确定性建模：

1. 探索主线适配器仍调用 Python `AStarPlanner`（`src/lunar_exploration_ppo/integrations/path_planner_adapter.py:12,146`）。A* 搜索的是车辆中心点，上游当前以车体半对角线约 \(0.4216\,\mathrm m\) 加 \(0.10\,\mathrm m\) margin 的旋转不变圆包络，以及 unknown buffer，生成二值掩膜；它考虑了车辆尺寸，但不是方向相关矩形扫掠，且尚未接入 V3。
2. Hybrid A* 已使用连续 \((x,y,\theta)\) 状态和约 \(0.612\,\mathrm m\times0.580\,\mathrm m\) 的方向相关矩形足迹，但它是 opt-in 的 pose/path-cost source，不是默认执行规划器；运动按有限时刻回放，每个姿态的矩形足迹按格心近似栅格化。
3. C++ V3 是独立 opt-in 能力，不替换默认 A*、不连接执行器（`path-planner/cpp/docs/interface-and-safety-boundary.md:27-33`）。
4. V3 地图输入结构可携带高程、法向、粗糙度、硬障碍、标量 confidence 和可选 ESDF，但没有高程方差、空间协方差、栅格相位不确定性或亚栅格后验（`path-planner/cpp/include/lunar_path_planner/v3/map/immutable_snapshot.hpp:62-80`）。当前 `SafeProjection` 从 known/hard-obstacle 重新计算二维 ESDF，并不把输入 ESDF 当作车底—地形净空。
5. 对当前轮式 profile，硬筛选实际包括 known、hard obstacle、`confidence>0`、与航向无关的标量 \(\arccos(n_z)\) 坡度、二维障碍净距和正速度；粗糙度与台阶上限均未启用。最低 confidence 参数虽为 0，但 confidence 等于 0 仍会拒绝；`1-confidence` 只进入启发式非致命风险，不能解释成校准后的安全概率（`path-planner/cpp/src/map/safe_projection.cpp:367-413,591-668`）。
6. V3 足迹检查覆盖多边形顶点所在格及中心位于多边形内的格，尚非完整多边形—格元相交。primitive-chain 分支按分辨率四分之一和航向变化 0.05 rad 取有限样本，并在起终点之间线性插值；spline/spin 分支沿各自参数式取 \(2^d\) 个固定样本。所有分支目前都没有 Lipschitz 连续证书（`path-planner/cpp/src/wheel/wheel_sweep_validator.cpp:192-247,396-540`）。
7. 轮式扫掠只变换 XY 多边形，配置中的 z 上下界尚未用于车体—地形或车底净空；当前误差集合也未形成“运行时 position/yaw/tracking 误差包含于认证集合”的完整轮式合同。

因此，对“项目是否已经解决专家问题”的准确回答是：**只解决了一部分。默认探索主线仍是“车辆中心搜索＋旋转不变圆包络掩膜”；方向相关矩形包络只存在于 opt-in 的 Hybrid A*/V3 能力中，尚未成为默认规划与执行链。现有能力可对检测到的未知、碰撞和资源不足拒绝输出，但有限采样和近似格元相交仍可能漏检，更未建立栅格生成误差、空间相关亚栅格残差、栅格相位与路径级风险之间的数学和实验闭环。**

## 2. 文献证据与研究空白

DEM 分辨率、精度和栅格方向会改变坡度等派生量，[Thompson 等](https://doi.org/10.1016/S0016-7061(00)00081-1)与[Zhou 和 Liu](https://doi.org/10.1016/S0098-3004(04)00016-0)已给出系统证据。DEM 误差还具有空间自相关，并会传播到坡度、坡向和流域等派生量；把格元误差设为相互独立并不普遍成立，[Oksanen 和 Sarjakoski](https://doi.org/10.1016/j.cageo.2005.02.014)以及[Hugonnet 等](https://doi.org/10.1109/JSTARS.2022.3188922)支持这一点。

机器人领域已经有包含定位不确定性的概率高程图[Fankhauser 等](https://doi.org/10.1109/LRA.2018.2849506)、使用不确定性与 CVaR 的风险地形规划框架[STEP](https://doi.org/10.1109/TFR.2024.3512433)、基于高分辨率地形和车辆仿真的多目标可通行学习[Wallin 等](https://doi.org/10.1016/j.jterra.2022.04.002)，以及多量程、多分辨率的学习式可通行图[RoadRunner M&M](https://doi.org/10.1109/LRA.2024.3490404)。风险规划中的机会约束、CVaR 和分布鲁棒优化也已有成熟基础，例如[Blackmore 等](https://doi.org/10.1109/TRO.2011.2161160)、[Rockafellar 和 Uryasev](https://doi.org/10.21314/JOR.2000.038)及[Hakobyan 和 Yang](https://doi.org/10.1109/TRO.2021.3106827)。

直接竞争工作还包括面向行星车 DEM 不确定性的鲁棒路径设计[Inoue 和 Adachi](https://doi.org/10.2322/jjsass.70.208)、保留地形随机性的三角网格表示[Lombard 和 van Daalen](https://doi.org/10.1016/j.robot.2020.103449)，以及显式区分地面和地上障碍状态的 2.5D 概率地形图[PTS-Map](https://doi.org/10.1109/LRA.2024.3519859)。这些工作进一步说明，本文不能把“考虑 DEM 不确定性”或“概率 2.5D 地图”本身作为首次贡献。

因此，本文不能把“概率地图”“CVaR”“多分辨率”或“车体碰撞检查”单独宣称为创新。可辩护的研究空白是：

> 现有工作较少把真实制图算子的 change-of-support 零空间、固定栅格相位、空间相关的亚栅格残差、方向相关车体扫掠与冻结规划算法的任务级风险审计放进同一可验证合同。

本文的潜在贡献是“表示误差到车辆风险的可验证桥接”，是否具备最终新颖性仍需在投稿前完成系统综述和相邻方法复现，不能仅凭本轮定向检索声称“首次”。

## 3. 研究范围与基本假设

第一篇论文严格限定为**不确定性 2.5D 几何可通行性**，不宣称完整物理可通行性。

**项目范围声明：** 轮式平台的滑移控制、轮土接触控制及沉陷补偿不属于本项目范围。本文不建立相关动力学或控制模型，不设计相应算法，不设置相关验收指标，也不把它们列为本项目的后续交付项；下文仅将其作为本文几何结论不能外推覆盖的外部边界。

- **A1 单值地形：** 研究区域是静态单值高程场 \(z(x,y)\)，不含悬垂结构；未解析锐跳变标为未知。
- **A2 准静态刚性：** 车辆低速，地面在载荷下近似刚性，不建模沉陷、滑移和塑性变形。
- **A3 已知车辆几何：** 完整刚性三维车体包络、车底/轮位/支撑几何及纵坡、横坡、台阶和净空阈值已知；横滚、俯仰由预注册的局部刚性支撑平面规则近似。
- **A4 同步误差管已标定：** 位置、高度、横滚、俯仰、航向和闭环跟踪误差的**整条轨迹同步管**，在声明的速度、控制器和状态时效范围内经独立标定，并保留时空相关性；逐时刻边际区间不满足本假设。
- **A5 场景级可交换性：** 对每个预注册的离散 stratum，校准场景与未来分布内场景在冻结的传感器和制图流程下严格可交换。独立单位是完整场景，不是格元或重叠 patch；若工程证据只能支持近似可交换，则本文的精确 conformal 覆盖与后续概率界均不签发，只报告敏感性结果。
- **A6 连续证书条件：** 连续区域的几何间隙函数具有已知正则性界；没有该界、存在未解析跳变或资源耗尽时，验证结果为“不确定”，而不是“安全”。

## 4. 从连续地形到栅格观测

### 4.1 显式栅格化算子

令 \(\Omega_g,\Omega_w\subset\mathbb R^2\) 分别为制图坐标域和参考世界坐标域，\(Z:\Omega_w\to\mathbb R\) 为真实连续高程。在制图坐标中，分辨率为 \(h\)、平移相位为 \(o\)、栅格方向为 \(\varphi\) 的格元为

\[
C_i^{h,o,\varphi}
=o+Q_\varphi h\bigl(i+[0,1)^2\bigr),
\]

其中 \(Q_\varphi\) 是二维旋转矩阵。用 \(A_{h,o,\varphi}\) 表示作用于制图坐标连续场 \(f:\Omega_g\to\mathbb R\) 的实际制图算子，而不是默认其为中心采样。例如面积平均为

\[
(A_{h,o,\varphi}f)_i
=\frac{1}{|C_i^{h,o,\varphi}|}
\int_{C_i^{h,o,\varphi}}f(u)\,\mathrm du.
\]

令 \(T_\xi:\mathbb R^2_{\mathrm{grid}}\to\mathbb R^2_{\mathrm{world}}\) 为栅格坐标到参考世界坐标的扫描级刚体配准变换。定义垂直基准偏置 \(b\) 的正方向为“从地图高程加到世界参考高程”，于是部署栅格写为

\[
R=A_{h,o,\varphi}\!\left[(Z-b)\circ T_\xi\right]+\eta,
\]

其中 \(\eta\) 表示测量和重建噪声，因而后续重构中的“\(+b\)”与此生成式符号一致。中心采样、最大值、滤波后均值或点云栅格化对应不同的 \(A\)，必须按真实流水线复现。水平平移和航向误差属于坐标变换 \(T_\xi\)，不能当作一般的加性高程噪声。

令 \(B_{h,o,\varphi}\) 为规划时使用的连续重构算子。在世界坐标中，候选真实地形场景表示为

\[
Z_{\xi,b,U}(s)
=
\bigl[B_{h,o,\varphi}R\bigr]\!\left(T_\xi^{-1}s\right)
+b+U_{h,o,\varphi}(s).
\]

这里 \(U_{h,o,\varphi}\) 是在完成全局刚体配准和垂直基准校正后剩余的空间相关亚栅格残差。为避免重复计数，\((\xi,b,U)\) 作为一个联合场景抽样；同一份水平配准误差不得再加入车辆跟踪误差。二值障碍不得与高程共用无约束线性生成式，应另定义集合值障碍域 \(O_{\xi,U}\) 或经阈值约束的占据变量。

### 4.2 高精度真值配对与整场残差

对第 \(n\) 个独立地形场景，高精度参考地形为 \(Z_n^{\mathrm{ref}}\)。用不参与最终误差评价的控制点估计全局 \(T_{\hat\xi_n}\) 和 \(\hat b_n\)，让真实部署流水线生成 \(R_n^{h,o,\varphi}\)，再定义

\[
U_n^{h,o,\varphi}(s)
=Z_n^{\mathrm{ref}}(s)
-\bigl[B_{h,o,\varphi}R_n^{h,o,\varphi}\bigr]\!\left(T_{\hat\xi_n}^{-1}s\right)
-\hat b_n.
\]

用保留控制点估计 \(T_{\hat\xi_n}\) 和 \(\hat b_n\) 的剩余误差分布，并与 \(U_n\) 联合保存；这样把全局坐标扭曲、垂直偏置和局部残差分开定义，而不是假定三者独立。每个残差场景还必须保存：物理坐标、栅格原点、分辨率、制图算子版本、传感器和配准质量、粗尺度坡度/坡向/曲率/粗糙度、地貌类别以及相关长度估计。应用规则是：

1. 相同 \(h\)、\(o\)、\(\varphi\) 和 \(A/B\)；
2. 主分析只在预先固定的离散地貌/观测 strata 内抽样和校准；连续加权匹配只作不带精确覆盖声明的敏感性分析；
3. 保留残差和栅格边界的相对位置；
4. 仅在经检验近似各向同性时允许旋转，禁止任意缩放；
5. 整条路径抽取覆盖完整扫掠走廊的连续场，不逐格或逐段独立抽样；
6. 找不到匹配上下文时标为 OOD 并拒绝签发校准声明。

### 4.3 场景级同步校准包络

整场残差样本用于保留相关性的软风险评估；硬层另需同时覆盖空间位置和多种几何量的校准包络。定义完整几何场景状态

\[
\omega=(Z_{\xi,b,U},O_{\xi,U}),
\]

并在独立校准集上对高程、方向坡度、台阶、障碍有符号距离和净空构造归一化误差。所有校准场景使用预先固定、面积与形状相同且足以覆盖最大声明路径走廊的物理评价域 \(\mathcal D_{\mathrm{cal}}\)；超出该走廊尺寸的在线请求标为 OOD。以每场景最大值作为非一致性分数：

\[
S_n=\max_{s\in\mathcal D_{\mathrm{cal}},\,m\in\mathcal M}
\frac{|r_{n,m}(s)|}{\hat\sigma_m(s,c_n)+\epsilon_0}.
\]

设独立校准场景数为 \(n_{\mathrm{cal}}\)，排序后分数为 \(S_{(1)}\le\cdots\le S_{(n_{\mathrm{cal}})}\)，并定义 \(S_{(n_{\mathrm{cal}}+1)}=+\infty\)。对预注册覆盖误差 \(\beta_Z\)，取

\[
k=\left\lceil(n_{\mathrm{cal}}+1)(1-\beta_Z)\right\rceil,
\qquad q_{1-\beta_Z}=S_{(k)}.
\]

在校准开始前冻结 \(\hat\sigma_m\)、\(\epsilon_0\)、特征、strata、评价域和分数函数，并构造

\[
\mathcal H_{1-\beta_Z}(R,c)=
\left\{\omega:
|r_m(s;\omega,R)|
\le q_{1-\beta_Z}
\bigl[\hat\sigma_m(s,c)+\epsilon_0\bigr],
\ \forall s\in\mathcal D_{\mathrm{cal}},\
\forall m\in\mathcal M
\right\}.
\]

这相当于把每幅完整地形场视为一个结构化响应，并以全场最大误差作为 split-conformal 分数；相关的有限样本边际覆盖基础见[Lei 等](https://doi.org/10.1080/01621459.2017.1307116)。每个预先固定的离散 stratum 单独校准，且只在该 stratum 内声明场景级边际覆盖。连续地貌条件下的加权匹配不具有这里的精确有限样本条件覆盖。为保持几何一致性，\(\mathcal H\) 同时包含连续高程、集合值障碍及其共同派生量；实现时可用满足正则性约束的连续场集合，或对其作保守的区间/障碍集合外包。它不覆盖 OOD 地貌，也不是对每个具体月面场景的确定性真值保证。

## 5. 车辆包络、执行误差与路径极限状态

### 5.1 实际车体扫掠

名义平面路径为 \(q(t)=(p(t),\psi(t))\)。令 \(\mathcal B\subset\mathbb R^3\) 为不可变的完整三维刚性车体包络，并令 \(\mathcal B_{\mathrm{under}}\) 和 \(\mathcal W\) 分别表示车底与轮位/支撑几何。对地形场景 \(\omega\)，预注册的支撑平面规则由 \(q(t)\) 给出名义刚体位姿 \(X_\omega(t)\in SE(3)\)。完整几何误差状态写为

\[
e(t)=
(e_x,e_y,e_z,e_{\mathrm{roll}},e_{\mathrm{pitch}},e_{\mathrm{yaw}})(t),
\]

其中垂向、横滚和俯仰误差包括刚性支撑平面拟合残差及跟踪误差。令 \(\Delta X(e(t))\in SE(3)\) 为按冻结的坐标系与左右复合约定构造的误差变换。实际三维车体及其 XY 投影为

\[
\mathcal B_t(\omega,e)=X_\omega(t)\Delta X(e(t))\mathcal B,
\qquad
F_t(\omega,e)=\Pi_{xy}\mathcal B_t(\omega,e).
\]

\(F_t^{\mathrm{under}}(\omega,e)\) 与 \(W_t(\omega,e)\) 分别由 \(X_\omega(t)\Delta X(e(t))\mathcal B_{\mathrm{under}}\) 和 \(X_\omega(t)\Delta X(e(t))\mathcal W\) 的 XY 投影得到。若实现仍使用固定二维多边形 \(F^{\mathrm{cert}}\)，必须证明它外包声明域内所有 \((\omega,e)\) 的上述投影；名义姿态下的二维足迹不能代替这一联合外包。

令 \(E(t)\subset\mathbb R^6\) 为各时刻允许集合，并定义整条轨迹的同步误差管

\[
\mathcal E=
\left\{e(\cdot):e(t)\in E(t),\ \forall t\in[0,T]\right\}.
\]

其覆盖声明必须是 \(\Pr\{e(\cdot)\in\mathcal E\}\ge1-\beta_E\)，而不是每个时刻分别满足边际覆盖。对校准地形集合 \(\mathcal H\)，鲁棒 XY 扫掠为

\[
F_t^{\mathrm{rob}}=
\bigcup_{\substack{\omega\in\mathcal H\\e(\cdot)\in\mathcal E}}
F_t(\omega,e),
\qquad
\mathcal S_\pi^{\mathrm{rob}}=
\bigcup_{t\in[0,T]}F_t^{\mathrm{rob}}.
\]

地图配准误差必须通过 \(\omega\) 与车辆状态误差共同进入扫掠，航向误差不能只用一个未经证明的平移半径替代。若启用车底垂向间隙约束，则车辆轮位/支撑几何、支撑平面选择规则、刚性或允许的有限悬架几何，以及 \(e_z/e_{\mathrm{roll}}/e_{\mathrm{pitch}}\) 误差界都必须进入同一 \(\mathcal E\)；缺少任一项时不签发车底间隙结论。

### 5.2 几何极限状态

本文统一采用“极限状态大于 0 表示违反”的符号。令 \(\mathcal R_t(\omega,e)\) 为 \(X_\omega(t)\Delta X(e(t))\) 的旋转块，\(a_1=(1,0,0)^\top\)，实际车头与横向单位向量定义为

\[
d_t(\omega,e)=
\frac{\Pi_{xy}\mathcal R_t(\omega,e)a_1}
{\|\Pi_{xy}\mathcal R_t(\omega,e)a_1\|},
\qquad
d_t^\perp(\omega,e)=
\begin{bmatrix}
0&-1\\1&0
\end{bmatrix}d_t(\omega,e).
\]

声明的横滚/俯仰范围必须保证分母严格为正。先定义具有物理单位的原始极限状态：

\[
\widetilde g_{\parallel}(t)=
\sup_{s\in F_t(\omega,e)}
\left|d_t(\omega,e)^\top\nabla Z_\omega(s)\right|
-\tan\theta_{\parallel,\max},
\]

\[
\widetilde g_{\perp}(t)=
\sup_{s\in F_t(\omega,e)}
\left|d_t^\perp(\omega,e)^\top\nabla Z_\omega(s)\right|
-\tan\theta_{\perp,\max},
\]

\[
\widetilde g_{\mathrm{step}}(t)=
\sup_{\substack{s,s'\in W_t(\omega,e)\\\|s-s'\|\le r_w}}
\left|Z_\omega(s)-Z_\omega(s')\right|-h_{\max},
\]

\[
\widetilde g_{\mathrm{clear}}(t)=c_{\min}-
\inf_{s\in F_t^{\mathrm{under}}(\omega,e)}
\bigl[H_b(t,\omega,e;s)-Z_\omega(s)\bigr],
\]

\[
\widetilde g_{\mathrm{obs}}(t)=c_{\mathrm{obs}}-
\operatorname{sdist}\!\left(F_t(\omega,e),O_\omega\right).
\]

这里 \(\operatorname{sdist}\) 在集合分离时为正、接触时为 0、相交时为负，并要求 \(c_{\mathrm{obs}}>0\)；\(H_b\) 是实际三维车底下表面在水平位置 \(s\) 的最低高度，由同一个刚体变换、预注册支撑平面算法和不可变车底几何得到。未知区域只在已知危险预筛层中离散拒绝，不进入连续 Lipschitz 证书或 CVaR。由于上述连续量具有不同单位，不能直接用其数值大小比较风险。对每个约束指定预注册的正常数尺度 \(s_k\)，定义无量纲极限状态 \(g_k=\widetilde g_k/s_k\)。路径级最大违反为

\[
G_\pi(\omega,e(\cdot))=
\sup_{t\in[0,T]}
\max_{k\in\mathcal K}g_k(t;\pi,\omega,e(\cdot)).
\]

只有 \(G_\pi\le0\) 才表示在本文假设下的 2.5D 几何通过。它不等价于无滑移、无沉陷、动力学稳定或任务成功。

## 6. 四层风险决策合同

### 6.1 P：确定性已知危险预筛

未知区域、地图外、已知硬障碍、输入缺层、时间戳过期、配置 hash 不一致和显式阈值超限均直接拒绝。该层只能称“已知地图预筛选”，因为名义低分辨率图本身可能漏掉亚栅格危险。

### 6.2 R：校准集合上的鲁棒几何扫掠

对致命几何约束要求

\[
\sup_{\omega\in\mathcal H_{1-\beta_Z}(R,c)}
\sup_{e(\cdot)\in\mathcal E}
G_\pi(\omega,e(\cdot))\le0.
\]

这表示路径在声明的地形校准集合和执行误差管内通过。更严格地，若在声明域和冻结流水线下分别有

\[
\Pr\{\omega\in\mathcal H_{1-\beta_Z}(R,c)\}\ge1-\beta_Z,
\qquad
\Pr\{e(\cdot)\in\mathcal E\}\ge1-\beta_E,
\]

则由失败事件包含于两个覆盖失效事件的并集，有

\[
\Pr\{G_\pi(\omega,e(\cdot))>0\}\le\beta_Z+\beta_E.
\]

该并集界不要求两类误差独立，但要求 A5 的严格场景级可交换性对部署任务成立、\(\mathcal E\) 覆盖的是整条轨迹而非逐时刻边际事件，并且规划路径确实对整个 \(\mathcal H\times\mathcal E\) 鲁棒通过。由于 \(\omega\) 已联合包含配准变换、垂直偏置、连续高程和集合值障碍，不再另加一份地图配准失效概率。若误差管是合同内的确定性界，可取 \(\beta_E=0\)。完整场景同步集合覆盖后，任何只依赖 \(R\) 和该集合选择的自适应路径在覆盖事件内均受同一鲁棒结论约束，无须按候选路径数量再次校正；如果集合只在预设路径点上覆盖，则此结论不成立。缺少这些条件时，不能把校准水平简写成车辆失败概率。无法计算最坏情况、正则性不足或确定性的细分/工作记录/内存上限耗尽时返回 `validation_inconclusive`。

### 6.3 S：整场残差上的经验尾部风险排序

只在通过 P/R 的候选集合 \(\Pi_{\mathrm{robust}}\) 中评估经验尾部风险。对 \(M\) 个完整场景实现 \(G_{\pi,1},\ldots,G_{\pi,M}\)，定义

\[
\widehat{\operatorname{CVaR}}_{\alpha_{\mathrm C}}(G_\pi)
=
\min_{\tau\in\mathbb R}
\left[
\tau+
\frac{1}{M(1-\alpha_{\mathrm C})}
\sum_{j=1}^{M}(G_{\pi,j}-\tau)_+
\right].
\]

风险门槛 \(r_{\max}\) 在最终测试前冻结，先筛选

\[
\Pi_{\mathrm{risk}}
=
\left\{\pi\in\Pi_{\mathrm{robust}}:
\widehat{\operatorname{CVaR}}_{\alpha_{\mathrm C}}(G_\pi)
\le r_{\max}\right\},
\]

再按 \(\bigl(\widehat{\operatorname{CVaR}}_{\alpha_{\mathrm C}},C\bigr)\) 的字典序选择路径。若 \(\Pi_{\mathrm{risk}}=\varnothing\)，返回 `RISK_BUDGET_EXCEEDED`。因此 P/R 的致命违反和 S 层的预注册尾部风险门槛都不能被低路径成本抵消。

每次 Monte Carlo 实现必须联合抽取一个覆盖整条走廊和完整时间域的 \((\xi,b,U,O,e(\cdot))\) 场景，不得逐格或逐时刻独立相乘：

\[
P(\text{路径安全})\ne
\prod_iP(\text{第 }i\text{ 格安全}).
\]

CVaR 是无量纲最大几何违反的尾部排序量，不是碰撞概率。若要使经验尾部至少含 30 个场景级观测，必须满足 \(M(1-\alpha_{\mathrm C})\ge30\)；该经验要求只改善数值稳定性，不构成安全证明。

### 6.4 V：冻结规划算法后的独立统计审计

训练残差模型、选择路径/超参数、校准阈值和最终测试必须使用场景隔离的数据集。最终审计对象是**冻结的完整规划算法在新任务上产生的任务级结果**，而不是从测试集挑出的最佳路径。对每条已接受路径，独立真值系统输出同步区间 \([L_\pi^{\mathrm{ref}},U_\pi^{\mathrm{ref}}]\)，并预先验证任务级失覆盖率 \(\beta_{\mathrm{ref}}\)：

\[
\Pr\left\{\text{accept}\Rightarrow
G_\pi^{\mathrm{true}}\in
[L_\pi^{\mathrm{ref}},U_\pi^{\mathrm{ref}}]\right\}
\ge1-\beta_{\mathrm{ref}}.
\]

预先区分潜在真实事件、可审计的保守首要事件及两个辅助事件：

\[
Y_{\mathrm{unsafe}}^{\mathrm{true}}
=\mathbf 1\{\text{accept 且 }G_\pi^{\mathrm{true}}>0\},
\qquad
Y_{\mathrm{unsafe}}
=\mathbf 1\{\text{accept 且 }U_\pi^{\mathrm{ref}}>0\},
\]

\[
Y_{\mathrm{accept}}
=\mathbf 1\{\text{规划器接受并输出路径}\},
\qquad
Y_{\mathrm{task}}
=\mathbf 1\{\text{任务未被真值确认安全到达并停车}\}.
\]

若真值区间跨越 0，则已接受路径保守计为 \(Y_{\mathrm{unsafe}}=1\) 和 \(Y_{\mathrm{task}}=1\)，不得从 \(K/N\) 删除或用补充任务替换。NO_ROUTE、HOLD 和规划验证 inconclusive 在 \(Y_{\mathrm{unsafe}}\) 中是安全弃权而非不安全接受，但在 \(Y_{\mathrm{task}}\) 中计为任务未成功；因此必须同时报告接受率，防止“全部拒绝”伪装成安全改进。条件于已接受路径的保守几何失效率另以接受任务数为分母报告。在真值区间同步覆盖条件下，由并集界有

\[
\Pr\{Y_{\mathrm{unsafe}}^{\mathrm{true}}=1\}
\le
\Pr\{Y_{\mathrm{unsafe}}=1\}+\beta_{\mathrm{ref}}.
\]

因此二项审计首先约束可观测的保守事件；若要外推潜在真实不安全接受率，必须再计入 \(\beta_{\mathrm{ref}}\)。

仅降低 \(Y_{\mathrm{unsafe}}\) 仍可能被“全部拒绝”投机满足，因此完整算法的成功判据采用**安全终点＋任务效用强制门控**。除首要安全假设通过外，还必须同时满足

\[
U_{1-\delta_g}\!\left(p_{\mathrm{task}}^{\mathrm{new}}\right)
\le r_{\mathrm{task,max}}<1,
\qquad
U_{1-\delta_g}\!\left(
p_{\mathrm{task}}^{\mathrm{new}}-p_{\mathrm{task}}^{\mathrm{base}}
\right)
\le\Delta_{\mathrm{task}},
\]

其中 \(U_{1-\delta_g}\) 是预注册的一侧上置信界，\(r_{\mathrm{task,max}}\) 是绝对任务失败率上限，\(\Delta_{\mathrm{task}}\) 是相对基线的非劣界。二者在 pilot 后、最终测试前冻结。任一门控失败时，只能报告安全—效用权衡，不能声称方法整体改进；全部 HOLD/NO_ROUTE 必然使 \(p_{\mathrm{task}}=1\)，因而不能通过绝对门控。

若在 \(N\) 个独立任务中对预注册事件观察到 \(K\) 次发生，则使用单侧 Clopper–Pearson 上界

\[
p_U=
\operatorname{Beta}^{-1}
(1-\delta_{\mathrm{test}};K+1,N-K),
\]

其中 \(K=N\) 时取 1。当 \(K=0\) 时：

\[
p_U=1-\delta_{\mathrm{test}}^{1/N}.
\]

因此，在 \(\delta_{\mathrm{test}}=0.05\) 且零事件时，把测试分布下的**所审计二元事件**发生率上界压到 5%、1% 和 0.1%，至少分别需要 59、299 和 2995 个真正独立任务。若目标是潜在真实不安全接受率 \(\varepsilon\)，则还须满足 \(p_U+\beta_{\mathrm{ref}}\le\varepsilon\)，上述数字仅在 \(\beta_{\mathrm{ref}}=0\) 或目标就是保守可观测事件时可直接套用。该结论依赖预先冻结、独立试验单位和同分布外推条件，[Clopper 和 Pearson](https://doi.org/10.1093/biomet/26.4.404)的区间不能修复测试集选模或重复地形伪独立。

## 7. 三个理论命题

### 命题 1：单幅线性格元平均图对亚栅格危险不可识别

若 \(A_{h,o,\varphi}\) 是线性格元平均算子，格元具有非空内部，且允许的地形函数类没有预先给定的统一幅值、梯度或曲率上界，则存在非零、连续或分片连续函数 \(w\in\ker A_{h,o,\varphi}\)，使得

\[
A_{h,o,\varphi}(Z+w)=A_{h,o,\varphi}Z,
\]

但 \(w\) 可在格元内部产生超过给定坡度或台阶阈值的局部危险。

**证明要点。** 在单元内部选取互不重叠的正、负光滑 bump，使二者积分相消，因而格元平均不变；在没有统一正则性上界时，通过压缩正 bump 的支撑宽度，可使局部梯度或高度差超过任意有限阈值。若物理函数类已有固定正则性界，该构造仍证明非唯一性，但不能声称危险幅值任意大。该命题也不直接覆盖最大值等非线性栅格算子。由此可见，单幅低分辨率均值栅格不可能唯一恢复全部亚栅格危险，必须引入参考真值、可验证先验或最坏情形集合。

### 命题 2：带裕量的有限采样可推出连续几何通过

令 \(\mathcal X\) 为紧致的完整“时间—足迹/车底参数”扫掠域。若对每个连续约束 \(k\)，\(g_k(x;\omega,e(\cdot))\) 对所有 \(\omega\in\mathcal H\)、\(e(\cdot)\in\mathcal E\) 具有一致的确定性 Lipschitz 上界 \(L_k\)，\(\mathcal N_\varepsilon\) 是 \(\mathcal X\) 的 \(\varepsilon\)-网，且

\[
g_k(x_i;\omega,e(\cdot))\le-L_k\varepsilon,
\quad
\forall x_i\in\mathcal N_\varepsilon,\quad
\forall\omega\in\mathcal H,\quad
\forall e(\cdot)\in\mathcal E,\quad
\forall k,
\]

则 \(\sup_{x,\omega,e(\cdot)}g_k(x;\omega,e(\cdot))\le0\) 对所有 \(k\) 成立。

**证明。** 任取 \(x,\omega,e(\cdot),k\)，存在网点 \(x_i\) 使 \(\|x-x_i\|\le\varepsilon\)。由一致 Lipschitz 性，\(g_k(x;\omega,e(\cdot))\le g_k(x_i;\omega,e(\cdot))+L_k\varepsilon\le0\)。

二值占据指示函数在边界不连续，不能直接赋予有限 \(L_k\)。障碍证书应作用于连续车体与闭合障碍集合的有符号距离；速度、角速度和足迹半径只足以界定该距离随位姿的变化。方向坡度还需要地形 Hessian 界，台阶需要跳变边界或局部变化界，车底间隙需要支撑平面映射与地形正则性界。缺少任一约束的 \(L_k\)、存在未解析跳变或只验证了错误的运动原语时，整套 \(G_\pi\) 只能称“高密度离散检查”或 inconclusive。

### 命题 3：独立任务测试给出有限样本失效率上界

对预先冻结的算法，在 \(N\) 个独立同分布任务上得到预注册二元事件，Clopper–Pearson 单侧上界满足至少 \(1-\delta_{\mathrm{test}}\) 的覆盖率；一般式和零事件公式如第 6.4 节。若路径、阈值或算法由同一最终测试集选择，或把同一场地的重叠路径当成独立任务，则命题条件不成立。

## 8. 分辨率、栅格相位与方向如何选择

“分辨率越高越好”并非一般定理。下面只给一个说明偏差—方差权衡的玩具上界：设一维高程 \(f\in C^3\)、\(|f'''|\le M_3\)、\(M_3,\sigma>0\)，使用两个点高程的中心差分；两端噪声独立、零均值、同方差 \(\sigma^2\)，且 \(\sigma\) 不随 \(h\) 改变。不含格元平均、空间相关、配准和相位误差时，

\[
\operatorname{MSE}(h)
\lesssim \frac{M_3^2h^4}{36}+\frac{\sigma^2}{2h^2},
\]

最小化该上界右端得到

\[
h_{\mathrm{RHS}}^*=\left(\frac{3\sigma}{M_3}\right)^{1/3}.
\]

它不一定最小化真实 MSE，只说明过小网格可能放大差分噪声，不能直接作为车辆地图分辨率。工程选择只在预注册有限集合 \(\mathcal H_{\mathrm{cand}}\times\mathcal O_{\mathrm{cand}}\times\Phi_{\mathrm{cand}}\) 中进行，并以车辆尺度的路径级不安全接受为主指标：

\[
h^*=\max\left\{h\in\mathcal H_{\mathrm{cand}}:\
\max_{\substack{o\in\mathcal O_{\mathrm{cand}}\\
\varphi\in\Phi_{\mathrm{cand}}}}
\operatorname{UCB}^{\mathrm{sim}}_{1-\delta_{\mathrm{sel}}}
\bigl(\Pr(Y_{\mathrm{unsafe}}=1\mid h,o,\varphi)\bigr)
\le\tau_{\mathrm{FN}},\;
\operatorname{LCB}^{\mathrm{sim}}_{1-\delta_{\mathrm{sel}}}
\bigl(\operatorname{Coverage}(h)\bigr)
\ge1-\beta_Z,\;
T_{95}(h)\le B
\right\}.
\]

即在满足不安全接受、同步覆盖和实时性预算的分辨率中选最粗者。上、下界必须对被比较的有限 \(h,o,\varphi\) 组合同时校正，同一场景的多相位/方向结果按场景聚类，不能当独立样本。ECE 可作描述性指标，但不能替代同步覆盖证据。

若世界坐标、地图原点和栅格轴已确定，\(o,\varphi\) 是固定设计量，只能报告敏感性、极差和所测试有限集合中的最坏值；只有当配准误差具有经标定的概率分布，或制图原点/方向由明确随机机制生成时，才能将其概率化。有限 \(4\times4\) 相位和四个方向不能证明连续 \(o,\varphi\) 的全局最坏情形；要作连续声明，还需相位/方向正则性证书或全局优化。没有可信分布时不能默认相位或方向均匀。

## 9. 可实现算法

### 9.1 离线阶段

```text
输入：独立高精度真值场景、真实制图流水线、车辆几何与误差标定数据
1. 对每个 h、固定/受控相位 o、方向 varphi 和地貌 stratum c 运行真实栅格化算子。
2. 用独立控制点估计全局变换 xi 与垂直偏置 b，计算连续整场残差 U，并保留三者联合样本。
3. 按独立地形场景划分训练、选择、校准、最终测试集。
4. 在训练集建立条件化残差场景库；禁止跨集合使用重叠 patch。
5. 在固定物理评价域上，以场景级最大非一致性分数和有限样本分位数构建同步几何场景集合 H_(1-beta_Z)。
6. 冻结候选生成、beta_Z、beta_E、alpha_C、r_max、尺度 s_k、OOD 规则、连续界来源和最终事件定义。
```

### 9.2 在线规划阶段

```text
输入：地图快照 R、uncertainty snapshot、车辆能力、当前状态及跟踪误差管
1. 校验 frame、map hash、resolution、origin、grid axes/orientation、calibration ref 和状态时效。
2. P：拒绝未知、越界、硬障碍、缺层和显式超限区域。
3. 生成方向相关候选路径。
4. R：用连续场区间/障碍集合外包，对完整场景集合和 6DoF 几何误差管取最坏扫掠。
5. 每个格元做完整多边形 SAT；每种真实原语按该约束的 L_k 做确定性分支界。缺界或资源上限耗尽即 inconclusive。
6. 对通过者联合抽取完整相关场景，计算经验 CVaR；先执行 r_max 风险筛选，再作安全优先的字典序排序。
7. 输出路径和绑定全部输入、最坏裕量、适用域及验证方法的条件性几何风险证书。
```

### 9.3 复杂度

设地图格元数为 \(N\)，一次足迹覆盖格元数约为 \(K_F=A_F/h^2\)，路径区间数为 \(S\)，场景数为 \(M\)，最大区间细分深度为 \(d\)。鲁棒投影约为 \(O(N)\)，单姿态完整多边形—方格 SAT 为 \(O(K_F)\)，场景风险评估为 \(O(MSK_F)\)，带分支界的连续复核最坏为 \(O(SK_F2^d)\)。全尺度精确 GP 的 \(O(N^3)\) 不宜作为第一版在线主线；可作为离线对照。

## 10. 与 V3 的最小接入设计

### 10.1 新输入合同

新增不可变 `TerrainUncertaintySnapshotV1`，至少包含：

```text
schema_version / content_ref / source_map_snapshot_ref
source_time / maximum_age / frame_id
grid_geometry_hash / width / height / resolution_m
origin_xy_m / grid_axes_or_orientation / fixed_grid_phase_xy_m
aggregation_operator_ref / reconstruction_operator_ref
calibration_ref / terrain_domain_id / predeclared_stratum_id
beta_Z / calibration_scene_count / conformal_score_ref / conformal_quantile
calibration_corridor_support_ref / whole_field_scenario_library_ref
height_interval / directional_slope_interval / step_interval
obstacle_signed_distance_or_set_envelope / calibration_valid_mask / ood_mask
per_constraint_regularity_ref  # L_k、适用域及推导版本
```

车辆 capability/algorithm config 另须提供不可变的 XY 足迹、车底几何、轮位/支撑几何、支撑平面模型、纵横坡/台阶阈值、`minimum_obstacle_separation_m` 与 `minimum_underbody_clearance_m`。这两个间隙必须分开：现有 V3 wheel 把示例 `minimum_clearance_m` 用作二维 ESDF 障碍 margin，并未实现车底垂向间隙。

运行时另绑定：6DoF 当前状态误差集合、预测跟踪误差管 ref、\(\beta_E\)、\(\alpha_{\mathrm C}\)、\(r_{\max}\)、状态时间戳和最大状态年龄。该快照不能复用当前 `confidence` 字段；uncertainty ref、风险配置与 support-model ref 必须进入 `SafeProjection` 缓存键。

### 10.2 几何修正

1. 以完整多边形—方格 SAT 替换“顶点格＋格中心”近似；可借鉴项目 V2 的严格二维分离思路。
2. 按真实直线、圆弧、原地旋转或样条参数方程验证，而不是用起终点弦线替代原语。
3. 对障碍有符号距离使用速度、角速度与足迹半径的运动学界；对方向坡度、台阶和车底间隙分别绑定地形 Hessian、跳变/局部变化和支撑映射正则性所得 \(L_k\)。缺少任一适用约束的 \(L_k\) 时，只能 dense-sampling 或 inconclusive。
4. 采用自适应二分；达到确定性的 subdivision、work-record 或 memory cap 时 fail-closed。现有 V3 没有内部 wall-clock deadline，本方案不暗示已有超时中断语义。
5. 6DoF 状态和跟踪误差共同进入 XY 足迹与垂向车底几何；当前误差若不包含于认证误差集合，立即拒绝。

### 10.3 输出证书与失败语义

`GeometricRiskCertificateV1` 不是部署认证，而是带适用域和覆盖条件的风险评估证书。它绑定 request、map、uncertainty、capability、algorithm、route/path、current-state 与 tracking-tube ref/hash，记录 \(\beta_Z,\beta_E,\alpha_{\mathrm C},r_{\max}\)、最小二维障碍分离、最小车底垂向间隙、最大纵/横坡与台阶、每项 \(L_k\) 来源、细分记录、OOD 状态和失败原因。至少区分：

- `UNCERTAINTY_BINDING_MISMATCH`
- `CALIBRATION_OUT_OF_DOMAIN`
- `CURRENT_ERROR_EXCEEDS_CERTIFIED_BOUND`
- `ROBUST_TERRAIN_INFEASIBLE`
- `SWEPT_COLLISION`
- `CONTINUOUS_PROOF_INCONCLUSIVE`
- `RISK_BUDGET_EXCEEDED`

在完成验证前，该能力保持 opt-in，不替换默认 A*、不连接 executor。

## 11. 高精度地形参考真值

### 11.1 参考系统

建议同时使用：

- 地面激光扫描或近距离结构光作为主几何测量；
- 摄影测量作为独立交叉重建；
- 全站仪/高精度控制点网络约束坐标框架与尺度；
- 对石块、台阶和沟槽使用量规或接触式测量做局部核验；
- 独立 6DoF 运动捕获或全站仪跟踪记录车辆真轨迹。

参考真值应保留为带测量不确定度的点云或三角网格，而不是先压成另一幅“更细栅格”。目标采样间距可初设为不大于 \(\min(h_{\min}/10,w_{\min}/5)\)，但必须通过继续加密/降采样收敛实验确认，而不能把“十倍”当普适定律。

### 11.2 真值质量控制

1. 建立与规划地图独立的控制点网络；
2. 至少两次独立重复扫描，报告配准残差、点间距和遮挡区；
3. 用保留控制点检验绝对坐标，不用参与配准的点做自证；
4. 显式估计参考面本身的协方差或区间；
5. 遮挡、低反射和无法分辨区域标记为真值未知；
6. 使用完全一致的 \(A_{h,o,\varphi}\) 从连续参考面生成不同分辨率、相位和方向的退化输入；
7. 记录原始数据、标定、坐标变换、版本和 hash，确保可复现。

## 12. 预注册实验与统计草案

本节是可执行的预注册骨架，不是假装已经完成的注册文件。先用训练/开发 pilot 确定下列数值，再生成带时间戳和 hash 的冻结注册件，之后才能打开最终测试集：

| 锁定项 | 注册规则 |
|---|---|
| 唯一首要安全假设 | 相对确定性 V3 基线，所提方法降低独立任务上的保守不安全接受事件 \(Y_{\mathrm{unsafe}}\) |
| 唯一首要终点 | 每个独立任务一次计数、truth-inconclusive 已接受路径按事件计入的 \(Y_{\mathrm{unsafe}}\) |
| 强制任务效用门 | \(U(p_{\mathrm{task}}^{\mathrm{new}})\le r_{\mathrm{task,max}}<1\)，且 \(U(p_{\mathrm{task}}^{\mathrm{new}}-p_{\mathrm{task}}^{\mathrm{base}})\le\Delta_{\mathrm{task}}\)；任一失败则不得声称整体改进 |
| 风险参数 | \(\beta_Z,\beta_E,\beta_{\mathrm{ref}},\alpha_{\mathrm C},r_{\max},s_k,\delta_{\mathrm{test}},\delta_g,r_{\mathrm{task,max}},\Delta_{\mathrm{task}}\) 在 pilot 后一次性冻结 |
| 工程门槛 | \(\tau_{\mathrm{FN}}\)、\(T_{95}\) 预算、最小有意义效应 \(\Delta_{\min}\) 由项目危害分析与 pilot 冻结，不在无数据时虚构数值 |
| 样本量 | 校准层每 stratum 至少满足 \(n_{\mathrm{cal}}\ge\left\lceil(1-\beta_Z)/\beta_Z\right\rceil\) 以取得有限 conformal 分位数；最终审计按目标 CP 上界计算，方法差异检验另做功效分析 |
| 基线公平性 | 固定软件版本、候选集、算力、调参预算和停止规则 |
| OOD | 在看最终结果前按地貌尺度、频谱/相关长度和观测质量定义；只报告退化与拒绝能力，不外推分布内风险界 |
| 多重检验 | 首要假设单独检验；所有次要假设预先分族并在族内使用 Holm 修正 |

### 12.1 数据划分

按地理上互不重叠的完整地形场景划分训练、模型选择、校准和最终测试集。同一母地形的重叠 patch、相邻路径或同一次地图重建不得跨集合，也不得被计为独立最终任务。至少保留一个完整地貌族只用于 OOD 测试。

### 12.2 因素矩阵

| 因素 | 最低设置 |
|---|---|
| 地形 | 刚性平缓起伏、刚性尖锐岩场、刚性沟槽/坑缘、刚性组合纵横坡；颗粒地形和滑移工况排除在本项目实验域之外 |
| 归一化分辨率 | \(h/L=\{1/32,1/16,1/8,1/4,1/2,1\}\) |
| 平移相位 | 每个分辨率至少 \(4\times4\) 个受控相位 |
| 栅格方向 | \(0^\circ,15^\circ,30^\circ,45^\circ\) |
| 危险宽度 | \(w/h=\{0.1,0.25,0.5,1,2\}\) |
| 误差注入 | 传感噪声、扫描级配准，以及位置/高度/横滚/俯仰/航向/跟踪误差的单独及联合条件 |

这里 \(L\) 是车辆特征尺度，可预注册为车宽或足迹对角线；主分析只使用一种定义，其余作为敏感性分析。

### 12.3 基线与消融

**基线：** 确定性 DEM 阈值、圆形膨胀、V3 式有限采样 XY 足迹、逐格独立高斯 Monte Carlo、空间相关 GP/随机场、整场残差、分位数风险、经验 CVaR、高分辨率真值 oracle。

**必须消融：**

- 逐格独立与整场相关残差；
- 是否按分辨率、相位、方向、地貌和观测质量条件化；
- 名义预筛与校准集合鲁棒层；
- 中心点、圆形和方向相关多边形足迹；
- 有限密集采样与带上界的自适应扫掠；
- 有无地图配准、位置、航向与跟踪误差；
- 均值、分位数与 CVaR；
- 分布内与地貌 OOD；
- 开环几何回放与独立实验 harness/既有控制器的闭环车辆验证；这不代表把 V3 接入项目 executor。

### 12.4 路径级事件与指标

一个独立 Bernoulli 任务定义为：独立地形场景、独立地图重建、预先固定的起终点和独立执行扰动下，对冻结规划器进行一次完整调用与验证。无路、HOLD 和 inconclusive 也属于任务结果，不能只保留产生路线的试验。

真值系统计算完整 \(G_\pi^{\mathrm{ref}}\) 区间，覆盖纵/横坡、台阶、二维障碍分离和启用时的车底垂向间隙。只有其同步真值区间上端 \(U_\pi^{\mathrm{ref}}\le0\)、实际 6DoF 姿态处于认证跟踪管、到达目标容差且完成几何安全停车，才判为几何成功；若区间跨越 0，则标为 truth_inconclusive。对已接受路径，它在首要分析中保守计为 \(Y_{\mathrm{unsafe}}=1\) 和 \(Y_{\mathrm{task}}=1\)，仍保留在原定分母中；可以另增补充 cohort，但不得替换该任务或混入冻结的首要 \(K/N\)。

首要指标是保守 \(Y_{\mathrm{unsafe}}\) 的任务级发生率；潜在 \(Y_{\mathrm{unsafe}}^{\mathrm{true}}\) 只通过 \(p_U+\beta_{\mathrm{ref}}\) 的并集上界报告。强制同时报告 \(Y_{\mathrm{accept}}\)、\(Y_{\mathrm{task}}\)、条件于接受路径的保守几何违反率和 truth_inconclusive 比例。次要指标包括假拒绝率、最小二维障碍分离、最小车底垂向间隙、最大真值纵/横坡与台阶、跟踪管越界率、到达/停车率、90/95/99% 同步覆盖率、CVaR、路径长度、规划时间及安全—效率 Pareto 前沿。Brier、NLL、ECE 只在另行预注册并冻结任务级概率预测量 \(\hat p_\pi\) 时报告；当前核心方法不依赖它们。

### 12.5 统计分析

- 同场景二分类对比采用 McNemar 或场景级配对 bootstrap；
- 跨地貌使用混合效应 logistic 模型或 cluster bootstrap；
- 置信区间以独立场景为聚类单位；
- 多重比较采用 Holm 修正；
- CVaR 必须满足 \(M(1-\alpha_{\mathrm C})\ge30\)，并报告场景级 bootstrap 区间；尾部次序统计量不被宣称为彼此独立；
- 最终风险审计预注册单侧 95% Clopper–Pearson 上界；
- 整体改进结论必须通过绝对任务失败率上限和相对基线任务失败率非劣两项一侧门控；安全终点通过但门控失败时不判成功；
- OOD 结果单独报告，不与分布内样本混池后宣称整体安全。

若只完成 12–20 次实车试验，它们只能支持机制和现实迁移证据，不能单独支持 5%/95% 的失效率声明。若要以零事件把**保守可观测事件**的单侧 95% 上界压到 5%，实车或明确定义的同类独立任务至少需要 59 次；若要对潜在真实不安全接受率作同样声明，还必须为 \(\beta_{\mathrm{ref}}\) 留出风险预算。仿真与实车不能无条件混池。

## 13. 多角色学术争辩与裁决

本稿由数学建模、机器人规划与证据审稿三个角色进行多轮立论和交叉质询。下表是内部方法审计记录，投稿时应移入补充材料或作者设计记录，不作为实证结果。

| 争议 | 初始意见 | 反方质疑 | 最终裁决 |
|---|---|---|---|
| 在线完整随机场还是经验场景库 | 完整 GP/DRO 更统一 | 在线代价高，且错误先验会产生虚假精度 | 主文采用条件化整场残差；GP 为基线，DRO 为扩展 |
| 硬层是否天然安全 | 硬阈值与软 CVaR 分开即可 | 在名义低分辨率图上的硬阈值仍会漏检 | 区分 P 名义预筛与 R 校准集合鲁棒扫掠 |
| V3 是否已有连续验证 | 接口和有界细分提供基础 | 当前是有限样本且弧线按弦插值 | 现状称 bounded dense-sampling；仅满足命题 2 才称连续证书 |
| 残差能否逐格抽样 | 逐格实现简单 | 破坏空间相关并低估长路径风险 | 每个实现抽取覆盖整条扫掠走廊的连续场 |
| 相位是否是随机量 | 多相位可做 Monte Carlo | 固定原点时没有概率含义 | 固定相位做敏感性/最坏分析；仅经标定配准误差可概率化 |
| 是否包含滑移与月壤力学 | 完整物理可通行模型通常会涉及该问题 | 本项目范围明确排除滑移控制，纳入后会改变研究问题 | 全文只声称刚性地形 2.5D 几何可通行；不建模、不验证，也不列为项目后续交付 |
| CP 是否证明任意路径安全 | 独立测试可给失效率上界 | 测试集选路和伪独立会使结论失效 | 审计冻结算法的任务级结果，预注册独立单位和样本量 |
| 低不安全接受率是否足够 | 首要安全终点可以单独比较 | 全部 HOLD/NO_ROUTE 可把事件率降为 0 | 增设任务失败率绝对上限与相对基线非劣的双门控；失败时不宣称整体改进 |

这轮争辩否决了四种常见但不成立的推论：整场残差不等于真实后验；低 CVaR 不等于低碰撞概率；格元小于车体不等于无栅格误差；有限采样次数多不等于连续安全证明。

## 14. 可发表主张、禁止主张与拒稿风险

### 14.1 在完成实验后可检验的主张

1. 显式建模分辨率、相位和空间相关残差，是否降低相对于高精度真值的路径级危险漏检。
2. 方向相关足迹与跟踪误差传播，是否改善不同栅格相位下的几何安全一致性。
3. 场景级校准包络是否达到预注册的同步覆盖率。
4. 整场 CVaR 排序是否在不显著增加假拒绝和计算开销的条件下降低尾部几何违反。
5. 冻结规划算法在独立分布内任务上的保守不安全接受率上界是否达到预注册门槛，并同时通过任务失败率的绝对与非劣效门控。

### 14.2 在现阶段禁止写入摘要或结论的主张

- “消除了栅格化误差”或“恢复了真实亚栅格地形”；
- “找到了普适最优分辨率”；
- “CVaR 保证碰撞概率”；
- “V3 已实现严格连续扫掠”；
- “XY 包络通过等于车辆物理可通行”；
- “适用于真实月面任务”或“达到部署认证”；
- 在没有独立试验和置信区间时声称优于基线。

### 14.3 主要拒稿风险

- 只向每格加入独立高斯噪声；
- 训练、选路、定阈值和最终检验复用同一场景；
- 把重叠 patch 或同一场地重复运行当独立样本；
- 只改变分辨率而不改变相位；
- 仍用车辆中心点代替方向相关包络；
- 没有高精度真值或地貌 OOD；
- 只报告平均路径长度，不报告路径级假安全、覆盖率与置信上界；
- 把未建模且不属于项目范围的轮土、滑移或沉陷现象纳入成果主张，或把几何结果扩写成完整物理可通行与月面任务成功率。

## 15. 预期贡献句与结论

建议投稿时使用如下贡献句：

> 本文将栅格化表述为显式依赖分辨率、平移相位与栅格方向的空间观测过程，从配准的高精度地形参考中建立条件化整场残差；随后将该残差与地图配准、位姿及跟踪误差共同传播至方向相关车体扫掠的路径级违反函数，通过已知危险预筛（P）、校准集合鲁棒扫掠（R）和经验整场 CVaR 门控（S）完成候选选择，并在场景隔离的多分辨率、多相位、多方向、地貌 OOD 与独立实验平台闭环几何试验中，对冻结规划算法实施任务级统计审计（V）。

本文给出的核心结论是方法论而非性能结论：寻找合适分辨率只是研究的一部分；更根本的任务是识别栅格化不可逆丢失的危险信息，以高精度参考真值刻画其条件分布或覆盖集合，再把这一不确定性传播到真实车体扫掠和路径级失败事件。当前项目有实现该研究的 V3 工程骨架，但尚未完成所需数据、模型、连续证书和独立验证。只有完成第 11–12 节的预注册实验，才能把本稿升级为符合期刊实证要求的投稿终稿。

## 参考文献

1. Thompson, J. A., Bell, J. C. & Butler, C. A. Digital elevation model resolution: Effects on terrain attribute calculation and quantitative soil-landscape modeling. *Geoderma* **100**, 67–89 (2001). [DOI](https://doi.org/10.1016/S0016-7061(00)00081-1)
2. Zhou, Q. & Liu, X. Analysis of errors of derived slope and aspect related to DEM data properties. *Computers & Geosciences* **30**, 369–378 (2004). [DOI](https://doi.org/10.1016/S0098-3004(04)00016-0)
3. Oksanen, J. & Sarjakoski, T. Error propagation of DEM-based surface derivatives. *Computers & Geosciences* **31**, 1015–1027 (2005). [DOI](https://doi.org/10.1016/j.cageo.2005.02.014)
4. Hugonnet, R. et al. Uncertainty analysis of digital elevation models by spatial inference from stable terrain. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing* **15**, 6456–6472 (2022). [DOI](https://doi.org/10.1109/JSTARS.2022.3188922)
5. Fankhauser, P., Bloesch, M. & Hutter, M. Probabilistic terrain mapping for mobile robots with uncertain localization. *IEEE Robotics and Automation Letters* **3**, 3019–3026 (2018). [DOI](https://doi.org/10.1109/LRA.2018.2849506)
6. Dixit, A. et al. STEP: Stochastic traversability evaluation and planning for risk-aware navigation; results from the DARPA Subterranean Challenge. *IEEE Transactions on Field Robotics* **2**, 81–99 (2024). [DOI](https://doi.org/10.1109/TFR.2024.3512433)
7. Wallin, E. et al. Learning multiobjective rough terrain traversability. *Journal of Terramechanics* **102**, 17–26 (2022). [DOI](https://doi.org/10.1016/j.jterra.2022.04.002)
8. Patel, M. et al. RoadRunner M&M—Learning multi-range multi-resolution traversability maps for autonomous off-road navigation. *IEEE Robotics and Automation Letters* **9**, 11425–11432 (2024). [DOI](https://doi.org/10.1109/LRA.2024.3490404)
9. Blackmore, L., Ono, M. & Williams, B. C. Chance-constrained optimal path planning with obstacles. *IEEE Transactions on Robotics* **27**, 1080–1094 (2011). [DOI](https://doi.org/10.1109/TRO.2011.2161160)
10. Rockafellar, R. T. & Uryasev, S. Optimization of conditional value-at-risk. *Journal of Risk* **2**, 21–41 (2000). [DOI](https://doi.org/10.21314/JOR.2000.038)
11. Hakobyan, A. & Yang, I. Wasserstein distributionally robust motion control for collision avoidance using conditional value at risk. *IEEE Transactions on Robotics* **38**, 939–957 (2022). [DOI](https://doi.org/10.1109/TRO.2021.3106827)
12. Clopper, C. J. & Pearson, E. S. The use of confidence or fiducial limits illustrated in the case of the binomial. *Biometrika* **26**, 404–413 (1934). [DOI](https://doi.org/10.1093/biomet/26.4.404)
13. Schwarzer, F., Saha, M. & Latombe, J.-C. Adaptive dynamic collision checking for single and multiple articulated robots in complex environments. *IEEE Transactions on Robotics* **21**, 338–353 (2005). [DOI](https://doi.org/10.1109/TRO.2004.838012)
14. Hengl, T. Finding the right pixel size. *Computers & Geosciences* **32**, 1283–1298 (2006). [DOI](https://doi.org/10.1016/j.cageo.2005.11.008)
15. Lei, J., G’Sell, M., Rinaldo, A., Tibshirani, R. J. & Wasserman, L. Distribution-free predictive inference for regression. *Journal of the American Statistical Association* **113**, 1094–1111 (2018). [DOI](https://doi.org/10.1080/01621459.2017.1307116)
16. Inoue, H. & Adachi, S. Path planning method for exploration rovers robust to uncertainty in elevation data. *Journal of the Japan Society for Aeronautical and Space Sciences* **70**, 208–214 (2022). [DOI](https://doi.org/10.2322/jjsass.70.208)
17. Lombard, C. D. & van Daalen, C. E. Stochastic triangular mesh mapping: A terrain mapping technique for autonomous mobile robots. *Robotics and Autonomous Systems* **127**, 103449 (2020). [DOI](https://doi.org/10.1016/j.robot.2020.103449)
18. Kim, D. W., Son, E. I., Kim, C., Hwang, J. H. & Seo, S. W. PTS-Map: Probabilistic terrain state map for uncertainty-aware traversability mapping in unstructured environments. *IEEE Robotics and Automation Letters* **10**, 1257–1264 (2025). [DOI](https://doi.org/10.1109/LRA.2024.3519859)
