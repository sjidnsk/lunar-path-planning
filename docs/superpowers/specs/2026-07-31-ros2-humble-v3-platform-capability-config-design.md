# ROS 2 Humble v3 平台能力配置设计

日期：2026-07-31

## 1. 目标

为 ROS 2 Humble 集成整理轮式、足式和飞跃式三类版本化平台能力配置。
配置以 Path Planner v3
`SafetyCapabilityProfile` 为唯一字段和约束标准，由 ROS 2 参数选择文件，
不把复杂能力对象平铺为节点参数。

## 2. 已确认边界

- 平台能力配置描述平台允许规划器依赖的静态能力。
- 平台当前位姿、速度和定位状态继续由
  `/localization/odometry`、`/localization/status` 和 TF 提供。
- 项目从运行状态计算当前误差界；能力配置中的
  `certified_state_error_bounds` 表示允许的认证上界。
- v3 输出带时间缩放的参考轨迹，因此保留速度、加速度、制动和姿态变化极限。
- v3 所需的认证运动原语及其扫掠几何引用属于能力合同，不能由宽松运行时默认值补齐。
- 规划器搜索分辨率、原语采样密度、资源预算和优化迭代数属于
  `PlannerAlgorithmConfig`。
- 自主探索的观察范围属于独立观察模型，不写入 v3
  `SafetyCapabilityProfile`。

## 3. 文件组织

```text
configs/platforms/v3/
├── README.md
├── wheeled_skid_steer_v3_example_v1.json
├── legged_body_v3_example_v1.json
└── hopper_ballistic_v3_example_v1.json
```

示例文件严格匹配：

```text
path-planner-v3-safety-capability-profile/v1
```

每份文件只包含：

```json
{
  "schema_version": "...",
  "content_ref": {},
  "content": {}
}
```

顶层禁止添加 `status`、`description` 或自定义 provenance 字段，以免违反
v3 `additionalProperties=false`。示例状态和替换规则统一写在目录 README。

## 4. 三类平台能力

### 4.1 轮式

当前轮式语义固定为滑移转向：

- 凸二维车体轮廓及高度挤出范围；
- 前进、倒退、原地旋转、加速、制动、偏航和横向加速度上限；
- 最大行驶曲率、最大坡度和最小净空；
- 前进/倒退直线、前进/倒退弧线、顺逆时针旋转和停止切换原语；
- 每个原语的认证扫掠几何引用；
- 轮式/足式形式的认证状态误差界。

不新增 `ACKERMANN`、`DIFFERENTIAL_DRIVE` 或 `OMNIDIRECTIONAL`
枚举，也不声称 Ackermann 可行性。

### 4.2 足式

当前足式语义固定为机体级规划：

- 固定机体参考点和三维凸包；
- 坡度、粗糙度、高程突变、沟壑、置信度、净空和机体高度阈值；
- 前后、横向、垂直和偏航速度区间；
- 线加速度和偏航角加速度；
- 机体平移和旋转原语及扫掠引用；
- `feasibility_scope=body_geometry_and_terrain_thresholds_only`；
- `footstep_feasibility_guaranteed=false`。

当前地图适配把沟壑宽度置为零。由于 v3 schema 仍强制要求
`maximum_gap_width_m`，示例配置使用零值并在 README 中禁止跨沟能力声明。

### 4.3 飞跃式

当前飞跃式语义固定为纯弹道飞跃：

- 三维机体凸包；
- 运动模型、代价模型和重力模型引用；
- 着陆坡度、粗糙度、平面残差、顶部/侧向净空和最小着陆面积；
- 最大发射速度与冲量、飞行时间区间、着陆速度与净空；
- 任意轴保守姿态角速度、角加速度和稳定等待时间；
- 飞跃式认证状态误差界；
- `translation_model=PURE_BALLISTIC_NO_INFLIGHT_TRANSLATION_CONTROL`。

## 5. 哈希和版本

`content_ref.content_hash` 必须等于顶层 `content` 对象经过 RFC 8785
JCS 规范化后的 SHA-256。

修改任何能力数值、几何、原语或引用时必须：

1. 创建新版本文件；
2. 增加 `content_ref.revision`；
3. 重新计算内容哈希；
4. 更新 ROS 2 参数中的期望哈希；
5. 重新执行 schema 和语义验证。

禁止在保留旧哈希的情况下修改内容。

## 6. ROS 2 生命周期

ROS 2 节点参数只提供：

- 配置包名；
- 包内资源路径；
- 期望内容哈希。

能力文件在 `configure` 阶段完成加载、schema 校验、哈希校验、引用解析和
v3 语义校验。节点进入 active 后配置冻结。运行中更换能力配置必须经过
`deactivate → cleanup/configure → activate`。

## 7. 示例与部署配置

仓库目前没有足式和飞跃式真实平台的认证参数，也没有 Scout Mini
全部动力学极限和误差界的认证来源。因此三份文件命名包含 `example`：

- 可以用于 schema、适配器和离线合同测试；
- 不得用于真实平台执行；
- 不得作为平台性能或安全认证证据。

真实部署配置必须由平台资料、试验或安全系统给出完整参数和全部引用对象。
