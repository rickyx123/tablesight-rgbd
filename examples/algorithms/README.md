# 核心算法：输入、输出与可运行示例

[项目首页](../../README.md) · [技术复盘](../../docs/技术复盘.md) · [源码导航](../../docs/运行说明.md)

下面三个例子直接调用 `reference/competition_snapshot/` 中的现有实现，没有重新写一套演示算法。它们演示桌面归属、深度证据和跨帧计数，使用**合成检测框与深度数组**，用于理解接口和验证行为，不是模型精度或比赛实测。

## 先运行一次

在仓库根目录执行，使用 Python 3.10 或更新版本：

```bash
python -X utf8 -m pip install -r examples/algorithms/requirements.txt
python -X utf8 examples/algorithms/table_scope.py --check
python -X utf8 examples/algorithms/depth_states.py --check
python -X utf8 examples/algorithms/temporal_count.py --check
```

每条命令输出 JSON；`--check` 同时与 `expected/` 中实际运行保存的输出比较，不一致时以非零状态退出。只想查看结果可去掉 `--check`。

已验证环境：Python 3.10.14、NumPy 2.2.6、OpenCV 4.13.0.92、Pillow 12.1.1，Windows。此依赖文件只用于这三个例子，不是完整板端环境；无需模型权重、相机、Ascend、OCR 模型或 Tk 窗口。

参考代码的 UI 类型导入链会加载 `tkinter`。标准 Windows Python 已验证可用；精简 Linux 环境若缺少该模块，需要另外安装与 Python 匹配的系统 Tk 包，但运行示例不会创建窗口。

`_snapshot.py` 会把参考源码目录加入搜索路径。其他程序中复用时，也需要将 `reference/competition_snapshot` 加入 `PYTHONPATH`，再导入 `national_competition`。`pip install -e .` 只安装独立工具包 `tablesight`，不会安装这份硬件源码快照。

## 共用约定

| 项目 | 约定 |
|---|---|
| 类别 | 固定 17 类；0=眼镜、1=胶带、2=遥控器、3=卷尺、16=TABLE，完整顺序见[类别配置](../../configs/class_contract.json) |
| 坐标 | `xyxy=[x1,y1,x2,y2]`，左上角为原点，x 向右、y 向下，像素单位，`x2>x1`、`y2>y1` |
| 对齐 | 检测框、桌面轮廓、深度数组必须对应同一参考图像和同一帧；模型 letterbox 坐标应先还原 |
| 置信度 | 浮点数，范围 0–1；类别阈值仍由参考源码的配置决定 |
| 深度 | 二维 NumPy 数组，单位 **mm**；0 或非有限值作为无效测量，不能当平面证据 |
| 输出 | 示例把 dataclass、tuple 和计数字典转为 JSON；JSON 中类别字典的键会显示为字符串 |

## 1. 桌面归属：哪些检测框可以进入计数？

示例：[table_scope.py](table_scope.py) · [实际函数 `scope`](../../reference/competition_snapshot/national_competition/runtime/timed_fusion.py) · [完整输出](expected/table_scope.json)

### 输入

```python
from national_competition.ui.competition_visualizer import UiDetection
from national_competition.runtime.timed_fusion import scope

detections = [
    UiDetection(16, 0.95, (100, 80, 540, 400)),  # TABLE
    UiDetection(3, 0.90, (250, 180, 300, 230)),  # 桌上卷尺
    UiDetection(3, 0.90, (560, 330, 610, 390)),  # 桌外卷尺
    UiDetection(3, 0.90, (105, 80, 125, 120)),   # 矩形框内、梯形后角外
]
quad = [(180, 100), (460, 100), (530, 390), (110, 390)]
table, accepted, rejected = scope(detections, "square", table_polygon=quad)
```

`UiDetection` 的参数顺序是**类别、置信度、坐标**，不同于其他文件中可能出现的 `Detection` 类型。`quad` 是沿桌沿依次排列的顶点，不应自交。本例将桌面定位视为上游输入，并未运行深度桌沿估计。

### 输出

返回三元组 `(table, accepted, rejected)`：`table` 为 `UiDetection` 或 `None`，后两项为物品行字典列表。该例接受 `[250,180,300,230]`，拒绝另外两个物品框，保留的行中 `depth_state` 初始化为 `UNKNOWN`。

判断依据是框底部附近的参考点，而不是“物品框与桌面有交叠就通过”。方桌使用 `y1 + 0.94 × (y2-y1)`；提供四边形时边界容差为 0。圆桌使用由 TABLE 框确定的椭圆及 `y2 - 0.08 × (y2-y1)` 作为参考点；返回行另含 `nx`、`ny`、`radius`、`angle`，供后续时序融合使用。

### 边界情况

- 没有满足阈值的 TABLE：返回 `None` 和空接受列表，即使传入四边形也不会直接通过物品。
- 低于类别阈值的物品被跳过，不保证出现在 `rejected` 中。
- 提供四边形时默认不绕过边界；`table_contacts` 是上游提供的特殊接触证据，本例未使用。
- 通过几何筛选只表示符合当前区域规则，不证明是立体实物。定位阶段的 RGB-D 输入与拟合实现见 [square_surface.py](../../reference/competition_snapshot/national_competition/runtime/square_surface.py) 和 [round_surface.py](../../reference/competition_snapshot/national_competition/runtime/round_surface.py)。

## 2. 深度证据：立体、平面和未知怎样区分？

示例：[depth_states.py](depth_states.py) · [实际函数 `evaluate`](../../reference/competition_snapshot/national_competition/runtime/depth_validation.py) · [完整输出](expected/depth_states.json)

### 输入

```python
depth = np.full((200, 200), 1400.0, dtype=np.float32)
depth[60:120, 60:120] = 1280.0  # 候选区比支撑面靠近相机 120 mm
row = {"class_id": 2, "confidence": 0.90, "xyxy": [60, 60, 120, 120]}
plane = {
    "coeff": [0.0, 0.0, 1 / 1400],
    "cx": 100.0, "cy": 100.0, "rx": 100.0, "ry": 100.0,
    "p90_error_mm": 0.0,
}
evidence = evaluate(depth, plane, row, other_boxes=[], profile=profile)
```

`profile` 从已保存的 [depth_rules.json](../../reference/competition_snapshot/national_competition/depth/depth_rules.json) 中读取 `profiles["遥控器"]`，本例没有自行调阈值。`other_boxes` 是同帧其他物品的 `xyxy` 列表，参与遮挡和深度归属检查，不能包含当前物品自身。

这里的 `plane` 是已知的合成支撑面。一般形式为 `1/Z = a*(x-cx)/rx + b*(y-cy)/ry + c`，`coeff=[a,b,c]`。在真实流程中应来自可靠的同帧桌面估计，并保留噪声统计，不能直接照抄此处的常数。

### 输出

返回证据字典，关键字段为 `state`、`reject`、`reason`；不同分支还会提供有效深度比例、整框凸起面积、平面误差等诊断数据。示例只打印三个共用字段：

| 合成输入 | `state` | `reject` | 实际返回的 `reason` |
|---|---|---|---|
| 桌面 1400，候选区 1280 mm | PASS | false | VISIBLE_3D_STRUCTURE |
| 桌面与候选区均为 1400 mm | FLAT | true | DENSE_PLANAR_PRINT |
| 候选区深度全部为 0 | UNKNOWN | false | INSUFFICIENT_OBJECT_DEPTH |
| 深度数组为 `None` | UNKNOWN | false | DEPTH_MISSING |

`reject=true` 表示单帧有拒绝证据，**不代表立刻删除最终实例**。正式时序流程还会调用 `consistent_reject` 汇总证据。`UNKNOWN` 不等于“假物”；`PASS` 也不能推断容量或保证物体属于真实桌面。此处差值是相机射线深度残差，不自动等于竖直高度。

## 3. 时序融合：怎样从多次观测得到物品数量？

示例：[temporal_count.py](temporal_count.py) · [实际函数 `round_fuse`、`square_fuse`、`burst_vote`](../../reference/competition_snapshot/national_competition/runtime/timed_fusion.py) · [完整输出](expected/temporal_count.json)

### 圆桌输入：有效时刻列表

```python
frames = [{
    "signed_rotation": 0.0,
    "objects": [{
        "class_id": 0, "confidence": 0.90,
        "xyxy": [380, 238, 410, 254],
        "radius": 0.40, "angle": 0.10, "depth_state": "UNKNOWN",
    }],
}]
selected, diagnostic = round_fuse(frames, use_depth=False)
```

这段仅展示数据结构，一个时刻不足以通过最终计票；完整脚本构造了 12 个有效时刻。

| 字段 | 含义 |
|---|---|
| `frames` | 已通过上游桌面有效性检查的观测时刻；一个时刻可能由两张原始帧经 `burst_vote` 合成 |
| `objects` | 该时刻桌面内物品行列表；桌面有效但没检到物品时为 `[]`，这个时刻仍须保留 |
| `radius` | 椭圆归一化坐标下的半径：`hypot(nx,ny)` |
| `angle` | 相对桌面中心的角度，弧度，`atan2(ny,nx)`，沿用图像 y 向下的坐标约定 |
| `signed_rotation` | 累计旋转角，弧度；与 `angle` 使用同一方向约定，用于跨时刻关联 |
| `depth_state` | 此例使用 UNKNOWN；`use_depth=False` 用于只观察时序规则 |

### 圆桌输出

`selected` 是最终 `UiDetection` 列表，每个元素表示一个保留实例。`diagnostic` 包含轨迹、计数上限、分母和门槛等。示例实际算得：

```json
{
  "counts": {"0": 2},
  "valid_moments": 12,
  "id_min_moments": 5,
  "multi_min_moments": 6
}
```

构造过程为 20 次采集尝试，其中 8 次桌面无效；12 个有效时刻中，前 6 个有两个眼镜，其余 6 个有效时刻没有检出。最终得到两个实例，而不是把 12 个框累计为 12 件。

当前实现使用 `max(4, ceil(有效时刻数×0.35))` 作为单件支持门槛，`max(6, ceil(有效时刻数×0.42))` 作为多件支持门槛。若错误地将那 8 次无效桌面当作有效空桌加入，门槛会变成 7 和 9，此例结果为空。这是展示接口差异的反例，不应当作正确调用。

### 两个相关接口

- **`square_fuse(frames, use_depth=False)`**：每帧只需 `objects` 行列表，物品行包含 `class_id`、`confidence`、`xyxy`。示例在 7 帧中有 5 帧看到同一个卷尺，实际输出一个卷尺；没有旋转字段，也不应用圆桌比例规则。
- **`burst_vote(samples)`**：输入一个时刻内的原始样本，每个样本有独立 `sequence` 和 `objects`。示例中眼镜、胶带只在第一帧出现，第二帧为空，输出保留眼镜、没有胶带。这是当前眼镜的单帧例外，并不表示最终圆桌只需一帧。

完整圆桌系统还处理容量竞争、类别冲突、遮挡与深度否决。这里不以少量合成输入证明那些分支都有效，也不实现相机采集、旋转估计或 OCR。

## 把输入换成自己的数据

1. 先把检测框映射回参考 RGB 图像，确认类别顺序与本项目一致。
2. 调用 `scope`，得到同帧桌面内候选。
3. 有配对深度时，用估计的支撑面和类别 profile 调用 `evaluate`；没有深度时保留 UNKNOWN。
4. 按方桌帧或圆桌有效时刻组织输入，再调用对应融合函数。要启用正式深度否决，应保留完整 `depth_evidence`、来源序号及其他上下文，不能只复制本例打印的三个字段。
5. 用独立标注比较输出数量；这些示例本身不是精度测试集。
