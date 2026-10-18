# 模型比较数值表

CSV 从对应历史评估产物逐字节复制，没有手动调整数字；SHA 见 manifest.json。

| 文件 | 内容 | 解释 |
|---|---|---|
| [v6_v7_per_class.csv](v6_v7_per_class.csv) | V6/V7 各组逐类 P/R/AP | split、imgsz 同时决定范围 |
| [v7_v8_same_split.csv](v7_v8_same_split.csv) | V7/V8 同集合服务器比较 | 包含现场与旧集，不能跨行换分母 |
| [v8_v9_same_split.csv](v8_v9_same_split.csv) | V8/V9 同集合服务器比较 | delta 为原始值算出的百分点 |
| [v8_v9_whole_per_class.csv](v8_v9_whole_per_class.csv) | 整体逐类本机对比 | 本机 batch/精度设置见评估说明 |
| [v8_v9_new_field_per_class.csv](v8_v9_new_field_per_class.csv) | 新现场逐类本机对比 | 样本少的类别需要谨慎解释 |
| [v8_v9_old_hard_per_class.csv](v8_v9_old_hard_per_class.csv) | 旧专项逐类本机对比 | 保留实际回退 |

P/R 是评估器最佳平均 F1 阈值处宏平均，不直接当固定部署门槛的 P/R。服务器总表与本机逐类表可能因 batch/精度设置产生细小差异。请阅读 [评估说明](../../archive/notes/05-evaluation.md)。
