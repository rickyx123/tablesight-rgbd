# 比赛算法参考快照

本目录保存工作区中可核对的团队项目 Python 代码与部分配置，内容 SHA 见 [source_manifest.json](source_manifest.json)。这是参考快照，未改原工作区运行版。

| 模块 | 阅读入口 |
|---|---|
| 方桌透视边界与后沿 | [square_surface.py](competition_snapshot/national_competition/runtime/square_surface.py) |
| 圆桌支撑面与图像边缘 | [round_surface.py](competition_snapshot/national_competition/runtime/round_surface.py) |
| 桌面范围与时序计票 | [timed_fusion.py](competition_snapshot/national_competition/runtime/timed_fusion.py) |
| 深度验证与不确定状态 | [depth_validation.py](competition_snapshot/national_competition/runtime/depth_validation.py) |
| 同序号后台处理 | [depth_pipeline.py](competition_snapshot/national_competition/runtime/depth_pipeline.py) |
| OCR 原版本与排除 | [ocr_worker.py](competition_snapshot/national_competition/runtime/ocr_worker.py)、[ocr_exclusivity.py](competition_snapshot/national_competition/runtime/ocr_exclusivity.py) |
| 模型与结果契约 | [config.py](competition_snapshot/national_competition/config.py)、[result_writer.py](competition_snapshot/national_competition/runtime/result_writer.py) |

## 运行边界

这不是随装随用的完整硬件包。部分模块依赖 OpenCV、NumPy、Pillow/Tk、相机共享内存、OpenNI 或 Ascend ACL；OCR 另有专用环境与模型。参考中的设备运行目录是历史约定，不能直接在另一设备执行后假定安全或可复现。

原代码部分批量线性求解以 NumPy 1.x 语义开发；NumPy 2.x 的 API 差异需要适配。主仓库的独立工具只依赖标准库，不受此限制。

配置里的旧默认 MODEL_VERSION 不是最终加载权重的证据；历史正式入口通过运行环境选择实际模型，最终 V9 见 [模型卡](../models/v9-model-card.md)。

未复制 board 启动/网卡/认证脚本、SDK、权重、原始照片和完整日志。也未把取消的全图 OCR 实验放入正式快照。

团队署名、原始代码来源与许可证仍需在公开前核对。此目录用于阅读和后续拆分，当前自动测试只覆盖 src/tablesight 的独立工具。
