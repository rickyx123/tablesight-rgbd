# 参与项目

欢迎交流 RGB-D 感知、桌面物品计数和边缘视觉部署中的问题。可以通过 Issue 描述具体场景、复现步骤和期望结果。

项目主要负责人：[rickyx123](https://github.com/rickyx123)，负责全部代码构建、数据集拍摄建议与图像标注。

## 提交问题或改进

请说明输入条件、实际输出与预期输出。涉及模型结果时，附上版本、评估集合、输入尺寸和阈值；涉及数据时，注意同一场景及同一原图的派生版本不能跨训练和验证集合。

改动独立工具后运行：

```bash
python -X utf8 -m unittest discover -s tests -v
python -X utf8 scripts/check_repository.py
```

上传原图、权重或第三方代码前，请确认相应的分享权限。算法参考快照的范围与运行依赖见[说明](reference/README.md)。
