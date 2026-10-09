# SCEG Reproduction

逐阶段复现 CVPR 2026 **From Few-way to Many-way: Rethinking Few-shot Fine-grained Image Classification**。

事实优先级：固定版本的官方代码 > 论文正文/补充材料 > README。
官方参考：[Legenddddd/SCEG](https://github.com/Legenddddd/SCEG)，commit `f7f130aeb5b1b49246150574c53a874d4ce8de80`。
官方仓库仅作为项目外参考，不复制为本项目工程。

入口文档：[仓库分析](docs/repository_analysis.md)、[工程设计](docs/engineering_design.md)、[差异台账](docs/discrepancy_report.md)、[关卡清单](docs/reproduction_checklist.md)。

所有实验使用独立的 `experiments/<experiment>/runs/<run_id>/`。运行目录记录配置、代码版本与差异、命令、环境、数据来源、日志、指标和 checkpoint。随机输入验证不计为数据集结果；overfit 不计为论文复现指标。

数据只从 `/root/projects/aris/fewshot` 读取并复制，不建立符号链接，不修改来源工程。缺少可核验的类别划分时，不自行生成划分。

环境重建：创建Python3.10虚拟环境，安装requirements.txt；GPU环境使用PyTorch官方cu124 wheels。当前项目已有`.venv`，复用了现有torch安装；实际环境见运行目录environment.json。

```bash
.venv/bin/python -m tools.tensor_skeleton --device cuda:0
.venv/bin/python -m pytest -q
.venv/bin/python -m tools.verify_official --reference /tmp/sceg-official-reference --gradients
.venv/bin/python -m tools.overfit --run-id your_fresh_debug_run
.venv/bin/python -m tools.analyze_checkpoint --checkpoint experiments/sceg_debug/runs/four_images_200_steps/checkpoint.pth
```

完整训练入口为`tools.train`，默认执行800epochs，不用短训练冒充复现。当前严格全参数梯度关卡失败；只有明确接受官方闲置参数例外后，才能使用`--accept-official-inactive`。debug checkpoint不能用于baseline正式测试。

```bash
.venv/bin/python -m tools.train --config experiments/sceg_original/config.yaml --accept-official-inactive
.venv/bin/python -m tools.test --checkpoint YOUR_BASELINE_CHECKPOINT --way 50
```

目前未启动完整训练；上述train命令是待确认后的操作说明。CUB自动正式测试为5-way，many-way另用50；其他数据集分别30/49/25，并需自己的准确配置与class split来源。所有未知结果维持null。
