# SCEG Reproduction

逐阶段复现 CVPR 2026 **From Few-way to Many-way: Rethinking Few-shot Fine-grained Image Classification**。

事实优先级：固定版本的官方代码 > 论文正文/补充材料 > README。
官方参考：[Legenddddd/SCEG](https://github.com/Legenddddd/SCEG)，commit `f7f130aeb5b1b49246150574c53a874d4ce8de80`。
官方仓库仅作为项目外参考，不复制为本项目工程。

入口文档：[仓库分析](docs/repository_analysis.md)、[工程设计](docs/engineering_design.md)、[差异台账](docs/discrepancy_report.md)、[关卡清单](docs/reproduction_checklist.md)。

所有实验使用独立的 `experiments/<experiment>/runs/<run_id>/`。运行目录记录配置、代码版本与差异、命令、环境、数据来源、日志、指标和 checkpoint。随机输入验证不计为数据集结果；overfit 不计为论文复现指标。

Git保存运行目录中的轻量审计记录。数据、虚拟环境和大型checkpoint留本地；每个checkpoint的大小/SHA256由checkpoint_manifest.json记录。清理明细见[docs/cleanup_report.md](docs/cleanup_report.md)，历史首轮overfit事实已合并到[debug_overfit_report.md](debug_overfit_report.md)。

数据只从 `/root/projects/aris/fewshot` 读取并复制，不建立符号链接，不修改来源工程。缺少可核验的类别划分时，不自行生成划分。

环境重建：创建Python3.10虚拟环境，安装requirements.txt；GPU环境使用PyTorch官方cu124 wheels。当前项目已有`.venv`，复用了现有torch安装；实际环境见运行目录environment.json。

```bash
.venv/bin/python -m tools.tensor_skeleton --device cuda:0
.venv/bin/python -m pytest -q
.venv/bin/python -m tools.verify_official --reference /tmp/sceg-official-reference --gradients
.venv/bin/python -m tools.overfit --run-id your_fresh_debug_run
.venv/bin/python -m tools.analyze_checkpoint --checkpoint experiments/sceg_debug/runs/four_images_200_steps/checkpoint.pth
```

完整训练入口为`tools.train`，默认执行800epochs。已接受官方闲置参数例外，保留这些参数并核验所有活动参数的梯度；严格全参数梯度关卡仍不标为通过。debug checkpoint不能用于baseline正式测试。

```bash
.venv/bin/python -m tools.train --config experiments/sceg_original/configs/cub.yaml --accept-official-inactive
.venv/bin/python -m tools.test --checkpoint YOUR_BASELINE_CHECKPOINT --way 50
```

CUB 已完成800 epochs，正式 run 为 `20261010_cub800_seed42_resume122`，从前序第122轮恢复并保留完整800轮记录。5-way 1-shot/5-shot与50-way 1-shot各2000 episodes已完成；50-way 5-shot按用户要求暂停。完整baseline复现尚未完成。

冻结配置见 [cub.yaml](experiments/sceg_original/configs/cub.yaml)，正式报告见 [cub_reproduction_report.md](docs/cub_reproduction_report.md)，差异见 [cub_discrepancy.md](docs/cub_discrepancy.md)，结果表见 [cub_result_table.md](docs/cub_result_table.md)。实验记录适配层位于 `experiments/sceg_original/runs/20261009_cub800_seed42_execution/`，调用既有训练和测试入口；其运行时副本保存在结果目录。训练使用获准的4 workers，其他训练数值配置遵循固定官方代码。

其他数据集仍需各自的配置、可核验的class split和正式实验；未测结果保持null。

权威结果：[results.json](experiments/sceg_original/results/20261010_cub800_seed42_resume122/results.json)。最终last/best及三组原始episode结果保留；旧中断权重和重复诊断文件已清理，重复JPEG在项目内部去重。详见[最终清理记录](experiments/sceg_original/results/20261010_final_cleanup/cleanup_report.md)。
