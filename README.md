# SCEG Reproduction

逐阶段复现 CVPR 2026 **From Few-way to Many-way: Rethinking Few-shot Fine-grained Image Classification**。

事实优先级：固定版本的官方代码 > 论文正文/补充材料 > README。
官方参考：[Legenddddd/SCEG](https://github.com/Legenddddd/SCEG)，commit `f7f130aeb5b1b49246150574c53a874d4ce8de80`。
官方仓库仅作为项目外参考，不复制为本项目工程。

入口文档：[仓库分析](docs/repository_analysis.md)、[工程设计](docs/engineering_design.md)、[差异台账](docs/discrepancy_report.md)、[关卡清单](docs/reproduction_checklist.md)。

所有实验使用独立的 `experiments/<experiment>/runs/<run_id>/`。运行目录记录配置、代码版本与差异、命令、环境、数据来源、日志、指标和 checkpoint。随机输入验证不计为数据集结果；overfit 不计为论文复现指标。

数据只从 `/root/projects/aris/fewshot` 读取并复制，不建立符号链接，不修改来源工程。缺少可核验的类别划分时，不自行生成划分。
