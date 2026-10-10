# Phase C — 续训运行中

2026-10-10按用户指示从正式run的epoch122恢复，继续至总800epochs。新run_id：`20261010_cub800_seed42_resume122`；没有重新从零训练。

1. 完成事项：恢复model、SGD momentum、StepLR、RNG和DropBlock计数；启动后已检查到epoch126、step3402，原始起点为epoch122、step3294。
2. 文件：resume_input.json/continuation.json记录来源和checkpoint hash；metrics.json合并前序及当前epoch历史；config_resolved.yaml、environment、command、source/data manifest、train.log和逐轮指标正常保存。
3. 验证结果：恢复时lr=.1、scheduler last_epoch122、DropBlock计数9294；当前训练持续更新，best仍沿用此前epoch100的91.155% validation。当前run只有latest.pth和checkpoint.pth两个实体权重，无逐轮快照。前126轮指标可完整追溯，正式测试尚未执行。
4. 风险：剩余训练及评测需要数小时；当前尚未达到baseline完成标准。保持计算机与WSL运行即可，checkpoint按常规last/best保存。
5. 下一阶段：总800epochs后自动用best执行5-way1/5-shot和50-way1/5-shot各2000episodes，核对mean/95%CI，生成results.json、论文格式表格和报告；低于论文时只核查协议，不调参。

screen会话：`sceg_cub800_seed42_resume122`。启动命令保存在results同名目录process.json，训练命令在本目录command.txt。实时进度以本目录metrics.json和train.log为准；流水线状态为 `experiments/sceg_original/results/20261010_cub800_seed42_resume122/status.json`。

冻结配置、数据split及已验证的模型/loss/dataset/trainer代码保持不变。每个run只保留常规last/best和轻量审计记录，评测引用best，不另存权重副本。
