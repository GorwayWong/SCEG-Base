# Phase C — running

1. 完成事项：Phase A/B通过后，从seed42重新初始化正式训练；已验证GPU进程、optimizer更新和逐epoch完整checkpoint写入。当前不是训练完成报告。
2. 产生文件：config.yaml/config_resolved.yaml、environment.json、git_commit.txt、command.txt、data_manifest.json、image_hashes.json、source_manifest.json、execution_adapter.json、model_summary.txt、parameter_summary.json、flops.json、train.log、metrics.json、epoch_metrics.jsonl、各epoch不可变checkpoint/RNG及checkpoint_last.pth/checkpoint_best.pth/rng_state.pt别名。
3. 验证结果：首轮27episodes、loss34.962846与独立dry run一致；epoch2的完整checkpoint与重启resume逐字节相同（8db6179b...17ec4d7），loss20.538388；启动后已检查到6/800epochs，GPU使用98%。已有源码和历史产物hash未变化。每20epoch才验证，第一次best选择尚待epoch20。
4. 风险：需要数小时；本机/WSL需持续运行，暂停或重启会中断任务；800epoch及正式评测均未完成，不能声明baseline reproduced。当前单seed与CUDA warn_only限制保留，无调参。
5. 下一阶段计划：supervisor PID16148、训练子进程PID16154、执行会话72949。workflow顺序等待800epochs及已有trainer的best checkpoint 5-way测试，然后tools.test完成50-way两种shot各2000episodes，独立重算CI并生成最终结果/文档。任一步失败停止并记录status.json，不自动重训练。

状态：`experiments/sceg_original/results/20261009_cub800_seed42_formal/status.json`；实时训练：本目录metrics.json/train.log；进程日志：`experiments/sceg_original/logs/20261009_cub800_seed42_formal/`。
