# Original SCEG on the publicly released DN4 Stanford Dogs split

状态 **paused**；记录时间 2026-10-13T12:55:15.315003+00:00。已完成 800/800 epochs；正式训练完整=True，四组正式评测完整=False。只有 seed42。未执行指标为 null，启动不等于完成。

用户于 2026-10-13T11:34:07+00:00 要求暂停。进程组15905已SIGSTOP挂起，最后一组30-way5-shot评测的内存现场保留，显存仍占用；自动收尾监听器已停用。本次提交/推送保存代码和现有证据，不代表四组评测全部完成，也不恢复任务。

## 数据与配置来源

DN4@440857b23a5bc9798004f81f732c5cc9272a99b8 的三个发布 CSV 只用于划分，未采用 DN4 的模型或训练配方。原始 CSV、SHA256、精确 breed 清单、archive 双向差集见 [split.json](data_protocol/split.json)。实际核验 70/20/30 类，两两互斥、合并120类；图片 11962/3478/5140，总20580，每个(label,filename)唯一，CSV/archive 差集为空。`protocol_id: dn4_dogs_2019`、`source_verified: true`、当前对照结论 `sceg_exact_split_match: mismatch`。verified 仅指DN4公开来源和映射已核验。冻结split.json/config中的unconfirmed保留为训练启动时的知识状态，不改写其内容或hash；最新作者对照记录独立保存。

用户提供作者原始类别清单后，逐类别核验确认当前DN4划分与作者划分不一致；62/120个breed归属不同。完整差异和作者文件SHA256见 [author_split_comparison.json](data_protocol/author_split_comparison.json)。部分Dogs专项训练配置仍未确认。

复用原 archive 及 [已有来源审计](/root/projects/aris/sceg_fsfg/data/provenance/raw_archive_audit.json)，未重复下载或全量哈希。独立根 `/root/projects/aris/sceg_dogs_baseline/data/dogs_dn4/StanfordDogs_fewshot`；原图字节保持不变；RGB→短边Resize92→CenterCrop84，val_pre/test_pre 各生成一次PNG，使用pre路径时不再resize。ImageFolder映射、文件状态及提取时的hash见 [prepared.json](data_protocol/prepared.json)。

SCEG 来源 [Legenddddd/SCEG@f7f130aeb5b1b49246150574c53a874d4ce8de80](https://github.com/Legenddddd/SCEG/tree/f7f130aeb5b1b49246150574c53a874d4ce8de80)；已核对固定版本 scripts/run.sh 和 utils/util.py，并复用模型/损失/trainer/evaluator及 [补充/差异核验](../../docs/discrepancy_report.md)。

| 配方 | 设置 | 来源性质 |
|---|---|---|
| Dogs网络 | stanford_dog，70类，ResNet12，84×84，stage3→stage2 | Dogs类数/分支 + 通用实现 |
| 训练episode | 10-way、5 support、15 query，support-first；按样本消耗，不固定每轮episode数 | 公开CUB脚本/通用trainer |
| 原四项I²L | intra=.85，inter=.05，固定lamda=38 | 沿用公开CUB脚本；非作者Dogs专项确认 |
| 优化与预算 | SGD .1/.9，Nesterov，wd=5e-4；StepLR400/.1；800 epochs | 通用实现/公开脚本及冻结工程 |
| 验证best | 每20 epochs，5-way5-shot，1000 episodes，均值最大 | 作者实际调度；20个val类 |
| 数值与资源 | FP32，AMP=false，TF32=false，deterministic warn_only；seed42，GPU0，workers4，CPU threads2 | 已验证工程环境 |
| 正式测试 | 5/30-way × 1/5-shot，各2000 episodes，query16 | 本轮固定公开协议 |

Dogs lamda=44 是 utils/util.py 的注释；不视作执行配置。loss 温度保持原实现，未加入 optimizer。没有调参、CUB checkpoint 微调或其他训练seed；从独立随机初始化开始。

## 执行与检查

工作区 `/root/projects/aris/sceg_dogs_baseline`，分支 `experiment/sceg_dogs_baseline`，正式运行启动时HEAD为 `3dec9cc569dc3b8b07e1dd35423f9ed07ca25a51`，没有回退或merge。后续按用户授权提交并推送附件和本轮代码/暂停记录，保持启动时的源码版本记录不变。环境复用 `/root/projects/aris/sceg_fsfg/.venv/bin/python`，没有安装/升级依赖。

准备命令：`rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_dogs_dn4.prepare`。
必要测试：`rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m pytest -q tests/test_dogs_protocol.py tests/test_dogs_dn4_protocol.py`；12项通过；包括CSV来源/映射、入口隔离、5-way复用及评测RNG记录的流不变检查。
真实数据检查：`rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_dogs_dn4.check`；1个训练forward/backward与2个验证episode通过，global=[200,70]，episode=[150,10]，loss=53.673241，活动参数梯度通过；6张原图/pre输入逐元素相同。没有保存smoke权重或继承smoke状态。旧 [Dogs合成smoke/恢复](../sceg_dogs_baseline/REPORT.md) 和 [CUB等价性结论](../../docs/cub_reproduction_report.md) 直接引用，没有重复实验。

真实正式命令（cwd为上述工作区）：
```bash
/root/projects/aris/sceg_fsfg/.venv/bin/python -u -m experiments.sceg_dogs_dn4.run --run-id dogs800_seed42_dn4
```

持久 screen 会话 `15904.sceg_dogs_dn4_seed42`，Python PID `15906`。日志 [train.log](runs/dogs800_seed42_dn4/train.log)、[metrics.json](runs/dogs800_seed42_dn4/metrics.json)、[epoch_history.jsonl](runs/dogs800_seed42_dn4/epoch_history.jsonl)；实际配置/环境/代码HEAD/差异/未跟踪源码小补丁均在同一run内。启动方式见运行command/cwd。只运行本任务一个GPU进程，没有终止其他任务。

恢复命令（确认原进程已经退出后使用同一run；latest、best原位滚动，不复制权重）：
```bash
rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -u -m experiments.sceg_dogs_dn4.run --run-id dogs800_seed42_dn4 --resume /root/projects/aris/sceg_dogs_baseline/experiments/sceg_dogs_dn4/runs/dogs800_seed42_dn4/latest.pth
```
若800轮已完成、仅评测中断，在该命令末尾添加 `--evaluate-only`。会复用已存在的5-way记录，只完成缺项；逐设置保存rng_after，以保持评测随机流续接。正常执行不额外重置各setting的seed。

## 进度、资源与正式结果

最近一次进度写入的执行耗时 46490.825255741 秒（尚非完整总耗时）；训练含验证耗时 46011.462662391 秒；完整测试耗时 null 秒；该进度点记录的峰值GPU allocated 10384317440 bytes。已完成setting的评测耗时在对应原始test JSON内。曲线见epoch_history及validation_epoch_*.json。

best epoch=420，best验证均值=91.07875；权重 `/root/projects/aris/sceg_dogs_baseline/experiments/sceg_dogs_dn4/runs/dogs800_seed42_dn4/checkpoint.pth`，SHA256=2428ab008377ea8f9559b3ef26c0cd271475d38907a10e93588c340d42180c9d。best尚未选出时文件仅为trainer的临时last，不能当作验证best。只长期保存该目录的latest.pth与checkpoint.pth；前者含model/optimizer/scheduler/RNG/DropBlock counters；无历史epoch权重或按setting复制权重。

| setting | accuracy (%) ± 原版95% CI半宽 | 正式episodes |
|---|---|---|
| 5-way 1-shot | 80.1806 ± 0.4365 | 2000 / 2000 |
| 5-way 5-shot | 90.3969 ± 0.2418 | 2000 / 2000 |
| 30-way 1-shot | 48.4604 ± 0.1667 | 2000 / 2000 |
| 30-way 5-shot | null | 0 / 2000 |

原始记录为run内test_*way_*shot.json：逐episode准确率、实际episode索引、样本路径表、同一checkpoint hash。生成报告时从逐episode记录重算均值与 `1.96 * sqrt(var(episode_accuracy, ddof=0) / N)`，并检查2000个任务及索引范围/无重复。CI仅反映固定模型与划分下episode抽样变化，不代表多训练seed稳定性。

## 结论与范围

公开协议与必要技术检查已经核验，正式任务尚未完整完成；当前不能宣称已建立完成的数值基线。 用户提供作者原始类别清单后，逐类别核验确认当前DN4划分与作者划分不一致；62/120个breed归属不同。完整差异和作者文件SHA256见 [author_split_comparison.json](data_protocol/author_split_comparison.json)。部分Dogs专项训练配置仍未确认。 本轮始终是公开DN4协议基线，不改称作者split复现，不把原论文差值视为模块增益，不设置追平/超论文数字门槛。一个seed不能宣称稳健性或保证符合其他论文的比较协议。

已有CUB、ProKeR v1及旧Dogs blocked记录元数据保持不变=True，核验见 [preservation_check.json](preservation_check.json)。旧Dogs prepare/run仅做兼容入口适配，旧配置/split/runs/报告没有回填成功。共享trainer仅适配同路径续训best不复制和过滤记录metadata，未改变公式、增强、初始化、采样或调度。没有运行ProKeR、CUB重训、其他数据集、多seed或调参。完成本轮后停止。
