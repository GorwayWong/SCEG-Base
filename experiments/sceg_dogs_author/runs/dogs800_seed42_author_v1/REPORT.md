# Stanford Dogs 原版 SCEG 实验报告

状态：**completed**。run：`dogs800_seed42_author_v1`；800 轮训练、四组各 2000 个正式任务及统计重算全部完成。完成时间：2026-10-14T12:58:14.433354+00:00（UTC）。

数据协议为 `sceg_author_dogs_v1`，采用用户通过作者邮件取得的类别清单；训练/验证/测试为 70/20/30 类、11,877/3,325/5,378 张原图，共 20,580 张。标签沿用 prep/split.json 的 ImageFolder 映射。训练使用原图随机增强，val_pre/test_pre 只做 ToTensor/Normalize。输入与清单哈希见 [data_check.json](data_check.json)，完整清单沿用只读 prep/。

模型为原版 ResNet12 + SC，保留 Dogs stage3→stage2 分支、三路原型分类及训练期 70 类代表。seed42 随机初始化；训练任务 10-way、5 support + 15 query，support-first，沿用样本消耗式采样，共 43,985 个训练 episode。四路 I²L 等权相加，intra=0.85、inter=0.05、lambda=38 固定；SGD lr=0.1、momentum=0.9、Nesterov、weight_decay=5e-4，800 epochs，StepLR(400,0.1) 在验证后调用。完整解析配置见 [config.yaml](config.yaml)。

每 20 轮进行 5-way 5-shot、1000 episode、query=16 验证，严格按验证均值选 best：**epoch 520，90.1438%**。正式测试均为 eval 模式、每类 16 query、每组 2000 episode，四组使用同一 best；episode 内 support/query 无重复。

| 设置 | 实测准确率（%） ± 95% 半宽（pp） | 论文参考（%） | 差值（pp） | 记录墙钟秒 | 峰值分配显存（GiB） |
|---|---:|---:|---:|---:|---:|
| 5-way 1-shot | 77.1644 ± 0.4826 | 77.89 | -0.7256 | 215.56 | 1.05 |
| 5-way 5-shot | 88.7400 ± 0.2569 | 89.13 | -0.3900 | 264.07 | 1.25 |
| 30-way 1-shot | 47.0756 ± 0.1711 | 47.55 | -0.4744 | 17290.95 | 5.37 |
| 30-way 5-shot | 66.2110 ± 0.0979 | 66.63 | -0.4190 | 3213.40 | 6.51 |

论文参考值来自 [正文 Table 1](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhao_From_Few-way_to_Many-way_Rethinking_Few-shot_Fine-grained_Image_Classification_CVPR_2026_paper.pdf)。四项均低于参考 0.39–0.73 个百分点。95% 半宽按 `1.96 × sqrt(var(episode_accuracy, ddof=0) / 2000)` 重算，单位为百分点；它仅反映单个 seed42 模型的 episode 抽样变化。论文未给出 CI，不能据此作跨训练 seed 稳定性或显著性结论。

训练、验证及 checkpoint 记录耗时 **12.79 小时**，训练峰值分配显存 **9.67 GiB**。四个完整设置的记录墙钟时间合计 5.83 小时；30-way 5-shot 的 3213.40 秒包含用户要求的暂停区间，记录暂停约 27.10 分钟，不等同于 GPU 活跃时间。首次启动至完成历时 20.12 小时，包含中断、等待和暂停；未落盘旧尝试的有效耗时无法精确确定。

环境：单张 NVIDIA RTX 4060 Ti 16GB，Python 3.10.20、PyTorch 2.6.0 / CUDA 12.4；workers=4、CPU threads=2、FP32、AMP/TF32=false、deterministic warn_only。作者补充材料使用 RTX 3090；硬件与历史软件环境并不完全一致。

配方依据为 Legenddddd/SCEG 固定提交 `f7f130aeb5b1b49246150574c53a874d4ce8de80` 的活动代码。lambda=38 及公开 run.sh 选项沿用 CUB 脚本，并非作者单独确认的完整 Dogs 配方；parser 的 36 和 Dogs 注释中的 44 未用于搜索。论文称温度可学习，代码实际将其排除在 optimizer 外，本次保留固定行为。详细来源与差异见 [recipe.md](../../recipe.md)。

恢复记录：原 pipeline 进程在最后一组完整落盘前消失，没有异常退出事件，原因仍未确定。只缺失的 30-way 5-shot 从保存的 RNG 边界启动恢复；前三组完整 JSON 被核验复用且未改写。恢复进程随后按用户要求暂停，再通过 SIGCONT 继续同一内存状态，没有再创建新的评测进程。完整事件、历史暂停和原生控制命令见 [events.jsonl](events.jsonl)、[pause.json](pause.json)、[native_control_commands.txt](native_control_commands.txt)。完成缺项不代表已查明原退出根因。

最终核验见 [final_checks.json](final_checks.json)：800 轮及 40 次验证记录完整，四组原始准确率与均值/CI 一致，样本索引、类别、路径映射、split、config 与 best 身份一致，评测 RNG 链连续。best SHA256：`1939b5e0513b88702dcc06a9413f67c14e247cea1d4be0eb51b2c3ba1810b7bb`。本地仅保留 `latest.pth`（完整 epoch 800 状态）及 `checkpoint.pth`（best epoch 520）；权重未纳入 Git。

运行启动时 HEAD 为 `5ee680fa9fbcdd93fc3381a4d2e3aefe5fd52c82`；当时差异由 git_diff.patch、新入口源码快照及 execution_sources.json 追溯。相同运行源码随后收录在 `a15f7556a9487ee426c99a0b87506ab7342fe5d8`，最终核验确认源码哈希未变。本报告在结果完成后整理。环境、原始命令、续评测命令与实现/模型检查分别见 environment.json、command.txt、resume_commands.txt、verification.json。改动限于作者任务及默认行为不变的公共记录选项；CUB、ProKeR、历史归档和 prep 未改写。

复现与恢复命令（cwd 为 `/root/projects/aris/sceg_dogs_baseline`；当前 run 已完成，完整评测文件会被复用）：

```bash
# 原始训练命令（创建新的 run；当前同名目录已存在）
rtk proxy env CUDA_VISIBLE_DEVICES=0 /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_dogs_author.run --run-id dogs800_seed42_author_v1 --mode pipeline
# 同 run 的 latest 完整 epoch 续训
rtk proxy env CUDA_VISIBLE_DEVICES=0 /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_dogs_author.run --run-id dogs800_seed42_author_v1 --mode pipeline --resume
# 仅补缺失正式评测
rtk proxy env CUDA_VISIBLE_DEVICES=0 /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_dogs_author.run --run-id dogs800_seed42_author_v1 --mode evaluate
# 查看当前状态
rtk proxy env CUDA_VISIBLE_DEVICES=0 /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_dogs_author.run --run-id dogs800_seed42_author_v1 --mode status
```
