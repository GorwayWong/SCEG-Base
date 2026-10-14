# SCEG + ProKeR Stanford Dogs 跨数据集验证

状态：**completed**。固定 seed42 / epoch520 Dogs 模型，CUB 验证锁定参数直接迁移。

## 目的与实现

验证纯视觉 ProKeR 的跨数据集收益，不训练、不搜索 Dogs 参数。独立 worktree 基于 Dogs author 分支。仅原样复制 CUB 的 72 行 head.py；其余使用原 SCEG encode、CosineClassifier、ImageFolder 和预处理。

核输入 u=concat(normalize(z2),normalize(z3),normalize(z4))/sqrt(3)，K=exp[-beta(1-u vᵀ)]；A=(I+Kss/lambda)^(-1)(Y-f0(S))，f(Q)=f0(Q)+w Kqs A。f0 为原三路 cosine 的平均值，原型先对未归一化 support 求均值。Baseline 使用原 stage3+stage4+stage2 分数和。Ridge 为同一 u 上无截距的线性岭回归。

1-shot ProKeR=(beta10,lambda1,w1)，Ridge lambda100；5-shot ProKeR=(beta1,lambda0.1,w1.5)，Ridge lambda0.1。两种 way 共享相同 shot 参数，来自同一 CUB lock。

## 正式结果

| 设置 | Baseline % ± CI pp | Ridge % ± CI pp | Ridge Δ ± paired CI pp | ProKeR % ± CI pp | ProKeR Δ ± paired CI pp |
|---|---:|---:|---:|---:|---:|
| 5way_1shot | 77.1644 ± 0.4826 | 77.2275 ± 0.4819 | +0.0631 ± 0.0252 | 77.2700 ± 0.4808 | +0.1056 ± 0.0345 |
| 5way_5shot | 88.7400 ± 0.2569 | 88.2375 ± 0.2680 | -0.5025 ± 0.1239 | 89.0062 ± 0.2552 | +0.2662 ± 0.0729 |
| 30way_1shot | 47.0756 ± 0.1711 | 47.4583 ± 0.1689 | +0.3827 ± 0.0314 | 47.4939 ± 0.1699 | +0.4182 ± 0.0342 |
| 30way_5shot | 66.2109 ± 0.0979 | 63.1290 ± 0.0980 | -3.0820 ± 0.0820 | 66.7900 ± 0.0978 | +0.5791 ± 0.0443 |

每组2000 episodes，每类16 query。CI 半宽=1.96×std(ddof=0)/sqrt(n)；paired CI 对逐 episode 方法准确率减 Baseline 的差值计算。准确率记录为百分数，增益和半宽为百分点。

## 耗时与资源

| 设置 | Baseline ms/episode | Ridge ms/episode | ProKeR ms/episode | 各头峰值 allocated MiB (B/R/P) |
|---|---:|---:|---:|---:|
| 5way_1shot | 0.802 | 1.662 | 2.731 | 32.85/33.87/33.87 |
| 5way_5shot | 0.786 | 1.611 | 2.664 | 33.00/34.17/34.17 |
| 30way_1shot | 0.774 | 4.467 | 5.522 | 37.22/43.44/43.50 |
| 30way_5shot | 0.789 | 4.614 | 5.675 | 38.10/45.09/45.15 |

CUDA 同步墙钟计时，包括每头 support 拟合和 query argmax，排除 CPU→GPU 传输、准确率统计及 Backbone。方法顺序逐 episode 轮转，每头一集 warmup。Ridge/ProKeR 使用原复用 head.fit（含原型、核输入及 one-hot 建立），Baseline 直接复用原 cosine 分类器；这些是实际实现的端到端分类头开销，不是核求解器单独耗时。

单份 CPU/disk FP32 三路缓存 39.39 MiB，编码 batch48 / workers2 / CPU threads2。编码 7.59 s，峰值 allocated 596.98 MiB / reserved 808.00 MiB。特征按 episode 送 GPU，不保存 episode 特征副本、核矩阵或 logits。在线一致性检查的少量重编码单独进行。

## 历史重放与验证

- 5way_1shot：2000/2000 历史 episode 准确率一致，均值差 +0.00000000 pp；差异定位记录见 metadata.json 的 validation.settings。
- 5way_5shot：2000/2000 历史 episode 准确率一致，均值差 +0.00000000 pp；差异定位记录见 metadata.json 的 validation.settings。
- 30way_1shot：2000/2000 历史 episode 准确率一致，均值差 +0.00000000 pp；差异定位记录见 metadata.json 的 validation.settings。
- 30way_5shot：1999/2000 历史 episode 准确率一致，均值差 -0.00010417 pp；差异定位记录见 metadata.json 的 validation.settings。

唯一差异为零起始 episode 74、query位置210：缓存版选局部类4，原整集在线版选类13。缓存top-2分数差为0；在线/缓存平均cosine最大误差4.7684×10⁻⁷，三路特征最大误差1.5736×10⁻⁵。在线版该集66.458333%，与历史一致，缓存版66.25%。这定位为FP32编码batch形状变化后的临界并列。历史文件未修改；全部增益使用同缓存Baseline，30-way 5-shot的正式Baseline为66.2109375%，历史为66.2110416667%。

全部8000集验证复用 head 与原分类器在同缓存上预测一致；每组前2个真实episode用原整集 batch 在线 encode 对照缓存特征和分数，验证 w=0 精确退化。历史未保存逐 query 预测，因此历史核对基于逐 episode 准确率，真实在线回归和异常诊断才比较逐 query 预测。所有原始索引/路径/类别映射均通过原路径和文件SHA引用，不复制清单。

验证：checkpoint/best epoch、32个 Dogs 运行源码、split类别不重叠、测试原图及PNG逐文件哈希、完整ImageFolder映射、每集支持/查询位置和无重复、CUB锁参数均通过。eval + inference_mode + requires_grad=False；BN buffer保持不变，无优化器/反向更新。统计已由CSV重算；历史实验目录内容指纹前后相同。

## 身份、命令与环境

Dogs base `727a6b1b3e9e60dd57cc0d2e7e4e014bb16abb6f`；CUB head source `3b95006c2f0ebe71e19e64668a796fe6a65c6a8a`。checkpoint SHA256 `1939b5e0513b88702dcc06a9413f67c14e247cea1d4be0eb51b2c3ba1810b7bb`，best epoch520。

实际生效配置与所有输入SHA见 config.yaml / metadata.json；环境、命令、数据内容汇总哈希、源码SHA、输入来源和统计核验集中记录在 metadata.json。缓存只包含三路未归一化 post-FC 特征，权重和数据只读引用。

cwd: `/root/projects/aris/sceg_proker_dogs`

```bash
rtk proxy env CUDA_VISIBLE_DEVICES=0 /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_proker_dogs.run --run-id dogs_seed42_proker_transfer_v1
```

环境：Python 3.10.20，Torch 2.6.0+cu124，CUDA 12.4，NVIDIA GeForce RTX 4060 Ti。FP32、AMP/TF32=false，deterministic warn_only。

未完成同身份恢复加 --resume；完整设置CSV只读复用。completed入口直接退出，不改写产物。只读重算使用 --audit。

完成后实际执行了 --audit，8000集统计及文件SHA核验通过；再次调用正式入口返回“completed run protected”，调用前后run内所有文件SHA完全不变。此次保护验证没有重新编码或重跑GPU评测。

启动前两次检查失败均发生在run创建与GPU评测之前：复制head.py多一个末尾换行导致源码SHA检查失败，修正后与CUB文件逐字节相同；入口初版错误地用YAML文件SHA核对历史config_sha256，查明Dogs原函数以排序JSON配置内容计算哈希后，改为同一口径。这两处实现修正未改变任何模型、数据、推理公式、参数或历史产物。正式运行没有OOM或失败，日志完整保留。

只读独立审计从四份CSV重新计算准确率、CI、gain与paired CI，与results.json误差为0；未发现公式或标签泄漏问题。审计指出一个未触发的恢复边界：若进程恰在缓存重命名完成、metadata缓存SHA落盘之前中断，--resume会因缺少缓存身份而停止。入口选择拒绝身份未记录的缓存，没有放宽检查；本轮完整产物不受影响。未为这一罕见边界改写已执行源码或重跑正式评测。

## 分析、局限与工程判断

- 5way_1shot：ProKeR paired区间 [+0.0711, +0.1402] pp（正向区间）。收益量级 +0.1056 pp，需同时考虑实际分类收益和额外拟合开销。
- 5way_5shot：ProKeR paired区间 [+0.1933, +0.3392] pp（正向区间）。收益量级 +0.2662 pp，需同时考虑实际分类收益和额外拟合开销。
- 30way_1shot：ProKeR paired区间 [+0.3840, +0.4524] pp（正向区间）。收益量级 +0.4182 pp，需同时考虑实际分类收益和额外拟合开销。
- 30way_5shot：ProKeR paired区间 [+0.5347, +0.6234] pp（正向区间）。收益量级 +0.5791 pp，需同时考虑实际分类收益和额外拟合开销。

单个训练 seed 的 episode CI 不能证明跨训练 seed 稳定性；paired normal CI 未作多重比较校正。固定2000个历史任务存在样本/类别复用；结论限于该checkpoint、作者划分及已锁定公式，不外推所有模型或数据集。测试标签仅用于准确率统计，没有用于拟合、调参或决策修改。

工程判断：值得继续开展预先规定的多训练seed及更多数据集验证。两组30-way增益为0.42/0.58 pp，paired区间均高于0；5-way也未退化，说明这组CUB参数在Dogs上保留了小幅正收益。但每集ProKeR约需2.7–5.7 ms，原头约0.8 ms；若只需最小开销，原头仍合理。30-way 1-shot线性Ridge已提升0.3827 pp，接近ProKeR的0.4182 pp，不能把全部增益归因于非线性核。5-shot Ridge在5-way/30-way分别下降0.5025/3.0820 pp，而ProKeR分别提升0.2662/0.5791 pp，这支持继续研究保留原分数的残差迁移策略。Ridge与ProKeR同时改变核及先验，当前对照不能独立识别每一成分的贡献，也不足以建立跨seed稳定性。下一轮另设实验协议；当前Dogs测试结果不用于重新选参。

Git范围：新分支experiment/sceg_proker_dogs、worktree /root/projects/aris/sceg_proker_dogs；仅新增experiments/sceg_proker_dogs/。原Dogs author、CUB original和CUB ProKeR实验目录（94/261/49个非pycache文件）运行前后内容汇总指纹相同，模型和数据身份复核也通过。

发布整理：用户在实验完成后授权提交并推送。发布入口将“当前HEAD必须等于Dogs基础提交”改为“该基础提交必须是当前HEAD的祖先”，以允许独立实验分支提交后使用；原模型/数据运行源码仍逐文件核对。completed只读审计允许验证与metadata一致的原执行源码快照，未完成恢复仍要求当前源码完全一致。原执行run.py保存在source_snapshot/run.py，SHA256为`51d517f04ae441f78ba50d8e5c1539007e05510f1038867a6f3c37f6e5ddb5ce`；模型、公式、参数、episode及原始CSV/results/config/metadata/run.log均未重写。源码、精简记录和5份统计CSV纳入提交；数据、权重和特征缓存不上传。不合并到master或其他实验分支。

任务后系统核查：正式运行于北京时间2026-10-15 01:49:48开始、01:52:10完成，无OOM或异常退出。随后Windows System事件1074记录07:44:25由MoUsoCoreWorker.exe发起计划内Service Pack重启，07:47:05由TrustedInstaller.exe发起计划内操作系统升级重启；发生在实验结束约6小时后。任务未执行关机、重启或终止其他任务的命令，现有日志不支持重启由实验导致。
