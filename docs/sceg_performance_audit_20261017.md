# SCEG 全部相关工作的性能与冗余审计

日期：2026-10-17（Asia/Shanghai）。范围：`sceg_fsfg` 原版CUB、`sceg_dogs_baseline` 的作者Dogs与旧DN4/技术检查、`sceg_cars_author`、`sceg_proker`、`sceg_proker_dogs`。这是对现有代码、实际运行记录和当前进程的只读审计；没有重跑训练、模型前向、GPU benchmark，也没有修改任何实验实现或停止其他任务。

**结论：确实存在严重的历史评测减速、会保留GPU资源的暂停方式，以及可避免的重复诊断/计算；但没有证据说明所有历史工作都在严重浪费算力。最严重的已证实问题集中在部分many-way评测及中断后的重放成本。公共训练器有大量诊断同步，但独立耗时尚未测量，不能把它直接宣布为训练的主要瓶颈。**

本次当前状态核查：Linux未见SCEG训练/评测进程，Windows未见Python进程，screen无会话；GPU约993 MiB占用、5%利用率。未发现当前仍有遗留SCEG任务占GPU。此结论是当前快照，不是对所有历史并发状态的证明。

## 1. 优先级与证据强度

| 优先级 | 问题 | 证据结论 | 状态 |
|---|---|---|---|
| 高 | Cars旧49-way整batch评测严重低效 | 约10 GiB共享GPU内存、CUDA同步等待、短时约34秒/任务；分块+新进程后完整评测约1.03/1.27秒 | 当前Cars已处理；不能把全部改善归因某一个因素 |
| 高 | Dogs两套协议30-way 1-shot异常慢 | 2000任务分别4.80h和3.65h，明显不随图片量合理缩放；作者30-way 5-shot扣暂停后反而更快 | 历史原因未证实，不应为审计重新跑完整实验 |
| 中 | SIGSTOP暂停仍占用显存；中断未完成设置必须重放 | Dogs/DN4有明确暂停与恢复记录；CUB、Dogs、Cars均有未完整落盘项的重放成本 | 当前无遗留进程；Cars采用退出释放显存 |
| 中 | 每训练batch逐参数finite检查和完整梯度范数 | 实际调用数以万计，逐张量检查数以百万计；Python读取CUDA布尔结果引入同步 | 机制明确、时间占比未知；属于下一轮训练优化候选 |
| 中低 | ProKeR同一episode多次fit、全量重复baseline验证 | 源码与已执行记录确认；但整个流程只有数分钟 | 可以简化，收益上限远小于数小时many-way异常 |
| 低 | 每epoch激活统计、逐epoch诊断文件 | CUB/旧DN4实际执行；作者Dogs/Cars已关闭 | 不是重复前向，文件量很小，非数小时慢速主因 |
| 低 | 输入重复哈希、少量重复预检 | 有实际额外CPU工作，但可测部分为秒级 | 不应优先优化，更不应扩大审计反复重跑 |

“高”表示值得优先避免的资源损失；不表示已找到唯一底层bug。“中”中的诊断开销不能在没有分项计时的情况下换算成某个确定加速比。

## 2. 正式训练：没有发现三套作者基线被无故反复从头训练

| 正式基线 | 完整epochs | 实际训练steps | 训练/验证记录的合计时长 | 判断 |
|---|---:|---:|---:|---|
| CUB原版 | 800 | 21,600 | 25,109s，约6.97h | 前122轮后精确续训至800，不是两次800轮 |
| Dogs作者split | 800 | 43,985 | 约12.79h | 数据更多，weighted-depletion每epoch任务更多 |
| Cars作者split | 800 | 38,400 | 38,216s，约10.62h | 一次随机初始化训练，后续只有评测 |

不同数据集的epoch并不等长。训练采用作者剩余样本消耗语义，不能因为Dogs/Cars每epoch比CUB长，就判定程序做了重复工作；也不能为了表面提速改成统一固定episode数。

CUB的40次验证合计约6,181s，占上述训练/验证记录24.6%；它是每20轮选best的协议计算，不能当作无效开销删除。证据：[CUB报告](/root/projects/aris/sceg_fsfg/docs/cub_reproduction_report.md:18)、[CUB逐轮指标](/root/projects/aris/sceg_fsfg/experiments/sceg_original/runs/20261010_cub800_seed42_resume122/metrics.json:9)、[Dogs报告](/root/projects/aris/sceg_dogs_baseline/experiments/sceg_dogs_author/runs/dogs800_seed42_author_v1/REPORT.md:20)。

Cars有较完整的分项计时：

- train（包含数据加载、前后向、优化器和训练诊断）：32,857.50s。
- validation：5,273.10s。
- 总epoch时间减去上述两项：85.66s，约0.22%，主要包含scheduler、checkpoint保存与相关记录开销；它不是纯磁盘benchmark。

因此，虽然滚动写800次约130MB的latest意味着较大的累计磁盘写入量，**Cars的现有实测不支持“checkpoint写盘导致严重训练拖慢”**。两份文件滚动覆盖也不等于长期保存800份权重。证据：[Cars逐轮记录](/root/projects/aris/sceg_cars_author/experiments/sceg_cars_author/runs/cars800_seed42_author_v1/epochs.jsonl)、[保存和计时代码](/root/projects/aris/sceg_cars_author/trainers/trainer.py:110)。

旧Dogs DN4训练不能直接复用为作者split基线：62/120类归属不同，旧训练集包含14个作者测试类。后续重新训练作者split有方法学必要；旧实验仍是其公开split的历史证据，不能全算“无效GPU浪费”。更早的Dogs技术阶段只有2轮、14个合成任务；正式数据关卡退出约0.213s、GPU分配为0，没有隐藏的正式长训。[划分对照](/root/projects/aris/sceg_dogs_baseline/experiments/_archive/sceg_dogs_dn4/dogs800_seed42_dn4/data_protocol/author_split_comparison.json:35)、[历史技术报告](/root/projects/aris/sceg_dogs_baseline/experiments/sceg_dogs_baseline/REPORT.md:79)。

## 3. 最明显的性能问题：部分many-way评测远慢于其他设置

| 设置 | 任务数 | 已记录耗时 | 每任务 | 口径/限制 |
|---|---:|---:|---:|---|
| CUB 50-way 5-shot | 2000 | 4035.3s | 2.0177s | 1050张/任务，完整整batch推理 |
| Dogs作者 5-way 1-shot | 2000 | 215.56s | 0.1078s | 85张/任务 |
| Dogs作者 30-way 1-shot | 2000 | 17290.95s | 8.6455s | 510张/任务；耗时约80倍，图片量仅6倍 |
| Dogs旧DN4 30-way 1-shot | 2000 | 13152.70s | 6.5764s | 同样严重慢；历史来源不同，不能混合分数 |
| Dogs作者 30-way 5-shot | 2000 | 3213.40s | 原始墙钟1.6067s | 包含约27.1分钟SIGSTOP；扣暂停后约1587s、约0.79s/任务，只是活跃时间估计 |
| Cars旧49-way 1-shot | 未完整保存 | 从12:00至22:10被主动终止 | 短时观察约34s；终止前观察到1431任务 | 不能当2000任务完整平均；约10h墙钟不等于精确GPU kernel计算时间 |
| Cars优化后49-way 1-shot | 2000 | 2068.88s | **1.0344s** | FP32、chunk128，峰值allocated **1.4022 GiB** |
| Cars优化后49-way 5-shot | 2000 | 2531.28s | **1.2656s** | FP32、chunk128，峰值allocated **1.4042 GiB** |

来源：[CUB50-way日志](/root/projects/aris/sceg_fsfg/experiments/sceg_original/runs/20261010_cub50way5shot_seed42/train.log:70)、[Dogs报告](/root/projects/aris/sceg_dogs_baseline/experiments/sceg_dogs_author/runs/dogs800_seed42_author_v1/REPORT.md:13)、[DN4原始结果末尾计时](/root/projects/aris/sceg_dogs_baseline/experiments/_archive/sceg_dogs_dn4/dogs800_seed42_dn4/run/test_30way_1shot.json:1031158)、[Cars慢速诊断](/root/projects/aris/sceg_cars_author/experiments/sceg_cars_author/profile_output/slow_eval_20261016.md)、[Cars完整结果报告](/root/projects/aris/sceg_cars_author/experiments/sceg_cars_author/runs/cars800_seed42_author_v1/report.md)。

这组对照有两个重要约束：

1. CUB、Dogs、Cars复现的ResidualBlock都使用命名变量同时保留前层特征、提前计算downsample与非原地残差相加；作者固定版覆盖`out`、末端downsample并原地相加。本地写法确实增加无梯度推理的存活张量，但**同样写法的CUB仍达2.02秒/任务**，所以不能把Cars和Dogs的所有异常都归咎这一个差异。[本地block](/root/projects/aris/sceg_fsfg/models/backbone/resnet12.py:85)、[官方block](https://github.com/Legenddddd/SCEG/blob/f7f130aeb5b1b49246150574c53a874d4ce8de80/models/backbones/ResNet.py#L94)。
2. Cars曾实测到Windows WSL GPU进程约15.22 GiB专用加9.99 GiB共享内存、CPU停在CUDA标量同步处。新进程加chunk128后吞吐和内存均改善。**共享内存压力确有证据，但新进程和分块是联合干预，未隔离cuDNN workspace、allocator历史、Windows资源竞争或驱动等待等因素。** Dogs历史没有同等详细的驱动/内存快照，不得套用Cars诊断为其确定病因。

当前Cars四项在2026-10-16 23:47:33完成。此次只读复核确认四项各2000任务、相同best SHA、采样索引/类别顺序/统计身份有效；第660轮确为逐轮验证均值最高者。原两份权重与两个5-way文件哈希未变。最终准确率/95% CI为：5w1 90.2525±0.3334；5w5 96.9294±0.1461；49w1 62.1985±0.1213；49w5 82.5095±0.0663（百分数/百分点）。这是单seed模型下episode抽样不确定性，不是跨seed稳定性结论。

## 4. 训练诊断：代码确认的额外GPU/CPU同步，尚不能断言严重占比

[gradient_summary](/root/projects/aris/sceg_cars_author/utils/diagnostics.py:19)对每个有梯度参数执行`torch.isfinite(...).all()`并在Python条件中读取结果，然后另算该参数梯度平方和，最终取整体范数`.item()`。[trainer](/root/projects/aris/sceg_cars_author/trainers/trainer.py:83)每个训练batch都调用它。

CUDA上的布尔结果被Python读取会要求结果可用，形成host/device同步；此外还有独立规约kernel。按实际step与固定分支推算，仅逐参数梯度finite检查的调用量约为：

| 运行 | steps × 有梯度参数张量 | 逐张量检查次数（静态推算） |
|---|---:|---:|
| CUB正式 | 21600 × 112 | 约242万 |
| Dogs作者 | 43985 × 112 | 约493万 |
| Cars作者 | 38400 × 106 | 约407万 |

这些不是实测的CUDA同步时间总量。训练大batch卷积、反向本来就耗时；没有关闭/合并诊断的短对照，无法可靠回答“可快多少”。这里最明确的冗余是：**每个batch都做完整范数汇总，但每epoch通常只把最后一个batch的范数存入metrics**。其余范数对当时是否数值有限有间接帮助，但不是每一步优化算法所需。

其它检查包括每个输出的finite、loss finite以及loss内的标签范围断言，也会引入额外小规约/同步。审计不建议盲目删掉所有数值保护。未来如果确实要优化训练，最小方向是保留必要的每batch非有限检测、合并设备端结果后只同步一次；完整梯度范数与详细缺失参数列表可在首batch/稀疏位置记录，并在发现异常时细查。**本次不更改任何已完成训练或据此重训。**

激活统计每epoch仅在第一个batch挂hook，收集mean/std/min/max与有限值，不是多跑一次前向。[activation_statistics](/root/projects/aris/sceg_cars_author/utils/diagnostics.py:40)。旧DN4确有800个activation JSON，后来清理，占用约3.1 MiB；作者Dogs/Cars已设`record_activations=false`，公共默认仍为true。[清理记录](/root/projects/aris/sceg_dogs_baseline/experiments/sceg_dogs_author/prep/cleanup_manifest.json:100)、[作者配置](/root/projects/aris/sceg_dogs_baseline/experiments/sceg_dogs_author/config.yaml:71)。它是低优先级额外计算，文件量不能解释数小时低速。

## 5. 暂停、恢复与重复工作：确有代价，但不是整套反复重跑

- **SIGSTOP不会释放GPU资源。** 作者Dogs 30-way 5-shot曾暂停约27.1分钟并SIGCONT接续同一进程；旧DN4也曾保留显存等待后续操作。如果用户只是短暂停顿且明确想保留现场，这有用途；如果准备让出GPU或改实现，它就是不合适的暂停方式。当前Cars采用终止专属进程组、保留磁盘状态、重启缺失设置，已经释放显存。[Dogs暂停](/root/projects/aris/sceg_dogs_baseline/experiments/sceg_dogs_author/runs/dogs800_seed42_author_v1/pause.json:13)、[Cars暂停](/root/projects/aris/sceg_cars_author/experiments/sceg_cars_author/runs/cars800_seed42_author_v1/pause.json)。
- **设置只在最后整体落盘，异常时会损失部分计算。** CUB早期many-way阶段09:11–14:14约5h03m，只留下完整1-shot结果，5-shot完成数量未知；随后单独完成5-shot约67分钟。不能把整段5小时全算浪费。Dogs原pipeline消失原因未知，前三项复用，仅缺失5-shot重跑。Cars旧49-way 1-shot约1431个内存结果未完整保存，优化后按保存边界重放该项；没有重算两个完整5-way。[CUB暂停记录](/root/projects/aris/sceg_fsfg/experiments/sceg_original/results/20261010_cub800_seed42_resume122/pause_record.json:4)、[Dogs事件](/root/projects/aris/sceg_dogs_baseline/experiments/sceg_dogs_author/runs/dogs800_seed42_author_v1/events.jsonl:10)。
- **没有证据支持当前存在重复评测/自动复活的GPU任务。** 旧DN4记录过自动收尾监听器，后已禁用/清理。作者入口有同run锁；当前无screen会话和SCEG进程。不能把历史曾有监听器描述为现在仍有后台系统。[DN4归档说明](/root/projects/aris/sceg_dogs_baseline/experiments/_archive/sceg_dogs_dn4/dogs800_seed42_dn4/report.md:3)、[Cars锁](/root/projects/aris/sceg_cars_author/experiments/sceg_cars_author/run.py)。
- CUB补50-way 5-shot前，只重放1-shot sampler索引来恢复随机流，没有重复读图和backbone前向；这是轻量恢复，不应误报为又跑一遍2000个GPU任务。

最小策略应优先避免进入长时间低效状态：沿用Cars每100任务稀疏进度及首任务显存/时间检查。只有出现证据显示长期运行中断重放代价仍不可接受时，才评估复用一个滚动的部分结果/RNG文件；**本审计不要求立即新建逐任务checkpoint、复杂恢复框架或看门狗**。

## 6. ProKeR：缓存有效，存在小量重复准备与全量对照成本

| 流程 | 已测记录 | 判断 |
|---|---|---|
| CUB特征缓存 | 5885张val/test图，一次编码8.54s；单份约45.6MB缓存 | 命中复用，未发现反复整套编码 |
| CUB整个ProKeR流程 | prepare15.77s、参数验证80.41s、正式评测116.71s，合计约212.89s | 调参是已申明方法工作，不是无理由重训 |
| Dogs特征缓存 | 5378张test图，一次编码7.59s；单份约41.3MB缓存 | 使用CUB锁定参数，无Dogs搜索 |
| Dogs迁移全流程 | 约142.24s，含8000任务replay与8000任务正式对比 | 存在审计性重复，但总成本为分钟级 |

来源：[CUB阶段计时](/root/projects/aris/sceg_proker/experiments/sceg_proker/runs/20261012_cub_seed42_proker_v1/stage_times.json)、[Dogs日志](/root/projects/aris/sceg_proker_dogs/experiments/sceg_proker_dogs/runs/dogs_seed42_proker_transfer_v1/run.log)。

已确认可避免的重复：

1. CUB每个episode在baseline/ridge/proker三个method循环内都重新`fit()`，8000任务共约24000次支持集准备；Dogs在Ridge/ProKeR各fit一次，共约16000次。原型、标签与部分核输入准备可以共享，但需确认各方法是否修改head状态，不能未经核对直接重用可变对象。[CUB循环](/root/projects/aris/sceg_proker/experiments/sceg_proker/run.py:353)、[Dogs循环](/root/projects/aris/sceg_proker_dogs/experiments/sceg_proker_dogs/run.py:350)。
2. Dogs replay对8000任务先计算原分类器分数，再计算新head的baseline作全量等价检查；正式流程再次算baseline。它是一次性完整性检查，并非ProKeR算法每次需要做的工作。将来输入/实现身份不变时，可复用已保存核验，而不是每次无条件重复整套。[replay](/root/projects/aris/sceg_proker_dogs/experiments/sceg_proker_dogs/run.py:283)。
3. Dogs入口运行前后会重复校验输入与来源，其中含测试图像哈希；日志可见启动/收尾约1.4–1.6s，未见其主导耗时。不要为了节约这几秒先引入复杂多级缓存。[输入检查](/root/projects/aris/sceg_proker_dogs/experiments/sceg_proker_dogs/run.py:124)。

正式头性能计时还会有意调用`cuda.synchronize()`和重置峰值计数，这是为各方法可比较的计时而做的测量成本，不能与训练中不必要的逐参数同步等同。没有分项A/B数据，不能给重复fit的独立节省秒数。

缓存评测与原版整episode重复编码工作量不同：原版CUB四项约418万次episode图像输入、Dogs约266万次；缓存分别只编码5885/5378张图后索引特征。不能把两者全流程时间差直接叫ProKeR算法收益，也不能在原版baseline中未经数值/RNG身份验证就悄悄改成缓存评测。当前Cars只分块，未引入该方法变化。

## 7. 不应夸大的小型预检与重复

CUB两次full-batch200预检各约2–2.5s、零optimizer更新，确有小型重复但合计不足5s；两次四图overfit各约19s，第二次修正为正式FRN类别身份；fresh/resume smoke用于检查状态恢复，不是再次训练800轮。它们不可能解释后来的数小时评测异常。[预检说明](/root/projects/aris/sceg_fsfg/experiments/sceg_preflight/README.md:10)、[四图日志](/root/projects/aris/sceg_fsfg/experiments/sceg_debug/runs/four_images_200_steps/train.log:1)、[FRN四图日志](/root/projects/aris/sceg_fsfg/experiments/sceg_debug/runs/four_frn_base_images_200_steps/train.log:1)。

一些归档已清理原始临时诊断，因此不能精确重建全部历史CPU时间，更不能把目录数量、文件数量或代码里存在某个入口直接等同实际执行次数。本报告只在有日志/结果支持时计入实际工作。

## 8. 后续最小改进顺序

1. **高way资源边界和异常推进优先。** 以本次Cars分块方案作为经验证的资源执行选项；对未来需要运行的其他数据集只做必要验证后使用，禁止为了本次审计重跑已经完成的CUB/Dogs。发现首批或稀疏日志显著异常时及时定位，不只看GPU利用率。
2. **先明确暂停目的。** 要释放GPU/改代码，就保存可复现起点后退出专属进程；不要用SIGSTOP后误以为显存已释放。不重测已完整匹配的设置。
3. **训练诊断只在有后续训练需求时优化。** 先用一个短的必要片段判断成本，再合并finite同步、稀疏记录完整范数，关闭非必要activation统计；不改损失、优化器或协议。现有结果无需因此重训。
4. **ProKeR只处理局部重复。** 验证方法间无状态污染后复用同一episode支持集准备；已有全覆盖baseline验证可按身份复用。不新建通用缓存/调度系统。
5. **不优先折腾已证明很小的成本。** checkpoint写盘、秒级哈希和预检不应成为第一优化对象。不要为证明它们“没问题”重复跑完整回归或大型profile。

最终可审计的表述是：**历史上存在严重但设置相关的many-way慢速问题；Cars已通过资源执行修复完成，Dogs历史根因仍未确定；训练与ProKeR存在可避免开销，但其严重性需与已测时间分开。** 本次没有发现当前仍占GPU的遗留SCEG任务，也没有发现正式作者基线被无故从头反复训练。

## 本次产物与限制

审计阶段仅新增本报告。未改动模型/训练器/评测器/配置、既有结果、数据和checkpoint，未安装依赖、未启动新GPU任务。当前Cars完整结果是此前正式进程自然完成后读取验证所得。后续用户授权的Cars文件清理记录见[cleanup.json](../experiments/sceg_cars_author/runs/cars800_seed42_author_v1/cleanup.json)，版本与推送状态以Git记录为准；其他实验的历史来源仍保留原工作区路径以便追踪。
