# SCEG CUB 纯视觉 ProKeR 迁移实验

状态：代码与必要测试完成；验证选参完成；四组正式评测完成。工程判断：**continue**。

本轮使用已有 seed42、epoch480 的 SCEG best，全部参数冻结；无重训练、文本、CLIP、外部预训练或测试增强。

## 结果

Accuracy (%) ± 原版95% episode CI半宽；增益与成对CI单位为百分点。每组2000 episodes，query=16。

|Setting|Method|Accuracy ± CI|Gain ± paired CI|Head ms/episode|Peak GPU MiB|
|---|---|---:|---:|---:|---:|
|5way_1shot|baseline|87.6137 ± 0.3825|+0.0000 ± 0.0000|1.449|54.6|
|5way_1shot|ridge|87.6425 ± 0.3823|+0.0288 ± 0.0197|1.616|55.5|
|5way_1shot|proker|87.6450 ± 0.3820|+0.0312 ± 0.0272|2.629|55.5|
|5way_5shot|baseline|94.5256 ± 0.1924|+0.0000 ± 0.0000|1.539|54.9|
|5way_5shot|ridge|94.5200 ± 0.1974|-0.0056 ± 0.0816|1.670|55.8|
|5way_5shot|proker|94.7512 ± 0.1876|+0.2256 ± 0.0513|2.733|55.8|
|50way_1shot|baseline|56.5246 ± 0.1259|+0.0000 ± 0.0000|6.268|63.1|
|50way_1shot|ridge|56.5832 ± 0.1240|+0.0586 ± 0.0328|7.014|72.7|
|50way_1shot|proker|56.7582 ± 0.1237|+0.2336 ± 0.0335|8.071|72.9|
|50way_5shot|baseline|74.1623 ± 0.0652|+0.0000 ± 0.0000|6.297|67.0|
|50way_5shot|ridge|72.3418 ± 0.0651|-1.8205 ± 0.0552|7.279|76.6|
|50way_5shot|proker|75.2641 ± 0.0627|+1.1018 ± 0.0325|8.319|76.7|

实测判断：**continue**。50-way的1-shot/5-shot成对95%区间分别为[+0.2001,+0.2671]、[+1.0693,+1.1342]pp；5-way没有实测退化，1-shot的+0.0312pp属于很小的实际收益。50-way头增加约1.80/2.02ms，即约29%/32%的分类头耗时；编码只做一次，未产生训练开销。5-shot many-way是本轮最明确的工程收益，但只支持这个固定checkpoint上的结论。

线性ridge已独立调参，50-way5-shot仍下降1.8205pp；确认验证集上其最佳lambda=.1也比原型头低0.6218pp。保留这一负结果，没有测试后增强或重新选参。它说明在本轮网格、无截距和归一化拼接的设计下，线性对照未取得many-way5-shot收益，不能泛化为所有ridge设计均无效。

## 方法与来源

作者 [ProKeR CVPR2025论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Bendou_ProKeR_A_Kernel_Perspective_on_Few-Shot_Adaptation_of_Large_Vision-Language_CVPR_2025_paper.pdf) 与 [补充](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Bendou_ProKeR_A_Kernel_CVPR_2025_supplemental.pdf) 使用冻结CLIP视觉特征及文本先验，11个基准、1/2/4/8/16-shot；不是本实验的CUB episodic协议。

实际读取代码commit `f2ecb3d2a0d7e0993f977e550286b34f06015edb`：[trainer](https://github.com/ybendou/ProKeR/blob/f2ecb3d2a0d7e0993f977e550286b34f06015edb/trainers/proker.py)、[RN50配置](https://github.com/ybendou/ProKeR/blob/f2ecb3d2a0d7e0993f977e550286b34f06015edb/configs/RN50/configs_proker/hyperparameters.json)。其1-shot参数beta=3、lambda=.05；没有作者CUB/5-shot参数。补充逐页视觉图表未另复核，公式与发布代码已核查。

本次迁移：每图SCEG三路FC后未归一化z2/z3/z4，各640维；支持特征先按shot平均，再用原版 normalize_episode。f0为原三路scores/3（原预测不变，但残差拟合尺度改变），f0(S)来自当前支持集原型，不用训练基类class_representative。

核输入u=concat(F.normalize(z2),F.normalize(z3),F.normalize(z4))/sqrt(3)，K=exp[-beta(1-u·u')]。作者求解 A=solve(I+KSS/lambda,Y-f0(S))，预测 f0(q)+w KqS A。若写成(K+lambda I)^-1，预测必须乘额外lambda，不能静默替换。w是迁移新增参数，不是论文默认参数。

线性ridge在同一u上拟合one-hot、无截距：W=U_S^T solve(U_SU_S^T+lambda I,Y)，不加f0。纯归纳：query间不交互，不读取query标签或整体统计构造特征。

## 选参与隔离

每shot先18组beta={1,5,10}、lambda={.1,1,10}、w={.5,1}，固定200个5-way和200个50-way验证episodes。对50-way粗筛最佳仅做一次6个单轴邻域扩展：beta×{.5,2}、lambda×{.5,2}、w={.25,1.5}，跳过重复，合计最多24组。线性ridge独立搜索lambda={.001,.01,.1,1,10,100}。

各头按50-way粗筛取前两名，在另一固定1000个50-way验证episodes上确认；5-way仅辅助记录。同shot两种way共用参数，稳定网格顺序破平分。完整搜索表见search_table.csv，验证逐episode准确率见validation_results.json，锁定参数见locked_parameters.json。测试后不追加搜索。

锁定参数：

```json
{
  "1": {
    "proker": {
      "beta": 10.0,
      "ridge_lambda": 1.0,
      "residual_weight": 1.0
    },
    "ridge": {
      "ridge_lambda": 100.0
    },
    "proker_candidates_tested": 23
  },
  "5": {
    "proker": {
      "beta": 1.0,
      "ridge_lambda": 0.1,
      "residual_weight": 1.5
    },
    "ridge": {
      "ridge_lambda": 0.1
    },
    "proker_candidates_tested": 24
  }
}
```

独立worktree/分支 experiment/sceg_proker，全部新代码/配置/缓存/结果位于experiments/sceg_proker；历史sceg_original只读，主工作区docs/paper/保留。完整历史索引按路径映射重放，并在此实验重新评价三种方法。

## 验证与统计限制

必要测试检查作者同输入公式、w=0退化、支持排列不变、query单独/分块一致、线性ridge原/对偶等价和正参数。另执行一次既有cosine代表性回归。6个真实episode在线/缓存scores与预测核验见online_cache_regression.json；BN统计保持不变。

6项新头测试加1项代表性回归：7 passed in 1.12s，真实命令及范围见../../test_record.json。正式结果的12组方法均从原始逐episode准确率独立重算均值、原版CI、增益和成对CI，全部通过；见final_verification.json。

原版95% CI半宽=1.96 sqrt(var(ddof=0)/n)；成对CI对同episode百分点差值使用同一公式。只反映固定checkpoint与数据下episode抽样不确定性，不代表多训练seed稳定性；episode复用图像使其并非独立训练重复。

历史逐episodebaseline核对：

- 5way_1shot: {"different_episode_accuracies": 0, "max_absolute_difference_pp": 0.0, "mean_difference_pp": 0.0}
- 5way_5shot: {"different_episode_accuracies": 0, "max_absolute_difference_pp": 0.0, "mean_difference_pp": 0.0}
- 50way_1shot: {"different_episode_accuracies": 1, "max_absolute_difference_pp": 0.125, "mean_difference_pp": 6.25e-05}
- 50way_5shot: {"different_episode_accuracies": 0, "max_absolute_difference_pp": 0.0, "mean_difference_pp": 0.0}

唯一差异已进一步核验：50-way1-shot第1171个episode（零基）、query offset 77，缓存原版两类sum分数恰为2.4185838699，argmax选择类4；按历史850图整episode编码，第一名为类8，第二名低4.77e-7，恢复历史53.75%准确率。缓存准确率53.875%，整episode与缓存最大sum分数差1.55e-6。缓存头f0逐元素等于缓存原版sum/3，除以3本身没有改变此episode预测。属于batch形状引发的float32临界判定差异，证据和完整可重放脚本见replay_diagnostic.json/replay_diagnostic.py。

不声称缓存重放与历史逐bit一致。本轮增益始终相对同一缓存、同一episode上重测的原型头；该历史均值差只有0.0000625pp，没有改写历史结果或重新选择参数。

## 资源、命令与记录

唯一特征缓存：`/root/projects/aris/sceg_proker/experiments/sceg_proker/cache/5d98358cdb71fc46e0af/features.pth`；验证2927图、测试2958图，每图3×640 float32，含路径映射，不存空间图/episode核/新权重。编码总耗时8.540s，编码峰值显存597.4MiB。本次入口总墙钟212.887s，含验证搜索与正式评测；细分见stage_times.json。

资源明细：prepare 15.767s、验证搜索80.413s、正式测试116.706s。单独的历史差异诊断额外前向1.719s，原850图整批导致峰值8957.5MiB（8.75GiB），这是本任务实际测到的最高torch allocated显存；日常缓存/头评测不需要这个整批大小。必要测试1.12s。缓存含元数据约43.89MiB，run记录约12.98MiB；无新网络权重。环境复用Python3.10.20、Torch2.6.0+cu124、CUDA12.4、RTX4060Ti 16GiB，cpu_threads=2、loader workers=4。全过程一次只运行本任务一个GPU进程，没有终止其他任务。

头耗时是单GPU同步墙钟，包括支持拟合和query打分，不含编码、cache gather、argmax；统一wrapper支持准备包含三路原型与u，因此baseline计时亦含少量通用准备开销。显存为torch峰值allocated，含常驻缓存、不含其他进程和CUDA驱动保留内存；GPU可同时受外部负载影响。

输入checkpoint：`/root/projects/aris/sceg_fsfg/experiments/sceg_original/runs/20261010_cub800_seed42_resume122/checkpoint.pth`，SHA256 `bf3c1079739c05d899c3227c7d8583fb40adec8a65eb67500e36749c046ba942`。已有数据清单直接引用cache_reference.json，未重复哈希全部图片。

实际命令见command.txt与invocations.jsonl，环境见environment.json，运行时基础代码版本/工作区见git_commit.txt/git_status.txt。实验源码由本分支Git提交及experiment_source_manifest.json追溯；实际执行入口保留source_snapshot/run.py，其他相同源码副本已去重。

改动位置：../../head.py是独立归纳推理头，../../run.py复用已有模型、ImageFolder/transform、sampler与ExperimentRun完成缓存和评测，../../config.yaml是专用网格与真实输入路径，../../test_head.py是必要测试。基础HEAD仍为8009b6b118d9f7d1c0c4a889de972c6482ce63f5；最终检查新增文件diff无空白错误，models/datasets/evaluators/utils/experiments/sceg_original/docs的tracked diff为空，主工作区仍仅有原先未跟踪的docs/paper/。

source_snapshot/run.py保留实际产生实验结果的入口版本；其余执行源码的位置和SHA在final_verification.json、experiment_source_manifest.json中记录，与提交源码逐字节相同。评测结束后仅给run.py增加3行“已完成run的--resume直接退出”保护，防止重写最终报告/资源计时；完整差异为post_evaluation_fix.patch，patch dry-run通过。清理前实际执行--resume后metrics/report/stage_times/lock/results/cache_reference的6个SHA均未变化。推理头、选参和正式指标没有重跑或改动；最终run.py版本与执行版本哈希在final_verification.json区分记录。

工作目录：`/root/projects/aris/sceg_proker`。已执行正式命令：

```bash
rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_proker.run --run-id 20261012_cub_seed42_proker_v1
```

可续跑：同一命令重用run_id并添加 --resume；已有cache、lock及完成的正式组直接复用，中断组按固定索引重测。没有训练恢复权重，因为本轮不训练。

```bash
rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_proker.run --run-id 20261012_cub_seed42_proker_v1 --resume
```

这次run已完整完成，当前无需续跑。新独立运行须换run_id；专用experiment仍为sceg_proker。差异诊断实际命令：

```bash
rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_proker.runs.20261012_cub_seed42_proker_v1.replay_diagnostic
```

原始episode索引位于episodes/*.npz，路径映射引用cache metadata；各test_*.json保存三种方法的逐episode准确率、配对指标和计时。不按episode保存模型或核矩阵。

工程规则：两个many-way成对CI上界均≤0则stop；两个下界均>0且few-way最差退化≤.5pp则continue；其他uncertain。它只决定后续工程判断，不触发本轮额外实验。最终判断应结合上表收益、few-way退化和头开销；完整结果可为负，不增加其他模块。

未完成项：无本轮要求内的指标；多训练seed及作者原CLIP实验不在范围内。提交和推送按后续用户授权执行，不合并到master。

## 本轮清理与提交

后续用户已明确授权清理、提交和推送本实验分支。删除results.json（与四份保留的test_*.json逐字段完全相同的汇总）、空git_diff.patch、4份与提交源码逐字节相同的快照；合计423936 bytes。清理独立worktree中本任务产生的Python字节码缓存。更新本报告与核验记录中的相关引用，未删除原始测试/验证episodes、逐任务准确率、路径映射、锁定参数、唯一最终报告或真实执行入口快照。不会合并到master。

保留单份feature cache供历史重放和后续纯推理复用，约43.89MiB；它没有重复副本且不推送.pth。提交其轻量metadata.json中的路径映射，并通过实验专用.gitignore确保10份压缩episode索引与执行入口快照进入Git。清理后run记录约12.58MiB，仍可独立核验四组指标及成对CI。
