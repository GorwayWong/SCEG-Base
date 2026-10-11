# Stanford Dogs 原版 SCEG：代码与技术检查完成，正式实验 blocked

2026-10-12。本轮未启动正式训练，四组正式指标均为 null。阻塞原因是没有可信、精确的 70/20/30 类别身份清单。原图可读和类别数量已核验，但不能据此宣称作者 FSFG 划分成立。没有随机生成划分，没有用 Stanford 的按图像 train/test 列表替代类互斥划分。

## 隔离与代码

原仓库 origin 为 `git@github.com:GorwayWong/SCEG-Base.git`，当前和历史参照 HEAD 均为 `8009b6b118d9f7d1c0c4a889de972c6482ce63f5`。原工作区只有未跟踪的 `docs/paper/`。已读取用户提供的 AGENTS 指令及 `/root/.codex/RTK.md`；当前仓库和父目录没有磁盘 AGENTS.md。已读取 README、三份指定设计/差异文档、冻结 CUB 配置、Dogs 加载器、sampler/transform、train/test 入口及 trainer/evaluator。

独立 worktree：`/root/projects/aris/sceg_dogs_baseline`，分支：`experiment/sceg_dogs_baseline`。环境使用原仓库 `.venv/bin/python`，未复制或升级环境。实验执行阶段没有 commit/push/merge；后续按用户要求将本轮代码和轻量记录提交并推送到该独立分支，不合并 master。没有扩展其他数据集、seed 或改进模块。

新增文件：

- [configs/dogs.yaml](configs/dogs.yaml)：冻结 Dogs 配置，从已有 CUB 正式配置派生。
- [prepare.py](prepare.py)：读取既有 tar，核验图像；只接受带来源文件哈希的精确 split。得到可信清单后才准备 Dogs 独立目录与官方 Resize92→CenterCrop84 的 PNG。
- [run.py](run.py)：Dogs 独立门槛、记录及正式 suite 薄入口。直接调用现有 `trainers.trainer.train` 和 `evaluate_episodes`，复用训练器完成的两个 5-way 结果，只补两个 30-way；纯推理不复制权重。
- [verify.py](verify.py)：总计两轮合成夹具训练和恢复检查；不生成真实 Dogs 划分。
- [../../tests/test_dogs_protocol.py](../../tests/test_dogs_protocol.py)：来源门槛、类别重叠、来源哈希、数据准备/变更拒绝、5-way 结果复用及同一 checkpoint 测试。

没有修改现有模型、损失、sampler、transform、trainer、evaluator、train/test 入口或 CUB 默认行为。实验执行时 `git diff HEAD` 对未跟踪新增文件为空，因此保存了 [implementation.patch](implementation.patch)，包含六个新增源码/配置/清单/测试文件，SHA256 `d68ca7b7cc8750aa31f8b5f06fec490ed0bf11d96e5ec49bb356788dd71e6598`。`git apply --reverse --check` 通过，确认补丁匹配最终源码。各运行还有实际执行版本的 `experiment_source.patch`、源码哈希、HEAD、git 状态和差异；运行记录保留执行时的 `8009b6b`，后续提交不改写历史记录。最终 Git 提交保存完整新增源码、配置与轻量实验记录。

## 数据来源与缺口

复用 `/root/projects/aris/sceg_fsfg/data/raw/dogs/images.tar`（793,579,520 bytes），来源为 `http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar`。已有 SHA256 为 `a443fc2f9a851aa1a7b8e4fa4af94b8459fce2c9f5a2f4dd4669cf9673a4ec58`；直接引用 [已有下载记录](/root/projects/aris/sceg_fsfg/data/provenance/downloads_dogs.json) 与 [归档审计](/root/projects/aris/sceg_fsfg/data/provenance/raw_archive_audit.json)，本轮检查大小并解码全部像素，没有重新哈希整个 archive，也没有复制 archive 或解压全部原图。

[raw_inventory.json](data_protocol/raw_inventory.json) 保存全部 120 个 breed 文件夹的精确名称与图像计数：总计 20,580 张，逐类 148–252 张；全部成功解码。该清单是原图 inventory，**不是 FSFG split**。正式 class-to-index 为 null；合成夹具的 ImageFolder 字典排序和类别标签映射验证通过。

已核对的来源范围：

- [固定 SCEG README](https://github.com/Legenddddd/SCEG/blob/f7f130aeb5b1b49246150574c53a874d4ce8de80/README.md) 只给出 FRN 引用和目录格式。
- [FRN](https://github.com/Tsingularity/FRN) 的已发布准备说明不提供 Dogs 的 70/20/30 精确清单。
- 同作者 [AIS-MLI](https://github.com/Legenddddd/AIS-MLI) 的 README 给出 Dogs 70/20/30 数量，但没有完整类别身份。
- 已有 [论文来源清单](../../docs/sources/manifest.yaml)、正文/补充材料核验记录：补充 §7.1 给出数量，没有完整 class IDs。原工作区未跟踪的正文 PDF 保持原样，未重复下载。
- `/root/projects/aris/fewshot/data/fsfg/provenance/` 有 CUB/Cars 来源记录，未找到可信 Dogs 类别清单。

[split.json](data_protocol/split.json) 如实保存 blocked 状态和 null 类别列表。后续需要作者或可验证发布物的完整 120 breed 归属，以及来源 URL、固定 revision、原始清单路径及 SHA256、ID 到 breed 名称的映射说明。`status: verified` 只能在来源已人工核验后设置；脚本只检查来源文件与声明的对应关系，不自动证明发布者身份。

数据准备在任何数据写入前拒绝当前 blocked 清单。获得可信清单后，准备记录保存互斥类别、class-to-index、原图与 PNG 的哈希、文件状态及像素等价性。后续训练复用该记录，并检查文件 inventory、大小和 mtime，不每次重算全部图片哈希。该轻量检查不防范刻意同时伪造文件内容和时间戳。

## 冻结设置与来源

“通用实现”表示固定官方代码或发布脚本的行为，不表示已发现 Dogs 专用完整命令。“沿用工程默认”表示当前 CUB 已验证环境的执行设置。

| 设置 | Dogs 配置 | 来源类别与依据 |
|---|---|---|
| 数据标识 / global head | stanford_dog / 70 classes | Dogs 明确设置：补充 §7.1、官方 Dogs 模型分支 |
| 类别划分 / many-way | 70/20/30；many-way=30 | Dogs 明确数量；精确身份仍缺失 |
| backbone / 输入 | ResNet12 / 84×84 | 通用实现：SCEG、ResNet 与 transforms |
| 训练 episode | 10-way 5-shot 15-query | 通用实现：发布 run.sh 与真实 trainer；不自定 episodes/epoch |
| 跨尺度分支 | stage3→stage2 开启 | Dogs 明确设置：官方 CUB/Dogs 分支 |
| I²L intra / inter | 0.85 / 0.05 | 通用 parser 和发布 CUB run.sh；未找到 Dogs 专用执行设置 |
| 固定 loss λ | 38，排除在 optimizer 外 | 沿用发布 CUB run.sh；不是已确定的 Dogs 专项参数 |
| 优化器 | SGD lr=.1, momentum=.9, Nesterov, wd=5e-4 | 通用实现/补充 §7.2 |
| 调度与轮数 | StepLR 400, gamma=.1；800 epochs | 通用 trainer 与发布脚本的 400×2 |
| 验证 | 每20 epochs，5-way 5-shot，1000 episodes，按均值选 best | 通用官方 trainer；Dogs 验证仅20类，不做30-way验证 |
| 增强 | RandomResizedCrop84、ColorJitter .4、flip .5；ImageNet normalize | 通用官方 transform，未改变 baseline 增强 |
| 评测预处理 | RGB，短边 Resize92，CenterCrop84，PNG 预处理 | 通用官方 transform_type=0，沿用冻结 CUB 路径 |
| 随机数 / 设备 | seed42，cuda:0，一个实验 GPU 进程 | 通用官方 seed；沿用工程的完整 RNG 控制 |
| workers / pin / CPU threads | 4 / false / 2 | 沿用已批准工程默认；官方 workers=48 |
| 数值环境 | FP32，AMP=false，TF32=false，deterministic warn_only=true | 沿用已验证工程默认；接受已记录 pool backward 非严格确定性 |
| 正式测试 | 5-way/30-way × 1-shot/5-shot；各2000 episodes，query16 | Dogs novel 类数 + 通用官方测试 sampler |
| CI | 1.96×sqrt(var(episode_accuracy, ddof=0)/N) | 原版95% normal episode CI；单位百分点 |

固定版 [utils/util.py](https://github.com/Legenddddd/SCEG/blob/f7f130aeb5b1b49246150574c53a874d4ce8de80/utils/util.py#L67-L69) 的 Dogs `lamda=44` **被注释**，不是运行设置。活动 parser 默认36，[发布 CUB run.sh](https://github.com/Legenddddd/SCEG/blob/f7f130aeb5b1b49246150574c53a874d4ce8de80/scripts/run.sh) 明确38。按任务采用38，未把注释当作论文配置。论文称温度可学习而发布代码不优化 λ 的既有差异继续保留。调参范围为空、预算0次，没有利用测试标签调参或选 checkpoint。

## 真实执行

所有命令在 `/root/projects/aris/sceg_dogs_baseline` 执行；实际 Python 来自原环境。完整命令与 exit code 见 [validation.json](validation.json)，运行内还有 `command.txt` 和 `cwd.txt`。

```bash
rtk git worktree add -b experiment/sceg_dogs_baseline /root/projects/aris/sceg_dogs_baseline
rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_dogs_baseline.prepare --audit-raw
rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m pytest -q tests/test_dogs_protocol.py tests/test_episode_sampling.py::test_training_nonoverlap_support_order tests/test_episode_sampling.py::test_evaluation_ignores_query_and_uses_disjoint_samples
rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_dogs_baseline.verify --run-id 20261012_dogs_seed42_technical
rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_dogs_baseline.run --run-id 20261012_dogs_seed42_formal_blocked
rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m pytest -q tests/test_dogs_protocol.py
rtk proxy git apply --reverse --check experiments/sceg_dogs_baseline/implementation.patch
```

技术 smoke：两个独立 run_id 为 `20261012_dogs_seed42_technical_epoch1` 与 `20261012_dogs_seed42_technical_resume2`，累计只训练 **2 epochs / 14 episodes**，不是两次各2 epochs。训练数据为明确标注的合成 PNG 夹具，70/20/30 合成类别；没有指定真实 breed 的任意归属。每类训练20张由原 depletion sampler 自然产生每轮7个 episode。诊断验证改为每轮、2 episodes，测试每 setting 2 episodes；这些缩减只存在技术运行的解析配置，不进入冻结正式配置。

第一轮验证后保存完整 latest 与 best，故意在完整 epoch 边界中断。第二个 run 用原 trainer 恢复，检验 model、SGD momentum、scheduler、RNG 与 DropBlock counters，最终 epoch=2、step=14。第一次训练 forward 输出 global `[200,70]`、episodic `[150,10]`，70类 head 权重 `[70,640]`。原 trainer 的有限值和活动参数梯度断言通过，保留12个官方闲置参数张量，不重新要求其梯度。

四组诊断评估各2 episodes，support-first、support/query 无重复、样本路径/索引、逐episode准确率和 ddof=0 CI 均检查通过；online resize/crop 与保存 PNG 的输入 tensor 相同。结果保存在技术 run 的 `test_*way_*shot.json`，**不作为 Dogs 指标**。技术 best epoch=1、loss/accuracy 只属夹具；正式 best epoch=null。技术权重哈希先保存，随后删除全部4个可再生诊断 `.pth` 和临时图片，不保留 synthetic checkpoint 冒充可续跑正式模型。

6个 Dogs 测试与2个既有 sampler 代表性回归均通过，共8个不同测试。数据准备 roundtrip 和 suite 复用测试新增后，仅重跑受影响的 Dogs 测试。已有 CUB 模型等价性及 overfit 结论直接引用 [冻结报告](../../docs/cub_reproduction_report.md) 和已有 `sceg_debug/result.json`；没有重复旧关卡。

## 结果与资源

| setting | 论文目标 accuracy (%) | 本地正式 accuracy / 95% CI | 正式 episodes |
|---|---:|---|---:|
| 5-way 1-shot | 77.89 | null / null | 0 / 2000 |
| 5-way 5-shot | 89.13 | null / null | 0 / 2000 |
| 30-way 1-shot | 47.55 | null / null | 0 / 2000 |
| 30-way 5-shot | 66.63 | null / null | 0 / 2000 |

论文目标来自已有正文 Table1 核验记录，来源 [CVPR2026 正文](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhao_From_Few-way_to_Many-way_Rethinking_Few-shot_Fine-grained_Image_Classification_CVPR_2026_paper.pdf)。没有本地增益或偏差可供解释，不能宣称 Dogs baseline reproduced。

机器：RTX4060Ti 16GiB，WSL2，Python3.10.20，Torch2.6.0+cu124 / torchvision0.21.0。启动前无 CUDA compute 实验进程，约1.9GiB显示占用；没有终止任何其他任务。技术完整墙钟耗时 **98.733秒**，CUDA峰值分配 **10,583,883,264 bytes（9.857 GiB）**；这不是 GPU 活跃时间。两轮训练/验证边界记录约9.059秒和7.991秒，其余时间包括 fixture、记录、诊断评测及清理。正式门槛运行0.213秒、GPU分配0，退出码2；正式训练/测试耗时=null。技术记录与源码补丁总量不到1MiB，不计只读复用的既有数据/环境；合成模型权重与图像临时空间已释放。

[results.json](results.json) 是完整状态与四个 null 指标；`result.json` 是 ExperimentRun 指向 blocked 正式 run 的摘要。技术指标不会发布覆盖正式摘要。运行目录保留解析配置、实际命令、环境、源码版本/补丁、日志、epoch记录、输入checkpoint路径/SHA和诊断episode结果。正式 episode 索引与逐任务准确率未生成，仍为 null。

## 后续执行条件与命令

本轮没有运行中的训练，也没有正式 checkpoint。唯一尚未完成的前置条件是可信的精确 class split。获得并核验来源后，填充 `data_protocol/split.json` 的三组类别及来源，再执行以下**未执行**命令；原图准备从既有 tar 直接写入独立 Dogs 根，不复制 tar、不使用 CUB checkpoint。正式命令重新 seed42 并重新初始化模型，不能从本轮合成 smoke 续训。

```bash
cd /root/projects/aris/sceg_dogs_baseline
rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_dogs_baseline.prepare
rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_dogs_baseline.run --run-id dogs800_seed42_fresh
```

若未来正式训练中断，保留该 run 的 `latest.pth` 与同目录 best `checkpoint.pth`，保持配置不变，用新的 run_id 恢复：

```bash
rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_dogs_baseline.run --run-id dogs800_seed42_resume --resume experiments/sceg_dogs_baseline/runs/dogs800_seed42_fresh/latest.pth
```

当前该命令所引用的正式权重尚不存在；它是未来恢复用法，不是本轮可恢复的正式状态。正常完成后只长期保留每个正式执行目录的滚动 `latest.pth` 与验证最优 `checkpoint.pth`；不保存逐epoch权重、不按测试setting复制best。四组评测使用同一best，记录原始episode索引和准确率。

## CUB 保护核验

旧 CUB 模型/配置/checkpoint/结果以及旧 CUB 数据和未跟踪论文共17,946个文件的路径、大小、mtime_ns、inode摘要，技术运行前后完全一致：`898b26a9a49b03caebed491cc67bcb52a734708dca236e33fe847ce85eea5b1d`。这是轻量元数据核验，未重复哈希全部图像或权重；权重原有SHA直接引用 [历史checkpoint清单](../../experiments/sceg_original/results/20261010_cub800_seed42_completed/checkpoint_manifest.json)。原 master 的状态仍只有 `?? docs/paper/`；本轮提交范围仅为 Dogs 独立实验目录和 `tests/test_dogs_protocol.py`。
