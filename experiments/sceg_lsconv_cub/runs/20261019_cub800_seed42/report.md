# SCEG × LSNet LSConv：CUB seed42 完成报告

状态：**completed**。实现、一次必要预检、800 轮正式训练、四项各 2000 任务的测试及一次 CPU 统计复核全部完成，训练与评测进程均正常退出。

本次仅替换 `SC.self_s`。四项测试相对历史基线均有提升，配对 95% 区间均高于 0；结果只支持本次 CUB、训练 seed42 的替换方案，不代表跨训练种子稳定性，也不能归因于 LSConv 内部某一个独立机制。未重训或重测历史基线，未进行额外消融、搜索或失败重试。

## 完整测试结果

准确率为百分数，`±` 为 95% CI 半宽；增益与配对区间为百分点。每项 query=16、2000 个任务，四项共用验证选出的 **epoch 420 best**，其验证准确率为 **94.77625%**。

| 测试 | 历史基线准确率 | LSConv 准确率 | 增益 | 配对 95% CI 半宽 | 配对 95% CI 区间 |
|---|---:|---:|---:|---:|---:|
| 5-way / 1-shot | 87.613750 ± 0.382493 | 87.893125 ± 0.379693 | +0.279375 | 0.179374 | [0.100001, 0.458749] |
| 5-way / 5-shot | 94.525625 ± 0.192395 | 94.785625 ± 0.192532 | +0.260000 | 0.094138 | [0.165862, 0.354138] |
| 50-way / 1-shot | 56.524563 ± 0.125892 | 56.754500 ± 0.124155 | +0.229938 | 0.082958 | [0.146979, 0.312896] |
| 50-way / 5-shot | 74.162313 ± 0.065189 | 74.661250 ± 0.064975 | +0.498938 | 0.055166 | [0.443771, 0.554104] |

直接复用历史 `episode_indices`，验证 `sample_index_paths`、类别及图像顺序一致后，用本轮编码器重新编码。配对使用同一任务的新旧准确率差值，区间半宽为 `1.96 * std(diff, ddof=0) / sqrt(2000)`，没有相减两个独立 CI。准确率区间沿用相同的总体标准差公式。CPU 复核已完成，没有为复核重新进行 GPU 评测。

机器可读汇总：[results.json](results.json)。四份逐任务记录：[5-way/1-shot](test_5way_1shot.json)、[5-way/5-shot](test_5way_5shot.json)、[50-way/1-shot](test_50way_1shot.json)、[50-way/5-shot](test_50way_5shot.json)。每份包含 2000 个准确率、历史索引文件路径及哈希引用，未复制历史索引或保存特征。

对应历史基线记录：

- [5-way/1-shot](/root/projects/aris/sceg_fsfg/experiments/sceg_original/runs/20261010_cub800_seed42_resume122/test_5way_1shot.json)
- [5-way/5-shot](/root/projects/aris/sceg_fsfg/experiments/sceg_original/runs/20261010_cub800_seed42_resume122/test_5way_5shot.json)
- [50-way/1-shot](/root/projects/aris/sceg_fsfg/experiments/sceg_original/runs/20261010_cub800_seed42_resume122_test50/test_50way_1shot.json)
- [50-way/5-shot](/root/projects/aris/sceg_fsfg/experiments/sceg_original/runs/20261010_cub50way5shot_seed42/test_50way_5shot.json)

## 接入与改动文件

```python
# F_s: [B, 320, 10, 10]
enhanced_stage3 = F_s * gate_4_to_3 + LSConv(F_s)
embedding_stage3 = t_s(enhanced_stage3)
```

LSConv 是 **LSNet 作者的方法**，本轮贡献仅是向 SCEG 的指定位置迁移。固定来源为 THU-MIG/lsnet 的 `cbe737c92b7c43ecf02d08545a07f03f1010177c`：[模块定义](https://github.com/THU-MIG/lsnet/blob/cbe737c92b7c43ecf02d08545a07f03f1010177c/model/lsnet.py)、[SKA 算子](https://github.com/THU-MIG/lsnet/blob/cbe737c92b7c43ecf02d08545a07f03f1010177c/model/ska.py)。保留来源署名；该固定版本未发现根目录许可证文件，未自行附加许可声明。

保留官方 LKP 的 `lks=7, sks=3, groups=8`，以及 SKA、GroupNorm/BatchNorm、ReLU、零填充和内部输入残差。动态核为 `[B,40,9,H,W]`，以 `channel_index % 40` 选择，非连续 8 通道共享。没有额外外层残差、softmax、门控零初始化、瓶颈或学习率分组。

先完整构造原 SCEG，再在 CPU `torch.random.fork_rng(devices=[])` 中构造 LSConv 并替换 `SC.self_s`，之后创建优化器；未再次初始化整个模型。预检确认未改动参数初始化与 CPU 随机状态精确一致，不据此声称训练轨迹相同。禁用替换时不构造新参数、不导入 Triton；训练、续训及评测使用同一构造函数并严格加载权重。

| 文件（相对工作树根目录） | 改动 |
|---|---|
| `experiments/sceg_lsconv_cub/lsconv.py`, `ska.py` | 只移植必要的官方模块及 Triton 核心，保留来源 |
| `experiments/sceg_lsconv_cub/model.py` | 隔离随机状态后替换 `SC.self_s` |
| `experiments/sceg_lsconv_cub/config.yaml`, `run.py` | 固定 CUB 配方、统一构造、原续训状态、独立评测、历史任务配对及统计 |
| `experiments/sceg_lsconv_cub/check.py` | 一次最小预检 |
| `experiments/sceg_lsconv_cub/README.md`, `AGENTS.md` | 来源、运行入口和关键约束 |
| `trainers/trainer.py` | 默认不影响旧入口的轻量开关、评测器选择与关闭训练末尾自动测试 |
| `evaluators/fewshot_eval.py` | 复用分块编码逻辑、历史索引加载与局部记录选项 |

原门控、其他自增强、通道投影、池化、FC、三路余弦分类和 E/G 损失保持原语义。执行优化包括 detach 后按 epoch 累计、移除本轮非必要梯度/激活审计、last 每 20 轮保存，以及独立评测分块 128。没有额外数值配方变化；分块仅用于无梯度编码，拼接后整任务分类，DropBlock 每逻辑任务推进一次。

## 有效配方与预检

训练 seed42，800 epochs；10-way/5-shot/15-query，每任务 200 张图。SGD lr=0.1、momentum=0.9、Nesterov=true、weight_decay=5e-4；StepLR step_size=400、gamma=0.1。I2L lamda=38、inter=0.05、intra=0.85，保留原损失对象生命周期和调度顺序。每 20 轮验证 5-way/5-shot/16-query、1000 个任务，仅严格改善更新 best；40 次验证全部完成，无提前停止。

环境为 WSL2、RTX 4060 Ti 16GB、Python 3.10.20、PyTorch 2.6.0+cu124、Triton 3.2.0。workers=4、CPU 线程=2、pin_memory=false、FP32、AMP/TF32=false，沿用确定性设置。数据划分为 100/50/50 类，对应 5903/2927/2958 张图，身份匹配既有清单；数据只读引用，未重做全盘哈希或全量解码。

预检在独立进程完成，不保存随机权重、不影响正式随机流；记录的检查耗时为 11.422 秒。

| 检查 | 结果与耗时 | 峰值 CUDA 分配量 |
|---|---|---:|
| A：SKA 前向及一阶梯度 | `[2,320,10,10]` 与 PyTorch 参考通过；非均匀上游梯度，`%40` 与零填充均验证；atol/rtol=1e-5。前向/输入梯度/核梯度最大绝对误差为 0 / 2.861e-6 / 0 | 6,923,264 B |
| A：编译与稳态 | 首次编译及前反向 1.586 s；稳态前反向 0.000559 s | 同上 |
| B：真实训练更新 | 2 次预热 + 3 次稳态更新；损失有限，全部 LSConv 参数收到有限梯度并更新；稳态约 0.902 s/更新，保留官方闲置参数例外 | 10,441,962,496 B |
| C：最大评测形状 | 验证集 1050 张图、chunk=128，拼接后得分 `[800,50]`；有限输出，DropBlock 逻辑增量为 1；1.304 s | 1,540,883,456 B |

启动前显存、WSL/宿主内存和磁盘检查通过；当时承载磁盘可用约 853 GiB，为两份权重及原子替换临时文件预留 1 GiB。训练过程中显存峰值稳定，交换空间未持续增长。

## 参数量、正式耗时与资源

参数量由 **16,785,795** 降至 **16,723,995**，减少 **61,800（0.368%）**。

表中显存为 PyTorch 记录的峰值 CUDA 分配量，不是整卡占用；正式训练耗时包含 40 次验证。测试在训练进程退出后独立执行，未同时驻留训练优化器。

| 阶段 | 耗时 | 峰值 CUDA 分配量 |
|---|---:|---:|
| 800 轮训练及验证 | 25,399.295 s（7 h 03 min 19 s） | 10,442,014,208 B（9.725 GiB） |
| 5-way / 1-shot | 218.896 s | 1,029,864,448 B（0.959 GiB） |
| 5-way / 5-shot | 267.448 s | 1,248,326,656 B（1.163 GiB） |
| 50-way / 1-shot | 2,059.146 s | 1,505,247,232 B（1.402 GiB） |
| 50-way / 5-shot | 2,536.975 s | 1,507,293,184 B（1.404 GiB） |

四项测试合计 5,082.465 s（1 h 24 min 42 s）；训练与测试阶段合计约 **8 h 28 min**，不含进程启动、阶段间隔及预检。40 次验证记录耗时合计 5,371.865 s。

历史基线已知训练峰值为 10,452,678,144 B（9.735 GiB），本轮少约 10.170 MiB；历史 50-way/5-shot 记录峰值为 11,571,371,520 B（10.777 GiB），耗时约 4,046.1 s。历史来源分别为原 `20261010_cub800_seed42_resume122/metrics.json` 与 `20261010_cub50way5shot_seed42/evaluation_progress.json`/报告。评测分块及轻量执行方式发生变化，**这些资源差异不能作为 LSConv 本身的独立效率收益**。

未解决的比较限制：历史基线没有已核实的单一完整训练总耗时，其他三项历史测试的耗时/显存未完整确认，故不提供这些项目的加速比。无需追加实验补齐。没有执行阻塞或未完成的授权任务。

## 产物与续训入口

独立工作树：`/root/projects/aris/sceg_lsconv_cub`；分支：`experiment/sceg_lsconv_cub`；基础提交：`8009b6b118d9f7d1c0c4a889de972c6482ce63f5`。实验执行期间未提交、推送或合并，未修改原工作树中的数据和历史产物；后续提交以 Git 历史为准。

提交整理仅补充结果文档，并清理 `lsconv.py`、`ska.py` 的尾部空白；已确认两份模块与执行时快照的 Python AST 完全一致。`run_info.json` 中的源码快照及哈希继续对应实际训练时的文件。

单一运行目录：`/root/projects/aris/sceg_lsconv_cub/experiments/sceg_lsconv_cub/runs/20261019_cub800_seed42/`。

- [run_info.json](run_info.json)：有效配置、执行命令、环境、源码快照/哈希（含未跟踪新文件）、差异、基线引用、预检及各阶段状态。
- [metrics.jsonl](metrics.jsonl)：800 条 epoch 汇总、40 次验证汇总及每 100 测试任务的进度。
- [checkpoint.pth](checkpoint.pth)：epoch 420 验证 best，67,088,121 B；SHA-256 `e785aad48a67d350a75b28fa3e75a3ea144cb7616da11740e08018f0cd4a8c74`。
- [latest.pth](latest.pth)：epoch 800 完整续训状态，129,094,758 B，包含优化器、调度器、随机状态及 DropBlock 计数。目录中仅这两份权重。
- 四份测试 JSON 与 [results.json](results.json)：逐任务准确率、历史索引引用、共同权重身份及 CPU 配对统计。

执行命令（工作目录为上述独立工作树；本轮已全部执行成功，不需重跑）：

```bash
rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_lsconv_cub.check
rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_lsconv_cub.run --mode train
rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_lsconv_cub.run --mode evaluate
```

同身份中断运行的原续训入口如下；本轮已完成 800 轮，不需要再续训或微调：

```bash
rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_lsconv_cub.run --mode train --resume
```
