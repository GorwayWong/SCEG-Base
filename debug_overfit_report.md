# Phase 4 — Four-image overfit report

**PASS**，运行 `four_frn_base_images_200_steps`，代码commit `40f2b978bb8508f44cf3ba29d74a9fc1b3f013e8`，seed42。旧划分首轮实验的独有事实合并在本文末尾，原始日志与指标保留。

命令：`.venv/bin/python -m tools.overfit --run-id four_frn_base_images_200_steps`。
环境：Python3.10.20、torch2.6.0+cu124、torchvision0.21.0+cu124、RTX4060Ti。

数据：遵循FRN原始class IDs1..200规则的CUB训练集，根data/frn/CUB_fewshot_raw。按类别名和文件名排序选择原始ID002/004/006/008各1张；准确路径、class ID和SHA256见[数据清单](experiments/sceg_debug/runs/four_frn_base_images_200_steps/data_manifest.json)。关闭augmentation/dropout/weight_decay。

四张原图复用为support和query，共8个tensor rows、4-way1-shot1-query。这是刻意泄露的记忆能力调试，不是独立query评测，不得与论文表格比较。全局class representative head改为4类，模型其他结构、残差变换及四项损失不变。

|指标|初始step1|完成200次更新后|
|---|---:|---:|
|总loss|34.17801285|0.01941306517|
|episode accuracy|100%|100%|
|global accuracy|25%|100%|
|eval-mode episode accuracy|未测|100%|
|eval-mode global accuracy|未测|100%|
|gradient norm|345.333588|0.004865（step200，四舍五入）|

episode初始就100%是同图复用的必然现象；global accuracy从25%升至100%且global loss下降才构成额外记忆学习证据。官方margin loss的4个正类项有非零下界 `4*softplus(-38*(1-0.85))=0.01336152`；因此loss不会精确到0。事先配置的门槛loss<=0.05且两种accuracy100%满足。

诊断：16,724,355参数（debug4类head），其中1,234,560是官方未调用MLP；共124个参数张量。8张tensor输入的Conv/Linear/cosine估计58,503,144,960 FLOPs，未计BN/激活/resize/pool/norm，不能当作完整精确FLOPs。全部活动gradient有限，activation有限，无NaN。

运行产物：[config](experiments/sceg_debug/runs/four_frn_base_images_200_steps/config.yaml)、[metrics](experiments/sceg_debug/runs/four_frn_base_images_200_steps/metrics.json)、[train.log](experiments/sceg_debug/runs/four_frn_base_images_200_steps/train.log)、[checkpoint哈希清单](experiments/sceg_debug/runs/four_frn_base_images_200_steps/checkpoint_manifest.json)、[diagnostics](experiments/sceg_debug/runs/four_frn_base_images_200_steps/diagnostics.json)。checkpoint.pth保留本地；运行目录同时保存git_commit、准确的模块命令、cwd、环境、完整step history、源码hash。

风险：严格全参数梯度关卡仍失败，原因见docs/gradient_flow.md；GPU adaptive maxpool反向不保证逐bit确定性。完整训练和论文结果尚未完成，overfit成功不代替这些验证。

## 历史首轮：旧划分上的四图调试

运行 `four_images_200_steps`，源码commit `07388e6`，seed42；命令 `.venv/bin/python -m tools.overfit --run-id four_images_200_steps`。使用旧0-based划分根data/CUB_fewshot_raw的训练类001/003/005/007各1张；该class身份不等于FRN正式训练集，结果仅属记忆能力调试。

环境、4-way1-shot1-query同图复用、全局4类head、200 steps、关闭augmentation/dropout/weight_decay、损失公式和预设门槛均与上文相同。参数数16,724,355，官方未调用MLP参数1,234,560，124个参数张量，8-row FLOPs估计58,503,144,960；活动梯度和activation有限，没有NaN，严格全参数梯度例外仍存在。

|指标|step1|200次更新后|
|---|---:|---:|
|loss|32.99537277|0.01786221378|
|episode accuracy|100%|100%|
|global accuracy|25%|100%|
|eval episode/global accuracy|未测|100%/100%|
|gradient norm|297.514374|0.00403194269（step200）|

历史证据：[图片清单](experiments/sceg_debug/runs/four_images_200_steps/data_manifest.json)、[配置](experiments/sceg_debug/runs/four_images_200_steps/config.yaml)、[指标](experiments/sceg_debug/runs/four_images_200_steps/metrics.json)、[日志](experiments/sceg_debug/runs/four_images_200_steps/train.log)、[诊断](experiments/sceg_debug/runs/four_images_200_steps/diagnostics.json)、[checkpoint清单](experiments/sceg_debug/runs/four_images_200_steps/checkpoint_manifest.json)。首轮command.txt缺少-m的记录问题见[原始调用更正](experiments/sceg_debug/runs/four_images_200_steps/invocation_correction.md)。独立checkpoint和全部step history仍保留，不能将这组数字替代正式FRN调试结果。
