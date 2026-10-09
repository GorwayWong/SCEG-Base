# Phase 4 — Four-image overfit report

**PASS**，运行 `four_images_200_steps`，代码commit `07388e6`，seed42。

命令：`.venv/bin/python -m tools.overfit --run-id four_images_200_steps`。
环境：Python3.10.20、torch2.6.0+cu124、torchvision0.21.0+cu124、RTX4060Ti。

数据：物理复制的CUB训练集，按类别名和文件名排序选择4类各1张；准确路径、class ID和SHA256见[数据清单](experiments/sceg_debug/runs/four_images_200_steps/data_manifest.json)。关闭augmentation/dropout/weight_decay。

四张原图复用为support和query，共8个tensor rows、4-way1-shot1-query。这是刻意泄露的记忆能力调试，不是独立query评测，不得与论文表格比较。全局class representative head改为4类，模型其他结构、残差变换及四项损失不变。

|指标|初始step1|完成200次更新后|
|---|---:|---:|
|总loss|32.99537277|0.01786221378|
|episode accuracy|100%|100%|
|global accuracy|25%|100%|
|eval-mode episode accuracy|未测|100%|
|eval-mode global accuracy|未测|100%|
|gradient norm|297.514374|0.00403194269（step200）|

episode初始就100%是同图复用的必然现象；global accuracy从25%升至100%且global loss下降才构成额外记忆学习证据。官方margin loss的4个正类项有非零下界 `4*softplus(-38*(1-0.85))=0.01336152`；因此loss不会精确到0。事先配置的门槛loss<=0.05且两种accuracy100%满足。

诊断：16,724,355参数（debug4类head），其中1,234,560是官方未调用MLP；共124个参数张量。8张tensor输入的Conv/Linear/cosine估计58,503,144,960 FLOPs，未计BN/激活/resize/pool/norm，不能当作完整精确FLOPs。全部活动gradient有限，activation有限，无NaN。

运行产物：[config](experiments/sceg_debug/runs/four_images_200_steps/config.yaml)、[metrics](experiments/sceg_debug/runs/four_images_200_steps/metrics.json)、[train.log](experiments/sceg_debug/runs/four_images_200_steps/train.log)、[checkpoint](experiments/sceg_debug/runs/four_images_200_steps/checkpoint.pth)、[diagnostics](experiments/sceg_debug/runs/four_images_200_steps/diagnostics.json)。运行目录同时保存git_commit、command、环境、完整step history、源码hash。

风险：严格全参数梯度关卡仍失败，原因见docs/gradient_flow.md；GPU adaptive maxpool反向不保证逐bit确定性。完整训练和论文结果尚未完成，overfit成功不代替这些验证。
