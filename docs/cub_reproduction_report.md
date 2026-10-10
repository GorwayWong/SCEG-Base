<!-- run_id: 20261010_cub800_seed42_completed -->
# CUB baseline reproduction report

## 实验目的

复现官方SCEG CUB baseline；不提升指标，不修改已验证模型/loss/dataset，不调参。完成标准通过后记录 `SCEG CUB baseline reproduced`；这代表协议执行完成，不代表论文数字逐bit或统计一致。

## 数据来源

FRN固定commit `64e43f3ca3a4953445788fc853e288e032d0083c`，原始ID偶数train、mod4=1 val、mod4=3 test。100/50/50类，5903/2927/2958图，总11788；tree SHA256 `a390d9ae0840bf6fd27311b937ee57a676942aa60c0598155a274fe7d21f0cb0`。Phase A全图decode/RGB/hash与5885预缩放图逐像素对照通过；manifest见 `experiments/sceg_original/runs/20261009_cub800_seed42_phase_a_verified/data_manifest.json`。

## 配置

官方commit `f7f130aeb5b1b49246150574c53a874d4ce8de80`；10-way5-shot15-query；SGD lr.1、momentum.9、Nesterov、weight_decay5e-4；StepLR400 gamma.1；800 epochs；lambda38/inter.05/intra.85；seed42。workers4是用户授权的唯一训练执行参数差异，官方48；AMP/TF32关闭，deterministic warn_only保留。所有参数在config_resolved.yaml显式记录。

## 训练细节

通过现有 `tools.train.main -> trainers.trainer.train`，补充记录层没有替换训练算法。正式run `experiments/sceg_original/runs/20261010_cub800_seed42_resume122` 从已完成122epochs的checkpoint续训至800epochs；原始seed为42，前序记录通过continuation.json追溯，metrics.json包含合并的完整epoch历史。每20轮5-way5-shot验证1000episodes，以strictly improved accuracy选best：epoch 480，val 95.0212%。全部epoch保留轻量指标、日志及验证证据；完整权重仅保留rolling last和validation选出的best，RNG只保留当前状态，不逐轮归档。预检独立重启后loss34.962846→20.538388。参数 16785795，trainable 16785795；官方闲置参数保留。200图eval支持算子前向FLOPs估算 1463450445592，非全算子/非backward计数。

## 测试协议

800epochs后不重训练，使用同一best checkpoint。5-way1/5-shot由既有trainer自动执行；50-way1/5-shot通过既有tools.test执行。各2000episodes、query16，mean episode accuracy与1.96×sqrt(var(ddof=0)/n)；原始indices/accuracies保留且独立重算通过。

## 结果

结果见 `docs/cub_result_table.md` 与 `experiments/sceg_original/results/20261010_cub800_seed42_completed/results.json`，best SHA256 `bf3c1079739c05d899c3227c7d8583fb40adec8a65eb67500e36749c046ba942`。本次补充50-way 5-shot评测耗时 4046.1秒；训练及前三组测试的耗时保留在既有run记录中。

Accuracy (%)，95% CI为官方正态区间半宽，每项2000 episodes、query16、seed42。

|Method|5-way 1-shot|5-way 5-shot|50-way 1-shot|50-way 5-shot|
|---|---:|---:|---:|---:|
|SCEG reproduced|87.61 ± 0.38|94.53 ± 0.19|56.52 ± 0.13|74.16 ± 0.07|
|Paper|87.79|94.56|56.02|73.69|

论文：[CVF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhao_From_Few-way_to_Many-way_Rethinking_Few-shot_Fine-grained_Image_Classification_CVPR_2026_paper.pdf)。论文CI未在此补填。

部分setting低于论文，已按split→sampler→augmentation→optimizer→loss顺序核查并生成 `experiments/sceg_original/runs/20261010_cub50way5shot_seed42/reproduction_debug_report.md`；没有调整参数。

## 完成标准

- [x] CUB manifest与配置冻结
- [x] 800epochs完成
- [x] best checkpoint可加载
- [x] 5-way1/5-shot各2000episode
- [x] 50-way1/5-shot各2000episode
- [x] result.json/results.json生成
- [x] discrepancy记录完成

限制与差异见 `docs/cub_discrepancy.md`。
