# CUB reproduction report

## 实验目的与当前状态

按固定官方代码复现CUB baseline，不调参、不修改已验证模型/loss/dataset。800轮训练完成，三组正式测试完成；2026-10-10按用户要求暂停50-way 5-shot。完整完成标准尚未满足，不能声明完整CUB baseline reproduced。

## 数据来源

FRN固定commit `64e43f3ca3a4953445788fc853e288e032d0083c`；原始ID偶数train、mod4=1 val、mod4=3 test。100/50/50类，5903/2927/2958图，总11788。正式根`data/frn/CUB_fewshot_raw`。tree SHA256 `a390d9ae0840bf6fd27311b937ee57a676942aa60c0598155a274fe7d21f0cb0`。

Phase A全部图片hash、decode/RGB与5885预缩放PNG像素对照通过，manifest保存在`experiments/sceg_original/runs/20261009_cub800_seed42_phase_a_verified/data_manifest.json`。原始记录保存的是实验准备时状态；实验后的内部JPEG去重另见清理记录。

## 配置

官方commit `f7f130aeb5b1b49246150574c53a874d4ce8de80`。10-way5-shot15-query；SGD lr0.1、momentum0.9、Nesterov、weight_decay5e-4；StepLR400 gamma0.1；800 epochs；lambda38/inter0.05/intra0.85；seed42。workers4为用户批准的执行差异，官方48。所有参数在`experiments/sceg_original/configs/cub.yaml`与run/config_resolved.yaml显式记录。

## 训练细节

使用既有`tools.train.main → trainers.trainer.train`。前序训练到122轮后中断，恢复模型/SGD/StepLR/RNG/DropBlock状态并训练至800轮；完整800轮历史保存在最终run的metrics.json。每20轮5-way5-shot、1000 episodes验证，以严格提升选best。best epoch 480，validation 95.02125%。总参数/可训练标记参数16785795；保留官方12个无梯度张量，112个活动张量通过检查。200张84×84图的支持算子前向FLOPs估算1463450445592，不包含backward或未支持算子。

独立预检epoch1→resume epoch2的loss34.962846→20.538388，状态与不中断运行对齐。最终仅保留last/best与当前RNG；完整权重不逐轮归档。

## 测试协议

使用同一validation-selected best checkpoint。三组已完成测试各2000 episodes、query16，mean episode accuracy，95% CI半宽1.96×sqrt(var(ddof=0)/n)。原始episode indices/accuracies保留，均值、CI与episode数量独立重算通过。best SHA256 `bf3c1079739c05d899c3227c7d8583fb40adec8a65eb67500e36749c046ba942`。50-way 5-shot进程已停止；原入口不保存中途episode状态，后续恢复评测仍使用此best checkpoint，无需重新训练。

## 结果

Accuracy (%)；95% CI为官方正态区间半宽。已完成项目各2000 episodes，query16，seed42。

|Method|5-way 1-shot|5-way 5-shot|50-way 1-shot|50-way 5-shot|
|---|---:|---:|---:|---:|
|SCEG reproduced (partial)|87.61 ± 0.38|94.53 ± 0.19|56.52 ± 0.13|—（暂停）|
|Paper|87.79|94.56|56.02|73.69|

50-way 5-shot尚无完成结果；不将已处理但未持久化的episodes计为正式结果。论文CI未补填。

论文：[CVF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhao_From_Few-way_to_Many-way_Rethinking_Few-shot_Fine-grained_Image_Classification_CVPR_2026_paper.pdf)。

权威结果：`experiments/sceg_original/results/20261010_cub800_seed42_resume122/results.json`。低于论文的两项5-way结果已按split→sampler→augmentation→optimizer→loss核查，报告`experiments/sceg_original/runs/20261010_cub800_seed42_resume122/reproduction_debug_report.md`；未调参或重新训练。

## 完成标准

- [x] CUB manifest与官方配置冻结
- [x] 800轮训练完成，best checkpoint可加载
- [x] 5-way 1-shot/5-shot各2000 episodes
- [x] 50-way 1-shot 2000 episodes
- [ ] 50-way 5-shot（用户暂停）
- [x] result.json/results.json与discrepancy记录

单seed结果不证明跨seed或逐bit一致性；workers、历史环境及官方行为差异见cub_discrepancy.md。
