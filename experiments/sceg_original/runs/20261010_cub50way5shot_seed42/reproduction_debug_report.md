# Reproduction debug report

指标低于论文时仅核查既有证据，不调整超参数或重新训练。下述通过仅排除已检查的配置/数据错误，不能证明性能差异原因。

|Setting|Reproduced|Paper|Gap (percentage points)|
|---|---:|---:|---:|
|5-way 1-shot|87.6137|87.79|-0.1763|
|5-way 5-shot|94.5256|94.56|-0.0344|

|hypothesis|experiment / evidence|result|
|---|---|---|
|数据split不一致|Phase A原始class IDs、逐图hash与FRN固定规则核对；50-way测试前再次hash原图/缓存|100/50/50类与5903/2927/2958图通过；官方SCEG未提供自己的独立类别清单，仍有来源限制|
|episode sampler/标签顺序错误|Phase A完整seed42 sampler检查与eval support/query标签断言、保存2000 episode indices|weighted depletion、support先行、query16；逐episode数量/无重复及均值CI重算通过|
|augmentation不同|已保护datasets/transforms.py源码哈希；Phase A全部PNG像素重算对照；冻结config|官方训练crop84/jitter.4/flip.5，eval Resize92→crop84；workers4可改变随机增强流|
|optimizer/scheduler不同|config_resolved.yaml及800轮epoch_metrics的lr_used、checkpoint完整SGD/StepLR状态|SGD.1/.9/Nesterov/5e-4，StepLR400 gamma.1；没有调参|
|loss不同|已保护loss源码哈希、既有数值/梯度验证与冻结config|四项原loss固定lambda38/inter.05/intra.85，lambda未进入optimizer；未重新实现|

未解决：作者历史环境、不同GPU非确定性、单seed波动及官方精确数据副本身份。未执行参数搜索，也不将上述不确定性当作已证明原因。
