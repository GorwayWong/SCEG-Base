# Phase 3 — Gradient flow audit

2026-10-09，RTX4060Ti，PyTorch2.6.0+cu124。使用真实训练目标：3个episode Intra-Inter + 1个全局 Intra-Inter，不加伪造的零权重项、不删除/冻结参数。

|分支|活动参数张量|无梯度参数张量|严格全参数关卡|活动参数验证|
|---|---:|---:|---|---|
|CUB/Dogs|112|12|失败|通过|
|Cars/Flowers|106|18|失败|通过|

12项来自 `SC.t_{t,s,l}.mlp.{0,1}.{weight,bias}`，官方未调用。Cars/Flowers另有 `SC.g_st.{0,1,3}.{weight,bias}` 6项，其门控输出被丢弃但BN统计仍更新。不是 detach、no_grad 或 frozen 引起。

官方对照：完全相同的缺失集合；相同输入、权重、随机种子下训练输出通过atol/rtol=1e-5对照，活动梯度最大绝对差CUB=5.340576171875e-05、Cars=6.246566772460938e-05。此处只报告梯度差，尚未声称GPU逐bit相同。

命令：`.venv/bin/python -m pytest -q tests/test_gradient_flow.py tests/test_loss.py`，结果 **5 passed, 2 xfailed**。2个xfail只在backward完成且缺失集合精确相符后标记，其他异常不会被吞掉。loss通过独立标量公式、float64 gradcheck及二分类广播边缘验证。

CUDA强制确定性首次backward失败：AdaptiveMaxPool2d反向不支持torch严格确定性。保留作者算子，显式 `deterministic_warn_only=true`；所有Python/NumPy/torch/CUDA随机源设seed。该环境的GPU结果只能按容差复核，不能承诺逐bit复现；CPU可以严格确定性。

下一步：真实4图、200 steps debug overfit，关闭augmentation/dropout/weight decay；该debug验证不覆盖或消除严格全参数关卡的失败。
