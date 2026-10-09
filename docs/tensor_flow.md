# Phase 1 — Tensor flow

随机结构测试：way=5、shot=1、query=15，B=80；正式测试query=16另记录。无训练和数据依赖。

|模块|输入shape|输出shape|变化|
|---|---|---|---|
|ResNet12 stage1|[80,3,84,84]|[80,64,42,42]|3个3×3卷积、residual、maxpool2|
|Backbone stage2|[80,64,42,42]|[80,160,21,21]|同上|
|Backbone stage3|[80,160,21,21]|[80,320,10,10]|同上；训练DropBlock|
|Backbone stage4|[80,320,10,10]|[80,640,5,5]|同上；训练DropBlock|
|SC协同门控|三阶段特征图|[80,1,21,21]及[80,1,10,10]|bilinear resize、640/320→16→1、sigmoid|
|SC增强|三阶段特征图+门控|输入同shape|门控广播相乘，加自增强；CUB/Dogs额外stage3→2|
|SC空间Transform|三阶段增强图|3×[80,640]|1×1 Conv→BN→ReLU→AdaptiveMaxPool→flatten|
|Feature transform|3×[80,640]|3×[80,640]|0.3z+0.7ELU(Linear(BN(z)))|
|Support prototype|[5,1,640]|[5,640]|support类内均值，然后归一化|
|Query|每阶段后75个embedding|[75,640]|独立episode归一化|
|Cosine classifier|[75,640]及[5,640]|3×[75,5]|einsum qd,wd→qw；范围[-1,1]|
|Global classifier|[80,640]及[100,640]|[80,100]|F.normalize后einsum bd,kd→bk；范围[-1,1]|
|Test aggregation|3×[75,5]|score[75,5]、prediction[75]|sum，范围[-3,3]；argmax|

官方forward输出顺序stage3/stage4/stage2/global。保留未用MLP与所有checkpoint键。debug/eval允许按小batch编码再拼接，只有eval下BN已固定才等价；正式训练不拆分BN batch。

运行：`python -m tools.tensor_skeleton --device cpu`。

2026-10-09 实测：RTX4060Ti、torch2.6.0+cu124，随机80张输入的embedding/cosine均与上表一致、全部finite。`pytest -q tests/test_forward_shape.py`：3 passed。

`python -m tools.verify_official --reference /tmp/sceg-official-reference`：CUB和Cars两个SC分支各226个state_dict键完全兼容；加载同一官方随机初始化权重后，4组eval输出最大绝对差均为0。此项是组件行为验证，不是训练后的论文指标。
