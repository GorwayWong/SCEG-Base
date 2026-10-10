# Reproduction checklist

- [x] 固定官方commit并调查入口/模块/数据流
- [x] 建立工程设计及差异台账
- [x] tensor skeleton及官方权重行为对齐（两个SC分支，各226 keys，eval max error=0）
- [x] forward通过（5-way,1-shot,15-query，有限且范围正确；3 tests passed）
- [ ] gradient严格全参数通过（已发现官方未调用参数，待实测）
- [x] 所有活动参数梯度通过、缺失参数精确对齐官方（CUB112/12，Cars106/18）
- [x] loss数学验证通过（标量公式、gradcheck、binary edge）
- [x] 真实FRN训练集4图片、200 steps overfit通过（loss0.0194131，episode/global/train/eval均100%）
- [x] 200-image完整episode前向/反向显存预检通过（0 optimizer updates）
- [x] Dogs/Flowers原始图片和标签下载、archive结构/数量检查通过
- [ ] 四数据集下载/物理副本/划分来源通过
- [x] CUB full training完成（800 epochs，best epoch480；其他数据集未完成）
- [x] CUB 5-way 1-shot/5-shot与50-way 1-shot正式评测完成（每项2000 episodes）
- [x] CUB 50-way 5-shot正式评测完成（2000 episodes）
- [x] 完整四组结果与论文对照（见cub_result_table.md）
