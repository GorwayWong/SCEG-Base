# CUB baseline discrepancy ledger

事实顺序：官方固定 commit > 论文正文 > README。官方源码为 [Legenddddd/SCEG](https://github.com/Legenddddd/SCEG/tree/f7f130aeb5b1b49246150574c53a874d4ce8de80)，本工程起点为 `900e06e2414a9873d2c9933eda8c417d3834a3c7`。本阶段只新增实验配置、执行记录及文档，不修改已验证实现。

|项目|论文 / 用户协议|官方真实行为|本次决定与影响|
|---|---|---|---|
|温度参数|论文描述可学习温度|lambda 有梯度，但未进入 optimizer，每 epoch 重建 loss|保留固定 lambda=38，inter=.05、intra=.85；不调参。详见既有 discrepancy P04|
|训练轮数|800 epochs|run.sh 的 epoch=400、stage=2，总轮数800；StepLR400、gamma=.1|总800；epoch 是按剩余图片数加权抽类并耗尽图片，episode数量动态|
|验证频率|用户要求每 epoch val accuracy / CI|每20 epochs，以5-way5-shot、1000 episodes验证；严格提高才更新 best|保持每20轮；其他epoch的val accuracy/CI为null，不能编造或额外验证改变DropBlock计数/RNG|
|评测query|接口包含query_shot|sampler忽略传值，固定16|所有正式5/50-way评测query16；各shot2000 episodes|
|95% CI|均值与95% CI|1.96 × sqrt(var(ddof=0)/episodes)|保存官方正态区间半宽及统计定义|
|CUB类别身份|论文补充只给100/50/50类；README参考FRN|FRN固定commit使用原始IDs1..200，偶数train、mod4=1 val、mod4=3 test|使用data/frn/CUB_fewshot_raw，5903/2927/2958图；保留旧CloserLook划分。依据[固定FRN脚本](https://github.com/Tsingularity/FRN/blob/64e43f3ca3a4953445788fc853e288e032d0083c/data/init_CUB_fewshot.py)|
|数据worker|官方run.sh为48|本机4核、可用RAM约6.4GiB|用户明确授权workers=4；只改执行配置，不改变sampler/augment/优化参数。worker数可改变随机增强流，不声称与48 workers逐bit一致|
|闲置参数|严格全参数gradient检查存在例外|CUB三个未调用MLP合计12个参数张量无梯度|本次用户要求保持官方行为，接受既有例外；保留全部state_dict键和112个活动参数检查|
|checkpoint / resume|官方resume接口存在|官方存裸state_dict，resume引用未定义name且未恢复optimizer/RNG|使用已有工程trainer的完整状态：model、SGD、StepLR、epoch、RNG、DropBlock计数；记录格式差异，不重新实现训练|
|确定性|seed42|当前PyTorch的AdaptiveMaxPool2d CUDA backward不支持强制确定性|沿用已确认warn_only=true、TF32=false、AMP=false；不承诺CUDA逐bit复现|
|记录与入口|用户要求checkpoint_last/best、每轮GPU/时间等|已有入口记录latest.pth/checkpoint.pth，缺少部分独立记录；publish_result会更新旧family result|只在实验执行入口中补充记录、停止检查和产物命名；正式结果按run保存；用户要求清理后删除未运行的family占位文件|

论文目标来自 [CVF论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhao_From_Few-way_to_Many-way_Rethinking_Few-shot_Fine-grained_Image_Classification_CVPR_2026_paper.pdf)：CUB 5-way 1/5-shot为87.79/94.56，50-way 1/5-shot为56.02/73.69。这些是论文值。三组已完成的正式指标见cub_result_table.md；50-way 5-shot暂停，accuracy/CI为null。

Phase A首轮发现固定FRN源码文件没有随旧manifest存储，已保留失败记录，并在新run重新从固定URL验证SHA256；这不代表数据split检查失败。

2026-10-09记录策略修正：按用户指示取消逐epoch完整checkpoint/RNG及临时best归档，只保留rolling last与真实validation-selected best，评测引用同一best而不复制权重；所有轻量配置、环境、命令、source/data hash清单、逐epoch/validation指标和评测原始记录保留。此次清理仅涉及本阶段新增的冗余权重，旧overfit/preflight验证产物不动；模型、优化和采样协议不变，当时训练暂停。

2026-10-10：恢复第122轮状态并完成800轮训练。三组测试完成，50-way 5-shot按用户指令停止，等待自动发布的进程也已取消。原评测入口只在完整setting结束时保存结果，不支持episode级续跑；没有把未保存的部分episode当作正式结果。完整CUB baseline完成标准尚未满足。

实验结束后的存储清理：移除旧中断权重、重复单轮JSON与激活诊断，保留完整800轮历史、全部40次validation指标、best epoch原始验证记录和三组原始测试episode结果；其余39次验证采样明细清理。11788张相同JPEG在项目内部硬链接去重，两套类目录及图像SHA256不变，未链接到外部来源工程。准备时的copy2/独立副本证明仍作为历史记录保留；当前存储变化在cleanup_result.json单独记录。记录入口今后仅验证已保护源码，不再锁定过时输出；运行时原始脚本副本保留。模型/loss/dataset源码与优化参数均未修改。

补测完成：只计算50-way 5-shot 2000 episodes，使用同一epoch480 best，未重新训练或重复前三组推理。原tools.test只在[1,5]循环前设一次seed；本次用原batch sampler重放1-shot采样，并逐集核对已有2000组indices，保留NumPy随机状态续接。没有加载这些图片或执行1-shot模型forward；eval变换确定，DropBlock关闭，省略的forward计数不影响预测。核心评测代码保持原样；进度记录仅包装loader迭代。历史暂停与清理记录保留，当前完整结果见新completed suite。
