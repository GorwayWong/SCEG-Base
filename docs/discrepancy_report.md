# Discrepancy ledger

代码基线：`f7f130aeb5b1b49246150574c53a874d4ce8de80`。代码优先；未核验≠矛盾。本文也记录代码/README内部问题及工程适配。

|ID|类型与证据位置|真实行为/影响|复现决定|
|---|---|---|---|
|D01|代码 vs 用户全参数梯度要求；SC.py:52–62|三个 Transform.mlp 从未调用，共12个参数张量无梯度|保留模块与checkpoint键；严格全参数梯度检查如实失败，另验证活动参数|
|D02|条件分支；SC.py:109–116|Cars/Flowers 的 g_st 执行但输出被丢弃，6个参数张量无梯度，BN统计仍更新|保留计算与BN更新；不可提前跳过门控|
|D03|代码 vs 参数；SCEG.py:30/test.py|short_cut_weight 没有传入，实际固定0.3|配置固定0.3，拒绝baseline配置改变|
|D04|脚本内部；run.sh/test.sh/loss.py|训练脚本λ38，test路径λ36，parser36，loss默认32|以run.sh38为CUB训练基准；不要按测试文件名换λ|
|D05|代码 vs sampler参数；samplers.py:70|query_shot传入值被忽略，评测总是16|正式评测16；用户shape测试15单独记录|
|D06|未优化参数；loss.py:22/SCEG_train.py:23/trainer.py:84|λ requires_grad=True，但optimizer只含model，每轮重建loss|保留loss参数与梯度，禁止加入optimizer|
|D07|运行缺陷；loss.py:7/ResNet.py:31,53|默认CUDA设备硬编码；CPU不可运行|仅做输入device适配，公式/采样顺序不变，GPU parity验证|
|D08|README vs 仓库|requirements.txt不存在|提供实际验证环境的锁定依赖，不称作者原环境|
|D09|运行缺陷；SCEG_train.py:46–48|multi-GPU i=0时除零，入口还只暴露单GPU|本轮只支持单GPU，不声称复现multi-GPU|
|D10|运行缺陷；utils/util.py:110|resume引用未定义name，未恢复optimizer/RNG/DropBlock counter|后续提供审计完善的resume，明确工程适配，不改训练公式|
|D11|可复现性；ResNet.py:95|DropBlock counter在eval也递增；原checkpoint未存counter|保持计数语义；工程checkpoint额外记录counter|
|D12|数据文档不足；README/FRN README|未提供Dogs/Cars/Flowers class清单；FRN README不覆盖这些数据集|不猜split；真实类别清单必须溯源|
|D13|数学细节；utils/l2_norm.py vs SCEG.py:96|episode除sqrt(sum(z²)+1e-12)，global用F.normalize的max(norm,eps)|两种归一化分别实现，不能统一替换|
|D14|源码初始化；SCEG.py:66–71|重置所有Conv/Linear，覆盖backbone初始化，Conv bias保留默认|保持构造与初始化次序，以state_dict parity验证|
|D15|标签二值化边缘；loss.py:10|sklearn label_binarize二分类仅输出1列，广播导致不同于标准one-hot|原baseline way>=3；四图overfit用4个全局类，不能悄悄按2-way换公式|
|D16|实测环境限制；torch2.6.0 CUDA backward|AdaptiveMaxPool2d反向无强制确定性实现，严格deterministic=true报错|保留官方算子；显式warn_only=true，控制全部RNG，但不声称GPU逐bit重复；CPU可用严格模式|
|D17|工程记录修复；utils/experiment.py|首个debug运行的sys.argv记录缺失Python的-m，命令回放需模块形式|保留原command.txt并新增invocation_correction.md；后续使用sys.orig_argv并保存cwd，不改既有结果|
|D18|**数据来源冲突**；已有cub_split_original.py:33 vs FRN init_CUB_fewshot.py:77–87|已有副本按0-based索引取even即原始odd IDs为train；FRN用原始IDs1..200取even IDs为train，三组class身份不同|保留旧副本/所有旧实验；正式根改为data/frn/CUB_fewshot_raw，严格执行发布FRN规则，逐图片bytes不变；禁止把旧split称FRN|
|P01|论文§3.4式20 vs loss.py:28–41|已核验的softplus形式与exp/log求和形式数学等价|保留官方exp/log计算，不用数值稳定改写替代|
|P02|论文§3.3式13/14、§3.4式15–17、补充§7.2|SC加法聚合、max-pool、support先h_phi再求均值、query和global使用h_phi、Cars/Flowers禁用stage3→stage2均与代码一致|保留源码结构；0.3残差混合为源码确定的具体h_phi实现|
|P03|补充材料§7.1已核验|类别数CUB100/50/50、Dogs70/20/30、Cars130/17/49、Flowers51/26/25；未提供可核验完整class清单|类别数不代表类别身份相同；仍禁止猜split|
|P04|**论文与代码已确认冲突**；论文§3.4式20后的定义 vs loss.py:22/trainer.py:84|论文称温度alpha可学习；代码lambda有gradient但不在optimizer，每epoch重建，所以固定|严格保留固定lambda38，不加入optimizer|
|P05|发布配置不足；论文§4.1.2 vs parser/run.sh|论文Cars/Flowers intra=.65；parser默认.85，发布run.sh仅给CUB(.85)，没有Cars/Flowers专用脚本|不把CUB设置静默视为四数据集作者设置；扩展数据集配置需标注来源与缺口|

P04是已确认论文/代码冲突；P03的精确class身份仍有开放项。每次适配需追加来源、理由、验证，不删除异常实现。
