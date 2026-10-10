# SCEG reproduction — engineering record

CUB 正式训练已完成800 epochs，三组正式测试完成，50-way 5-shot按用户要求暂停。当前实验状态、正式结果和协议见 [CUB reproduction report](cub_reproduction_report.md)、[CUB result table](cub_result_table.md) 与 [CUB discrepancy](cub_discrepancy.md)。

以下保留正式训练前Phase4的工程验证记录，其中“未开始”“待确认”等表述记录当时状态，不代表当前CUB实验状态。

## 1. 环境

Python3.10.20，PyTorch2.6.0+cu124，torchvision0.21.0+cu124，RTX4060Ti。requirements锁定实测版本。项目.venv以system-site-packages复用已有torch环境，只在本项目.venv安装scikit-learn以运行官方loss对照；没有修改原fewshot工程或全局Git身份。作者README仅给torch>=1.7，没有完整依赖锁定；不宣称这是作者历史环境。

全部随机源可设seed。CUDA AdaptiveMaxPool2d的反向在此版本无强制确定性实现，保留算子并显式warn_only；GPU只承诺按容差检查，不能承诺逐bit相同。

## 2. 数据

已有CUB和Cars物理副本及全文件hash/PIL verify通过。进一步发现已有CUB class身份与真实FRN不同；旧副本完整保留，正式根另严格按FRN已发布的1-based class ID规则准备，5903/2927/2958图、100/50/50类、图像bytes全部不变，详见data_pipeline.md。Dogs/Flowers原图和标签已下载，archive图数/类别数检查通过，精确few-shot split仍未提供，不自行猜测。Cars镜像与作者原始像素等价性仍未核验。

## 3. 模型

ResNet12、SC、feature transform、cosine classifier、Intra-Inter loss独立实现。完全保留官方未调用MLP与Cars/Flowers丢弃门控，226个state_dict键可直接加载官方权重。两个SC分支eval四组cosine输出对照差值均0。张量流程见tensor_flow.md。

## 4. 训练配置

基准configs/sceg_original.yaml来自官方CUB run.sh：10-way5-shot15-query，SGD lr.1/momentum.9/Nesterov/weight_decay5e-4，StepLR400、gamma.1，总800epochs，λ38/inter.05/intra.85，seed42，官方data workers48。完整trainer已增量实现但未进入完整训练执行阶段；实际设备与所有配置将保存于独立运行目录。debug独立配置，200steps/no aug/no dropout/no weight decay。

完整训练入口主动拒绝未接受的全参数梯度例外；损失lambda不进optimizer。checkpoint同时存optimizer/scheduler/RNG/DropBlock计数，并另存latest.pth以支持epoch边界resume。已通过隔离synthetic unit test验证scheduler在validation后step及最终选择最佳val权重，此unit test不是模型性能结果。

## 5. 结果与关卡

|关卡|证据|结果|
|---|---|---|
|仓库调查|repository_analysis.md/固定commit|完成|
|tensor skeleton|80张随机输入，完整shape|通过|
|shape/range|3个pytest|通过|
|活动gradient|CUB112、Cars106个张量；缺失集合精确对照|通过|
|严格所有参数gradient|官方闲置参数CUB12、Cars18|失败，不能勾选通过|
|loss数学|标量公式/gradcheck/binary edge|3个测试通过|
|真实4图overfit|FRN训练集，debug_overfit_report.md/独立日志和checkpoint|loss.0194131、accuracy100%，通过|
|full training/正式few-shot/many-way|未产生完整训练和测试产物|未完成|

四图overfit是记忆调试，不是论文数据集accuracy。正式metrics中的未测accuracy/CI保持null。严格gradient例外正在等待用户选择；未收到选择前不进入完整训练。

最新模型验证：**12 passed, 2 xfailed**；2个xfail就是严格全参数梯度的已确认失败。CUB真实test_pre的5-way1-shot/5-shot各1个random-weight smoke episode通过，batch分别85/105；仅检查I/O/标签顺序/finite stats，不能作为正式结果。正式FRN CUB与Cars共11,326个预缩放PNG通过source Resize92→CenterCrop84的像素逐图对照。旧CUB预缩放缓存已清理，其历史哈希清单和原始JPEG保留；未修改模型或实验指标。

200张10-way5-shot15-query随机输入的完整forward/backward预检通过；GPU峰值allocated约9.61GiB，reserved约12.07GiB（16GiB4060Ti），所有112个活动参数梯度finite、官方12项缺失一致。预检optimizer updates=0，accuracy/CI=null，不等于完整训练，untrained checkpoint位于独立sceg_preflight实验，不能充作baseline。

已提交代码后的预检运行：[full_batch_200_committed metrics](../experiments/sceg_preflight/runs/full_batch_200_committed/metrics.json)，源码commit `9b39c67b8d9435b53f60e24db610fa3c17cd2a18`。FRN overfit运行源码commit为`40f2b978bb8508f44cf3ba29d74a9fc1b3f013e8`；所有必需运行文件与准确模块命令已核验。

下一步：收到对D01/D02官方闲置参数例外的明确选择后，才执行已准备的CUB800epoch命令；随后做5-way与50-way的1/5-shot2000episode正式评测。Dogs/Flowers需要可核验的作者class清单，当前仅有原始图像和原始标签，不按类别数猜测划分。未启动完整训练、尚无正式论文复现accuracy。

首个debug运行的command.txt未记录-m，保留原文件并附invocation_correction.md；后续使用sys.orig_argv和cwd.txt记录准确命令。checkpoint分析与重载已通过。

清理与发布审计见[cleanup_report.md](cleanup_report.md)。Git保存所有轻量实验记录；数据和大型checkpoint保留本地，checkpoint_manifest.json提供大小与SHA256。初次预检的重复未训练checkpoint已按完全相同SHA256去重，正式预检的同内容文件仍保留。
