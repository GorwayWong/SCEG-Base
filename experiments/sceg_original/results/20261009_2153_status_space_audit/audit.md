# CUB训练状态与磁盘占用审计

检查时间：2026-10-09 21:53（Asia/Shanghai）。仅检查和记录；**没有恢复训练，没有删除、压缩或覆盖任何checkpoint/日志/配置/结果**。用户已明确要求暂不恢复训练。

## 1. 工作是否完成

**未完成，训练目前中断。** 原正式run为 `20261009_cub800_seed42_formal`；当前无baseline/workflow/evaluate训练评测进程。旧status.json仍写running，这是遗留状态，不能当作进程存活证明。

|完成标准|本次检查|
|---|---|
|CUB manifest、官方配置冻结|已完成；已有源码/历史产物hash未变化|
|dry run、独立resume|此前通过；本次保留全部记录|
|800epoch|仅完成122/800，未完成|
|checkpoint可加载|latest到epoch122可完整读取；当前best(epoch100)通过CPU strict load，未运行训练|
|5-way1/5-shot正式2000episode测试|未执行|
|50-way1/5-shot正式2000episode测试|未执行|
|最终results.json/result.json|未生成|
|discrepancy|已有记录；最终报告与结果表仍处于待完成状态|

当前训练记录为step3294、train loss9.1320559、train episode accuracy83.7037%；当前best validation为epoch100的91.155%。这是训练中验证结果，**不是正式测试结果**。

## 2. 中断与文件完整性

`latest.pth`为129,581,522 bytes，可加载完整model/optimizer/scheduler/RNG/DropBlock状态，completed_epochs=122、scheduler last_epoch=122、112个SGD momentum buffers。其SHA256为：

`898b5ac85fbb1bc55318a77cc0ad53dfef24f21bf8396f3a47a69d73e310937e`

该hash与metrics.json内第122轮历史记录完全一致，说明latest的字节与记录时相符。epoch121归档和当前best(epoch100)也可加载。

以下文件实际为0字节，不能作为有效恢复或审计产物：

- checkpoints/epoch_0122.pth
- rng_state_epoch_0122.pt
- epoch_0122.json
- activations_epoch_123.json

train.log和进程输出末尾还有NUL字节；旧日志最后可读内容为epoch121/120记录。metrics与latest共同证明完成122轮；epoch123的activation文件存在不等于完成123轮。git_diff.patch为空属于预期的无tracked diff记录，未算损坏。

本次检查时WSL uptime约199秒，说明系统近期启动；未发现旧进程或训练异常栈。文件状态与异常终止/重启后的部分写入丢失相符，但仅凭这些证据**不能断言最初中断由OOM、手动退出或何种停机操作导致**。本轮不修复、不恢复；有效latest122及best100应优先保护。

## 3. 空间为什么增长

本项目实际分配空间约**24.31 GiB**；其中数据目录约**5.31 GiB**，本阶段sceg_original目录约**18.61 GiB**。空间增量主要来自我在上一轮添加的**每epoch完整checkpoint归档策略**，比用户要求的last/best保留量更重。

每份完整checkpoint约129.6 MB（123.6 MiB），其中模型tensor约67.25 MB、SGD momentum约62.20 MB，另有配置、scheduler和RNG等小量状态。保存历史完整快照，会近似按epoch数量线性增长。

|正式run中的内容|数量|实际分配空间|
|---|---:|---:|
|逐epoch归档|122个文件，121个非空|14.603 GiB|
|best归档，包括首次验证前的provisional版本|24份|2.897 GiB|
|latest.pth / checkpoint.pth实体副本|2份|0.241 GiB|
|其余JSON、配置、日志、RNG等|剩余文件|约0.015 GiB|
|正式run合计|543个实体文件|17.756 GiB|

另外，dry run、resume检查和eval I/O检查约占0.855 GiB。sceg_original下共有155个.pth文件，合计18.585 GiB；它们解释了该目录绝大多数占用。checkpoint_last/best等别名是符号链接，统计未将链接目标重复计算。

**具体冗余：**前19轮还未进行正式验证，原trainer保存的是provisional last；记录适配器却同时把它归档成每轮best，额外保存19份完整payload，共2.293 GiB。这些并非19次验证选出的best。抽查epoch1、19、100的epoch/best对应文件，归一化ZIP成员大小与CRC元数据一致；不同容器路径可导致文件SHA不同，这不等于模型/optimizer状态不同。抽查不冒充全部归档的密码学等价证明。

因此，这次空间增长主要是记录层保留策略造成，不是训练超参数变化。若继续原策略，**仅800份epoch快照就约96.55 GiB**，还不包含best归档与其他运行副本。正式训练并不需要保留800份完整快照。

## 4. Windows / WSL空间

只读查询Windows WSL注册信息后，Ubuntu-22.04的ext4.vhdx文件长度为152,918,032,384 bytes，即**142.416 GiB**，属性为Archive。它覆盖整个Ubuntu发行版，包含其它项目与系统文件，不能把142.416 GiB全部归因于SCEG。

Linux文件系统当前可用约815.83 GiB。旁边的旧fewshot项目约38.47 GiB，本次未修改或清理。缺少本阶段之前的Windows VHDX文件大小快照，因此无法精确给出Windows端VHDX增长delta；本报告也没有执行任何VHDX压缩操作。

## 5. 本轮决定与后续约束

- 保持训练停止；不启动GPU训练或评测。
- 保留所有原始文件，包括损坏文件，便于审计；未自动清理历史checkpoint。
- 保持模型、loss、dataset、tests及冻结训练配置不变。
- 后续记录策略可考虑last/best加少量里程碑，同时保留全部轻量epoch metrics；best应仅在正式validation确实提高时归档。这里只指出可精简的记录策略，未实施。
- 暂不将历史epoch快照全部认定为可删除垃圾；它们仍有回滚价值。当前确定的额外provisional-best占用为2.293 GiB。

机器可读证据见同目录audit.json。本报告是本次观察，旧running状态和旧启动报告保留作为历史记录。
