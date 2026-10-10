# 轻量记录策略与清理结果

按用户最新指示已完成修改和清理；**训练继续暂停在122/800 epochs，本轮没有执行训练或正式评测**。

## 清理结果

|项目|清理前|清理后|
|---|---:|---:|
|本项目实际分配空间|24.307 GiB|5.961 GiB|
|本阶段baseline实体完整checkpoint|155个，含一个损坏空文件|2个|
|已回收Linux文件分配空间|—|**18.346 GiB**|

共删除283个本阶段冗余权重、逐轮RNG文件及其别名。包含全部逐epoch快照、历史best、前19轮临时best副本，以及dry/resume/eval I/O检查中不再需要的权重副本。逐轮JSON、JSONL、validation/evaluation原始记录、日志、配置、命令、环境、source/data manifests和旧源码快照保留；完整删除清单见cleanup_manifest.json。

保留的正式权重：

- latest.pth：epoch122，129,581,522 bytes，SHA256 `898b5ac85fbb1bc55318a77cc0ad53dfef24f21bf8396f3a47a69d73e310937e`。
- checkpoint.pth：当前best epoch100，129,582,834 bytes，SHA256 `221fccca971f7dc9bf852078abd47ff8f9e541d9704728ca9a084550ff670ab2`。

二者删除前后hash完全一致。checkpoint_last.pth与checkpoint_best.pth分别指向上述文件，不复制权重。已将原来指向损坏空归档的last别名修正为有效latest；单个rng_state.pt从latest内完整RNG恢复，不进行训练。

原有sceg_debug/旧sceg_preflight、数据和models/losses/datasets/tests没有改动；Phase A protected_source_manifest与existing_artifact_hashes逐项检查均无变化。历史指标中的旧checkpoint hash作为当时证据保留；相应历史权重已按本次指示清理，不伪称仍可逐轮加载。

## 后续记录方式

- 完整状态只保留rolling last和正式validation选出的best，约0.24 GiB，数量不随epoch增长。
- 第一次validation之前只存last，不额外保存临时best。
- RNG只保留当前一个sidecar，完整RNG同时包含在last中。
- 每轮保留轻量loss/accuracy/LR/validation CI/GPU/time指标和日志；配置、环境、命令、git/source/data指纹及评测原始episode结果保留，支持论文数字核查和协议追溯。
- 评测引用训练best，不另外复制模型文件。
- 删除原来为全epoch归档设置的150 GiB硬性空间预留。
- 取消每epoch重复复制、重新加载和hash完整checkpoint；只在恢复或最终结果需要时处理完整权重。

修改仅在experiments/sceg_original的记录适配器及报告中；原trainer、optimizer、scheduler、loss、sampler与增强代码不变。

## 验证

在临时目录用小型CPU tensor payload检查记录行为：多轮后只有last/best两个实体权重，best不被非improvement覆盖，当前RNG只有一个文件，全部轻量epoch指标保留；resume best别名及eval引用通过。该检查不包含模型训练或正式accuracy。实际保留权重hash和别名验证通过，当前无训练评测进程。机器可读验证见verification.json。

## Windows空间说明

本次确认释放的是**Linux文件系统空间**。Windows侧Ubuntu的VHDX文件长度检查后仍约**142.48 GiB**；本轮未执行WSL关闭或VHDX压缩，因此不宣称Windows虚拟磁盘已经缩小。
