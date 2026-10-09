# Step 2 — 工程结构与隔离

```text
configs/                       # 作者配置与debug配置；每次运行保存解析后副本
datasets/{cub,cars,dogs,flowers}.py
datasets/{folder,samplers,transforms}.py
models/backbone/resnet12.py
models/modules/{sc,transform_fc,cosine_classifier}.py
models/sceg.py                  # 仅组装，不加入改进方法开关
losses/intra_inter_loss.py
trainers/trainer.py             # overfit通过后增量实现；完整执行仍受关卡限制
evaluators/fewshot_eval.py
tools/{tensor_skeleton,train,test,analyze_checkpoint}.py
utils/{config,reproducibility,experiment,diagnostics}.py
tests/                         # shape → gradient → loss/equivalence
experiments/sceg_original/{config.yaml,README.md,result.json,runs/}
experiments/sceg_debug/{config.yaml,README.md,result.json,runs/}
experiments/future_methods/     # 新方法的独立入口、配置和实现
data/                          # 物理副本/下载文件；不提交图片
data/provenance/                # 哈希、split来源与数据清单
docs/                          # 调查、张量流、阶段验证与最终报告
```

每次运行独占新 run_id，用目录创建的排他性拒绝覆盖。目录中保存 `config.yaml,git_commit.txt,git_diff.patch,command.txt,environment.json,data_manifest.json,train.log,metrics.json,checkpoint.pth`。checkpoint另存 optimizer/scheduler/RNG/DropBlock counters，避免原实现resume缺陷。若还没有Git commit，记录 UNCOMMITTED 和源码文件hash，不捏造commit。

baseline配置由run.sh提取；debug配置允许关闭随机增强/dropout/weight decay，但所有差异明示。论文指标只来自完整训练及正式测试，随机输入、overfit、未运行实验accuracy为null。

模块docstring记录论文段落、官方行号、数学式、input/output shape；forward检查square images、support-first样本数、通道等约束。关键张量由可选trace/activation hooks保存统计，不在每次forward打印。cosine内积优先einsum，等价性对照官方F.linear。

梯度关卡同时输出严格全参数结果和活动参数结果；官方未用MLP不能删除/冻结来伪造全参数通过。若用户要求保持严格全参数关卡，该关卡失败时不能称全流程通过。
