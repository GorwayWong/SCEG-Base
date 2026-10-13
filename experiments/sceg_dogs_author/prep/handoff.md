# Author Dogs data preparation handoff

本轮终点为数据准备，不是论文复现完成。工作区 `/root/projects/aris/sceg_dogs_baseline`，分支 `experiment/sceg_dogs_author`，起点与当前Git HEAD为 `56627bf7184f200a946c7378df789fd3e28e842c`。保留未提交代码/迁移差异，没有commit、push、merge、回退或历史改写。实际生效指令来自用户任务、`/root/.codex/AGENTS.md` 与 `/root/.codex/RTK.md`。

## 四阶段实际结论

1. 旧任务：2026-10-13T15:26:27.670537+00:00 核对当前命令/cwd、专属参数、打开文件、screen及CUDA拥有者。旧PID/进程组/监听器已不存在，已确认无需停止，没有发送信号，没有恢复旧评测。没有旧任务继续写入；GPU旧CUDA占用已释放，桌面显示占用不作为失败。
2. 作者来源：两份原件同文件系统移至 `data/provenance/sceg_author/`，移动后SHA256与已知原件一致；未改变编码、换行或排序。来源kind为 `author_via_email_user_supplied`；邮件日期和发布许可均为unknown。Dogs完整70/20/30类互斥、合并120目录名；Cars原始1-based IDs完整130/17/49类，互斥覆盖1..196。作者只给类别清单，没有逐图片清单或完整训练配方。唯一活跃原件和精简source.json见 [来源目录](../../../data/provenance/sceg_author/)。
3. 旧证据：唯一档案 `experiments/_archive/sceg_dogs_dn4/dogs800_seed42_dn4/`。同文件系统rename保留原始文件字节和inode，不复制大权重。磁盘实际为800轮、best420，三组各2000 episodes；30-way5-shot未落盘、继续缺失/null，没有补跑。原始配置/命令/环境、训练与验证记录、测试准确率/索引/路径表、历史报告、来源与执行patch都保留。旧协议标注superseded_by_author_split，但其DN4协议测量没有被改称无效或作者结果。
4. 新数据与清理：作者split是唯一类别归属依据。新根 `/root/projects/aris/sceg_dogs_baseline/data/dogs_author/StanfordDogs_fewshot`。全量20,580原图字节保持一致；train/val/test图片数重新统计为 **11877/3325/5378**，不沿用DN4数量。无软链接。复用20,580原图和4,453等价PNG的硬链接，新增4,250张PNG；旧目录删除后新路径仍完整可读，硬链接没有路径依赖。25个从旧训练侧转入新评测侧的类别均补齐pre图。

## 数据记录与必要验证

精确类别、每类/每组计数和实际ImageFolder class_to_idx在 [split.json](split.json)；唯一完整样本清单为 [samples.jsonl](samples.jsonl)，其他记录只引用它。来源、配置与清单哈希、复用统计及验证在 [prepared.json](prepared.json)。原始 `images.tar` 来源与已有hash直接引用 `/root/projects/aris/sceg_fsfg/data/provenance/raw_archive_audit.json`，未下载、未重复全量hash原archive、未另建中转解压目录。

预处理核对固定 [SCEG utils/util.py](https://github.com/Legenddddd/SCEG/blob/f7f130aeb5b1b49246150574c53a874d4ce8de80/utils/util.py#L19-L47) 和本地活动 `datasets/transforms.py`：type=0为RGB、短边Resize92、CenterCrop84、PNG；pre加载只做ToTensor/Normalize，不重复resize/crop。不裁剪狗框，不重新JPEG编码。本轮没有更改模型/训练参数或依赖版本。

全量路径/类别/图片覆盖与ImageFolder标签核对通过；实际数据组件在CPU读取3个普通小批次，输入3×84×84，有限且可读。27张真实图片的在线原图与保存PNG输入逐元素一致，覆盖全部25个新转入评测的类别；删除旧供体目录后复查再次通过。没有加载训练模型，没有前向/反向、训练、模型评测、调参或多seed。数据准备测试3项及既有代表性采样回归2项通过，共 **5 passed / 1.12s**；未重跑旧合成训练/恢复或CUB全套检查。

## 实际执行命令

以下命令的cwd为本工作区；reuse-root仅在删除旧供体之前使用过，新脚本独立从neutral来源和原tar启动，不依赖旧实验/prepared.json/blocked报告。

```bash
rtk git switch -c experiment/sceg_dogs_author
rtk proxy env CUDA_VISIBLE_DEVICES= /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_dogs_author.prep.closeout archive
rtk proxy env CUDA_VISIBLE_DEVICES= /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_dogs_author.prepare --reuse-root /root/projects/aris/sceg_dogs_baseline/data/dogs_dn4/StanfordDogs_fewshot
rtk proxy env CUDA_VISIBLE_DEVICES= /root/projects/aris/sceg_fsfg/.venv/bin/python -m pytest -q tests/test_dogs_author_data.py tests/test_episode_sampling.py::test_training_nonoverlap_support_order tests/test_episode_sampling.py::test_evaluation_ignores_query_and_uses_disjoint_samples
rtk proxy env CUDA_VISIBLE_DEVICES= /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_dogs_author.prep.closeout clean
rtk proxy env CUDA_VISIBLE_DEVICES= /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_dogs_author.prep.finalize
```

## 归档、删除与空间

- best：`/root/projects/aris/sceg_dogs_baseline/experiments/_archive/sceg_dogs_dn4/dogs800_seed42_dn4/run/checkpoint.pth`，129430130 bytes，SHA256 `2428ab008377ea8f9559b3ef26c0cd271475d38907a10e93588c340d42180c9d`。
- latest：`/root/projects/aris/sceg_dogs_baseline/experiments/_archive/sceg_dogs_dn4/dogs800_seed42_dn4/run/latest.pth`，129428754 bytes，SHA256 `bb36df54ccedc803d287ae2e307bc09c7da838038a7e8baba3a9a55c9c26833d`。
- 两份权重各一份；单次归档校验后rename保全，不加载模型证明归档。权重被Git忽略，因此档案是本地实物，不能依赖Git重建它们。
- [cleanup_manifest.json](cleanup_manifest.json) 逐项记录keep/archive/delete、原因、旧→新路径和分配块计数，解释历史报告中的旧路径；源配置/原始结果/历史报告没有回写。
- 已清理旧DN4派生图片目录、800份逐轮activation、过时DN4入口/仅对应它的测试、重复family结果/screen日志、已停用的旧唤醒控制文件。当前活跃索引是 [experiments/INDEX.md](../../INDEX.md)。
- 旧 `sceg_dogs_baseline` 合成/blocked证据及被保留测试引用的共享辅助入口没有贸然删除；这是单列保留项，不是新数据准备依赖或活跃split指引。没有扩展全仓库删除，没有git clean/reset。
- 删除最后链接释放分配块 **71213056 bytes / 67.91 MiB**。移动释放0；仍被新数据硬链接引用的内容释放0。新增PNG分配 66965504 bytes、单一样本清单分配 5574656 bytes；因此该删除释放量不冒称整个任务净容量收益。
- CUB数据/权重/结果、ProKeR v1代码/锁定参数/缓存/结果、原始数据包与共享环境的前后元数据/版本摘要一致，详见同一manifest。没有复制CUB权重或环境、升级依赖、Cars图片准备或映射。

## 代码追溯与下一轮边界

已有提交加迁移manifest解释旧文件；本轮未提交新准备/收尾代码和测试保存一份 [source.patch](source.patch)，SHA256 `8ad3ab6d6c25fbc2b95fbfcfcf4ec8eae0505a531e4c4491c17e31f2ffacca96`。补丁仅覆盖实际新源码及ignore差异，不含大图片/权重，不是不能覆盖未跟踪文件的空git diff。源文件哈希在同一cleanup_manifest中。

训练入口适配、配方最终核对和模型级检查尚属下一轮；本轮没有实现或提供未经验证的一键训练命令。作者与旧DN4有62类归属不同，旧训练含14个作者测试类，因此 **旧权重禁止用于作者协议初始化、续训或正式评测；后续必须随机初始化**。

```yaml
stage: author_data_preparation
status: completed
old_task_stopped: true
old_evidence_archived: true
author_sources_preserved: true
dogs_data_ready: true
cars_class_list_checked: true
cars_image_mapping_checked: false
training_started: false
accuracy: null
```

本轮没有必要阻塞；未确认无引用的历史辅助项已保留并说明。到数据准备为止，不继续旧训练/评测任务。
