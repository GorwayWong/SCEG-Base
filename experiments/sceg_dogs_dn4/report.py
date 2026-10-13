"""Recompute published metrics from raw episodes and render the current run report."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from tools.prepare_data import sha256_file
from utils.experiment import write_json
from .prepare import BASE


def protected_state():
    """Lightweight file metadata over existing experiments/data; no content rehash."""
    roots = [Path("/root/projects/aris/sceg_fsfg/experiments/sceg_original"),
             Path("/root/projects/aris/sceg_fsfg/data"),
             Path("/root/projects/aris/sceg_dogs_baseline/experiments/sceg_original"),
             Path("/root/projects/aris/sceg_proker/experiments/sceg_proker"),
             BASE.parent/"sceg_dogs_baseline"]
    result = {}
    for root in roots:
        digest, count = hashlib.sha256(), 0
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".py":
                stat = path.stat()
                digest.update(json.dumps([str(path.relative_to(root)), stat.st_size,
                                          stat.st_mtime_ns, stat.st_ino]).encode())
                count += 1
        result[str(root)] = {"files": count, "metadata_sha256": digest.hexdigest()}
    return result


def generate_report(run_id):
    run = BASE/"runs"/run_id
    metrics = json.loads((run/"metrics.json").read_text())
    config = yaml.safe_load((run/"config.yaml").read_text())
    split = json.loads((BASE/"data_protocol/split.json").read_text())
    comparison_path = BASE/"data_protocol/author_split_comparison.json"
    comparison = json.loads(comparison_path.read_text()) if comparison_path.exists() else None
    split_match = comparison["dogs"]["sceg_exact_split_match"] if comparison else "unconfirmed"
    split_note = "SCEG作者精确split及部分专项配置未确认。"
    if comparison:
        dogs = comparison["dogs"]
        split_note = (f"用户提供作者原始类别清单后，逐类别核验确认当前DN4划分与作者划分{('不一致' if split_match == 'mismatch' else '一致')}；"
                      f"{dogs['changed_class_count']}/120个breed归属不同。完整差异和作者文件SHA256见 "
                      "[author_split_comparison.json](data_protocol/author_split_comparison.json)。部分Dogs专项训练配置仍未确认。")
    rows, results = [], []
    checkpoint_hash = sha256_file(run/"checkpoint.pth") if (run/"checkpoint.pth").exists() else None
    for way in (5, 30):
        for shot in (1, 5):
            path = run/f"test_{way}way_{shot}shot.json"
            result = {"way": way, "shot": shot, "accuracy": None, "confidence_interval": None,
                      "episodes": 0, "checkpoint_sha256": None, "raw_record": None}
            if path.exists():
                raw = json.loads(path.read_text())
                values = np.asarray(raw["episode_accuracies"], dtype=float)
                assert len(values) == raw["episodes"] == 2000
                assert len(raw["episode_indices"]) == 2000 and raw["sample_index_paths"]
                assert raw["checkpoint_sha256"] == checkpoint_hash
                for indices in raw["episode_indices"]:
                    assert len(indices) == way*(shot+16) and len(set(indices)) == len(indices)
                    assert min(indices) >= 0 and max(indices) < len(raw["sample_index_paths"])
                mean = float(values.mean())
                ci = float(1.96*np.sqrt(values.var(ddof=0)/len(values)))
                assert np.isclose(mean, raw["accuracy"]) and np.isclose(ci, raw["confidence_interval"])
                result.update(accuracy=mean, confidence_interval=ci, episodes=len(values),
                              checkpoint_sha256=checkpoint_hash, raw_record=str(path.relative_to(BASE)))
            results.append(result)
            value = "null" if result["accuracy"] is None else f"{result['accuracy']:.4f} ± {result['confidence_interval']:.4f}"
            rows.append(f"| {way}-way {shot}-shot | {value} | {result['episodes']} / 2000 |")
    protection_path = BASE/"preservation_check.json"
    preservation = json.loads(protection_path.read_text()) if protection_path.exists() else {}
    if "before" in preservation:
        preservation["after"] = protected_state()
        preservation["unchanged"] = preservation["before"] == preservation["after"]
        write_json(protection_path, preservation)
    status = metrics["status"]
    if status == "completed" and not metrics.get("full_evaluation_completed", False):
        status = "running"
    summary = {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(),
               "run": str(run), "protocol_id": split["protocol_id"],
               "source_verified": split["source_verified"], "sceg_exact_split_match": split_match,
               "author_split_comparison": str(comparison_path.relative_to(BASE)) if comparison else None,
               "completed_epochs": metrics.get("completed_epochs", 0), "required_epochs": 800,
               "full_training_completed": metrics.get("full_training_completed", False),
               "full_evaluation_completed": metrics.get("full_evaluation_completed", False),
               "best_epoch": metrics.get("best_epoch"), "best_validation": metrics.get("best_validation"),
               "checkpoint_sha256": checkpoint_hash, "results": results,
               "protected_artifacts_unchanged": preservation.get("unchanged")}
    write_json(BASE/"results.json", summary)
    command = (run/"command.txt").read_text().strip()
    restore = f"rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -u -m experiments.sceg_dogs_dn4.run --run-id {run_id} --resume {run/'latest.pth'}"
    tests = json.loads((BASE/"technical_checks.json").read_text())
    pause_note = ""
    if metrics["status"] == "paused":
        pause_note = (f"用户于 {metrics.get('paused_at_utc')} 要求暂停。进程组15905已SIGSTOP挂起，最后一组30-way5-shot评测的内存现场保留，显存仍占用；自动收尾监听器已停用。"
                      "本次提交/推送保存代码和现有证据，不代表四组评测全部完成，也不恢复任务。")
    text = f"""# Original SCEG on the publicly released DN4 Stanford Dogs split

状态 **{summary['status']}**；记录时间 {summary['updated_utc']}。已完成 {summary['completed_epochs']}/800 epochs；正式训练完整={summary['full_training_completed']}，四组正式评测完整={summary['full_evaluation_completed']}。只有 seed42。未执行指标为 null，启动不等于完成。

{pause_note}

## 数据与配置来源

DN4@{split['source']['revision']} 的三个发布 CSV 只用于划分，未采用 DN4 的模型或训练配方。原始 CSV、SHA256、精确 breed 清单、archive 双向差集见 [split.json](data_protocol/split.json)。实际核验 70/20/30 类，两两互斥、合并120类；图片 11962/3478/5140，总20580，每个(label,filename)唯一，CSV/archive 差集为空。`protocol_id: dn4_dogs_2019`、`source_verified: true`、当前对照结论 `sceg_exact_split_match: {split_match}`。verified 仅指DN4公开来源和映射已核验。冻结split.json/config中的unconfirmed保留为训练启动时的知识状态，不改写其内容或hash；最新作者对照记录独立保存。

{split_note}

复用原 archive 及 [已有来源审计](/root/projects/aris/sceg_fsfg/data/provenance/raw_archive_audit.json)，未重复下载或全量哈希。独立根 `{config['data']['root']}`；原图字节保持不变；RGB→短边Resize92→CenterCrop84，val_pre/test_pre 各生成一次PNG，使用pre路径时不再resize。ImageFolder映射、文件状态及提取时的hash见 [prepared.json](data_protocol/prepared.json)。

SCEG 来源 [Legenddddd/SCEG@{config['official_commit']}](https://github.com/Legenddddd/SCEG/tree/{config['official_commit']})；已核对固定版本 scripts/run.sh 和 utils/util.py，并复用模型/损失/trainer/evaluator及 [补充/差异核验](../../docs/discrepancy_report.md)。

| 配方 | 设置 | 来源性质 |
|---|---|---|
| Dogs网络 | stanford_dog，70类，ResNet12，84×84，stage3→stage2 | Dogs类数/分支 + 通用实现 |
| 训练episode | 10-way、5 support、15 query，support-first；按样本消耗，不固定每轮episode数 | 公开CUB脚本/通用trainer |
| 原四项I²L | intra=.85，inter=.05，固定lamda=38 | 沿用公开CUB脚本；非作者Dogs专项确认 |
| 优化与预算 | SGD .1/.9，Nesterov，wd=5e-4；StepLR400/.1；800 epochs | 通用实现/公开脚本及冻结工程 |
| 验证best | 每20 epochs，5-way5-shot，1000 episodes，均值最大 | 作者实际调度；20个val类 |
| 数值与资源 | FP32，AMP=false，TF32=false，deterministic warn_only；seed42，GPU0，workers4，CPU threads2 | 已验证工程环境 |
| 正式测试 | 5/30-way × 1/5-shot，各2000 episodes，query16 | 本轮固定公开协议 |

Dogs lamda=44 是 utils/util.py 的注释；不视作执行配置。loss 温度保持原实现，未加入 optimizer。没有调参、CUB checkpoint 微调或其他训练seed；从独立随机初始化开始。

## 执行与检查

工作区 `/root/projects/aris/sceg_dogs_baseline`，分支 `experiment/sceg_dogs_baseline`，正式运行启动时HEAD为 `3dec9cc569dc3b8b07e1dd35423f9ed07ca25a51`，没有回退或merge。后续按用户授权提交并推送附件和本轮代码/暂停记录，保持启动时的源码版本记录不变。环境复用 `/root/projects/aris/sceg_fsfg/.venv/bin/python`，没有安装/升级依赖。

准备命令：`rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_dogs_dn4.prepare`。
必要测试：`rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m pytest -q tests/test_dogs_protocol.py tests/test_dogs_dn4_protocol.py`；12项通过；包括CSV来源/映射、入口隔离、5-way复用及评测RNG记录的流不变检查。
真实数据检查：`rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_dogs_dn4.check`；1个训练forward/backward与2个验证episode通过，global=[200,70]，episode=[150,10]，loss={tests['loss']:.6f}，活动参数梯度通过；6张原图/pre输入逐元素相同。没有保存smoke权重或继承smoke状态。旧 [Dogs合成smoke/恢复](../sceg_dogs_baseline/REPORT.md) 和 [CUB等价性结论](../../docs/cub_reproduction_report.md) 直接引用，没有重复实验。

真实正式命令（cwd为上述工作区）：
```bash
{command}
```

持久 screen 会话 `{metrics.get('screen_session')}`，Python PID `{metrics.get('process_id')}`。日志 [train.log]({run.relative_to(BASE)}/train.log)、[metrics.json]({run.relative_to(BASE)}/metrics.json)、[epoch_history.jsonl]({run.relative_to(BASE)}/epoch_history.jsonl)；实际配置/环境/代码HEAD/差异/未跟踪源码小补丁均在同一run内。启动方式见运行command/cwd。只运行本任务一个GPU进程，没有终止其他任务。

恢复命令（确认原进程已经退出后使用同一run；latest、best原位滚动，不复制权重）：
```bash
{restore}
```
若800轮已完成、仅评测中断，在该命令末尾添加 `--evaluate-only`。会复用已存在的5-way记录，只完成缺项；逐设置保存rng_after，以保持评测随机流续接。正常执行不额外重置各setting的seed。

## 进度、资源与正式结果

最近一次进度写入的执行耗时 {metrics.get('elapsed_seconds')} 秒（尚非完整总耗时）；训练含验证耗时 {metrics.get('training_with_validation_seconds')} 秒；完整测试耗时 {json.dumps(metrics.get('test_seconds'))} 秒；该进度点记录的峰值GPU allocated {metrics.get('gpu_peak_allocated_bytes')} bytes。已完成setting的评测耗时在对应原始test JSON内。曲线见epoch_history及validation_epoch_*.json。

best epoch={summary['best_epoch']}，best验证均值={summary['best_validation']}；权重 `{run/'checkpoint.pth'}`，SHA256={checkpoint_hash}。best尚未选出时文件仅为trainer的临时last，不能当作验证best。只长期保存该目录的latest.pth与checkpoint.pth；前者含model/optimizer/scheduler/RNG/DropBlock counters；无历史epoch权重或按setting复制权重。

| setting | accuracy (%) ± 原版95% CI半宽 | 正式episodes |
|---|---|---|
{chr(10).join(rows)}

原始记录为run内test_*way_*shot.json：逐episode准确率、实际episode索引、样本路径表、同一checkpoint hash。生成报告时从逐episode记录重算均值与 `1.96 * sqrt(var(episode_accuracy, ddof=0) / N)`，并检查2000个任务及索引范围/无重复。CI仅反映固定模型与划分下episode抽样变化，不代表多训练seed稳定性。

## 结论与范围

{('在可公开追溯的 Dogs 协议上建立了原版 SCEG 本地基线。' if summary['full_training_completed'] and summary['full_evaluation_completed'] else '公开协议与必要技术检查已经核验，正式任务尚未完整完成；当前不能宣称已建立完成的数值基线。')} {split_note} 本轮始终是公开DN4协议基线，不改称作者split复现，不把原论文差值视为模块增益，不设置追平/超论文数字门槛。一个seed不能宣称稳健性或保证符合其他论文的比较协议。

已有CUB、ProKeR v1及旧Dogs blocked记录元数据保持不变={summary['protected_artifacts_unchanged']}，核验见 [preservation_check.json](preservation_check.json)。旧Dogs prepare/run仅做兼容入口适配，旧配置/split/runs/报告没有回填成功。共享trainer仅适配同路径续训best不复制和过滤记录metadata，未改变公式、增强、初始化、采样或调度。没有运行ProKeR、CUB重训、其他数据集、多seed或调参。完成本轮后停止。
"""
    (BASE/"report.md").write_text(text, encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--capture-before", action="store_true")
    args = parser.parse_args()
    if args.capture_before:
        path = BASE/"preservation_check.json"
        if path.exists():
            raise ValueError("Preservation baseline already exists")
        write_json(path, {"before": protected_state()})
    else:
        summary = generate_report(args.run_id)
        print(json.dumps({key: summary[key] for key in ("status", "completed_epochs", "best_epoch", "full_evaluation_completed")}))


if __name__ == "__main__":
    main()
