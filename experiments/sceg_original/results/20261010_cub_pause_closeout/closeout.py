"""Record the requested evaluation pause and publish the three completed tests."""
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
RUN_ID = "20261010_cub800_seed42_resume122"
RUN = ROOT / "experiments/sceg_original/runs" / RUN_ID
MANY = RUN.with_name(RUN_ID + "_test50")
OUTPUT = ROOT / "experiments/sceg_original/results" / RUN_ID
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(OUTPUT))
from workflow import verified_result, write_debug_report
from utils.experiment import write_json


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(HERE / "closeout.log", mode="x")])
    now = datetime.now(timezone.utc).isoformat()
    metrics = json.loads((RUN / "metrics.json").read_text())
    assert metrics["completed_epochs"] == 800 and len(metrics["epoch_history"]) == 800
    commit = (RUN / "git_commit.txt").read_text().strip()
    checkpoint_sha = hashlib.sha256((RUN / "checkpoint_best.pth").read_bytes()).hexdigest()
    results = [verified_result(parent / f"test_{way}way_{shot}shot.json", checkpoint_sha, commit)
               for way, shot, parent in [(5, 1, RUN), (5, 5, RUN), (50, 1, MANY)]]
    assert not (MANY / "test_50way_5shot.json").exists()
    pause = {"time_utc": now, "reason": "user_requested_pause", "way": 50, "shot": 5,
             "episodes_requested": 2000, "accuracy": None, "confidence_interval": None,
             "completed_episodes": None, "episode_progress_persisted": False,
             "note": "The existing evaluator saves at setting completion; no valid completed 5-shot result exists.",
             "automatic_cleanup_publisher_cancelled": True, "training_resume_required": False}
    write_json(OUTPUT / "pause_record.json", pause)
    debug = write_debug_report(results, None, RUN)
    aggregate = {"dataset": "CUB", "run_id": RUN_ID, "status": "paused", "phase": "D",
        "full_training_completed": True, "completed_epochs": 800,
        "formal_evaluation_completed": False, "baseline_protocol_completed": False,
        "seed": 42, "commit": commit, "checkpoint_sha256": checkpoint_sha,
        "evaluations": results, "pending_evaluations": [pause],
        "initial_completed_epochs": 122, "best_epoch": metrics["best_epoch"],
        "reproduction_debug_report": debug, "pause_time_utc": now}
    for result in results:
        directory = OUTPUT / f"{result['way']}way_{result['shot']}shot"
        directory.mkdir(exist_ok=False)
        write_json(directory / "results.json", result)
        write_json(directory / "result.json", result)
    for name in ["results.json", "result.json", "status.json"]:
        write_json(OUTPUT / name, aggregate)
    write_json(OUTPUT / "phase_d_checks.json", {"status": "partial", "completed_settings": 3,
        "episodes_per_completed_setting": 2000, "same_best_checkpoint": True,
        "raw_means_and_ci_recomputed": True, "pending": "50-way 5-shot paused by user"})
    many_metrics = json.loads((MANY / "metrics.json").read_text())
    many_metrics.update(status="paused", pending_evaluation=pause,
                        completed_evaluation=results[-1], pause_time_utc=now)
    write_json(MANY / "metrics.json", many_metrics)
    write_json(MANY / "evaluation_checks.json", {"status": "partial", "completed": "50-way 1-shot",
                                                "pending": "50-way 5-shot", "pause_time_utc": now})
    write_json(ROOT / "experiments/sceg_original/logs/20261010_final_cleanup/status.json",
               {"status": "manual_closeout", "automatic_finalizer_cancelled": True,
                "run_id": RUN_ID, "experiment_status": "paused"})
    cells = [f"{r['accuracy']:.2f} ± {r['confidence_interval']:.2f}" for r in results]
    table = "# CUB result table\n\nAccuracy (%)；95% CI为官方正态区间半宽。已完成项目各2000 episodes，query16，seed42。\n\n"
    table += "|Method|5-way 1-shot|5-way 5-shot|50-way 1-shot|50-way 5-shot|\n|---|---:|---:|---:|---:|\n"
    table += "|SCEG reproduced (partial)|" + "|".join(cells) + "|—（暂停）|\n|Paper|87.79|94.56|56.02|73.69|\n"
    table += "\n50-way 5-shot尚无完成结果；不将已处理但未持久化的episodes计为正式结果。论文CI未补填。\n"
    table += "\n论文：[CVF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhao_From_Few-way_to_Many-way_Rethinking_Few-shot_Fine-grained_Image_Classification_CVPR_2026_paper.pdf)。\n"
    (ROOT / "docs/cub_result_table.md").write_text(table, encoding="utf-8")
    report = "# CUB reproduction report\n\n"
    report += "## 实验目的与当前状态\n\n按固定官方代码复现CUB baseline，不调参、不修改已验证模型/loss/dataset。800轮训练完成，三组正式测试完成；2026-10-10按用户要求暂停50-way 5-shot。完整完成标准尚未满足，不能声明完整CUB baseline reproduced。\n\n"
    report += "## 数据来源\n\nFRN固定commit `64e43f3ca3a4953445788fc853e288e032d0083c`；原始ID偶数train、mod4=1 val、mod4=3 test。100/50/50类，5903/2927/2958图，总11788。正式根`data/frn/CUB_fewshot_raw`。tree SHA256 `a390d9ae0840bf6fd27311b937ee57a676942aa60c0598155a274fe7d21f0cb0`。\n\n"
    report += "Phase A全部图片hash、decode/RGB与5885预缩放PNG像素对照通过，manifest保存在`experiments/sceg_original/runs/20261009_cub800_seed42_phase_a_verified/data_manifest.json`。原始记录保存的是实验准备时状态；实验后的内部JPEG去重另见清理记录。\n\n"
    report += "## 配置\n\n官方commit `f7f130aeb5b1b49246150574c53a874d4ce8de80`。10-way5-shot15-query；SGD lr0.1、momentum0.9、Nesterov、weight_decay5e-4；StepLR400 gamma0.1；800 epochs；lambda38/inter0.05/intra0.85；seed42。workers4为用户批准的执行差异，官方48。所有参数在`experiments/sceg_original/configs/cub.yaml`与run/config_resolved.yaml显式记录。\n\n"
    report += "## 训练细节\n\n使用既有`tools.train.main → trainers.trainer.train`。前序训练到122轮后中断，恢复模型/SGD/StepLR/RNG/DropBlock状态并训练至800轮；完整800轮历史保存在最终run的metrics.json。每20轮5-way5-shot、1000 episodes验证，以严格提升选best。"
    report += f"best epoch {metrics['best_epoch']}，validation {metrics['best_validation']:.5f}%。总参数/可训练标记参数16785795；保留官方12个无梯度张量，112个活动张量通过检查。200张84×84图的支持算子前向FLOPs估算1463450445592，不包含backward或未支持算子。\n\n"
    report += "独立预检epoch1→resume epoch2的loss34.962846→20.538388，状态与不中断运行对齐。最终仅保留last/best与当前RNG；完整权重不逐轮归档。\n\n"
    report += "## 测试协议\n\n使用同一validation-selected best checkpoint。三组已完成测试各2000 episodes、query16，mean episode accuracy，95% CI半宽1.96×sqrt(var(ddof=0)/n)。原始episode indices/accuracies保留，均值、CI与episode数量独立重算通过。"
    report += f"best SHA256 `{checkpoint_sha}`。50-way 5-shot进程已停止；原入口不保存中途episode状态，后续恢复评测仍使用此best checkpoint，无需重新训练。\n\n"
    report += "## 结果\n\n" + table.split("# CUB result table\n\n", 1)[1]
    report += f"\n权威结果：`{OUTPUT.relative_to(ROOT)}/results.json`。低于论文的两项5-way结果已按split→sampler→augmentation→optimizer→loss核查，报告`{debug}`；未调参或重新训练。\n\n"
    report += "## 完成标准\n\n- [x] CUB manifest与官方配置冻结\n- [x] 800轮训练完成，best checkpoint可加载\n- [x] 5-way 1-shot/5-shot各2000 episodes\n- [x] 50-way 1-shot 2000 episodes\n- [ ] 50-way 5-shot（用户暂停）\n- [x] result.json/results.json与discrepancy记录\n\n单seed结果不证明跨seed或逐bit一致性；workers、历史环境及官方行为差异见cub_discrepancy.md。\n"
    (ROOT / "docs/cub_reproduction_report.md").write_text(report, encoding="utf-8")
    write_json(OUTPUT / "phase_e_checks.json", {"status": "recorded_partial_closeout",
        "documents": ["docs/cub_reproduction_report.md", "docs/cub_result_table.md", "docs/cub_discrepancy.md"],
        "baseline_protocol_completed": False, "missing": "50-way 5-shot"})
    logging.info("Training800 retained; three 2000-episode results verified; 50-way5-shot paused.")


if __name__ == "__main__":
    main()
