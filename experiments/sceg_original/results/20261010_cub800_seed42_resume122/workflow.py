"""Durable Phase C/D/E execution; no retries, tuning or additional training."""
import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from baseline import PHASE_A, verify_protected, sha
from utils.experiment import write_json

RUN_ID = "20261009_cub800_seed42_formal"
OWNER_MARKER = f"<!-- run_id: {RUN_ID} -->"
PAPER = {(5, 1): 87.79, (5, 5): 94.56, (50, 1): 56.02, (50, 5): 73.69}


def update_document(path, text):
    path = ROOT / path
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(OWNER_MARKER + "\n" + text, encoding="utf-8")
    temporary.replace(path)


def invoke(arguments, path):
    command = ["rtk", "proxy", str(ROOT / ".venv/bin/python"), *arguments]
    logging.info("command=%s", command)
    with path.open("x", encoding="utf-8") as output:
        process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        logging.info("child_pid=%d output=%s", process.pid, path)
        returncode = process.wait()
    if returncode:
        raise RuntimeError(f"Command exited {returncode}; see {path}")


def verified_result(path, checkpoint_sha, commit):
    raw = json.loads(path.read_text())
    values = np.asarray(raw["episode_accuracies"], dtype=float)
    assert raw["episodes"] == len(values) == 2000
    assert raw["query"] == 16 and np.isfinite(values).all()
    assert len(raw["episode_indices"]) == 2000
    assert all(len(indices) == raw["way"] * (raw["shot"] + 16) and len(set(indices)) == len(indices)
               for indices in raw["episode_indices"])
    assert np.isclose(raw["accuracy"], values.mean(), atol=1e-12)
    ci = 1.96 * np.sqrt(values.var(ddof=0) / len(values))
    assert np.isclose(raw["confidence_interval"], ci, atol=1e-12)
    return {"dataset": "CUB", "checkpoint_sha256": checkpoint_sha, "way": raw["way"],
        "shot": raw["shot"], "query": 16, "episodes": 2000, "accuracy": raw["accuracy"],
        "confidence_interval": raw["confidence_interval"], "seed": 42, "commit": commit,
        "official_commit": "f7f130aeb5b1b49246150574c53a874d4ce8de80",
        "confidence_interval_type": raw["confidence_interval_type"],
        "confidence_interval_bounds": [raw["accuracy"] - ci, raw["accuracy"] + ci],
        "raw_result": str(path.relative_to(ROOT)), "raw_result_sha256": sha(path)}


def write_debug_report(results, config, run):
    deficits = [r for r in results if r["accuracy"] < PAPER[(r["way"], r["shot"])]]
    if not deficits:
        return None
    text = "# Reproduction debug report\n\n"
    text += "指标低于论文时仅核查既有证据，不调整超参数或重新训练。下述通过仅排除已检查的配置/数据错误，不能证明性能差异原因。\n\n"
    text += "|Setting|Reproduced|Paper|Gap (percentage points)|\n|---|---:|---:|---:|\n"
    for r in deficits:
        target = PAPER[(r["way"], r["shot"])]
        text += f"|{r['way']}-way {r['shot']}-shot|{r['accuracy']:.4f}|{target:.2f}|{r['accuracy']-target:.4f}|\n"
    text += "\n|hypothesis|experiment / evidence|result|\n|---|---|---|\n"
    text += "|数据split不一致|Phase A原始class IDs、逐图hash与FRN固定规则核对；50-way测试前再次hash原图/缓存|100/50/50类与5903/2927/2958图通过；官方SCEG未提供自己的独立类别清单，仍有来源限制|\n"
    text += "|episode sampler/标签顺序错误|Phase A完整seed42 sampler检查与eval support/query标签断言、保存2000 episode indices|weighted depletion、support先行、query16；逐episode数量/无重复及均值CI重算通过|\n"
    text += "|augmentation不同|已保护datasets/transforms.py源码哈希；Phase A全部PNG像素重算对照；冻结config|官方训练crop84/jitter.4/flip.5，eval Resize92→crop84；workers4可改变随机增强流|\n"
    text += "|optimizer/scheduler不同|config_resolved.yaml及800轮epoch_metrics的lr_used、checkpoint完整SGD/StepLR状态|SGD.1/.9/Nesterov/5e-4，StepLR400 gamma.1；没有调参|\n"
    text += "|loss不同|已保护loss源码哈希、既有数值/梯度验证与冻结config|四项原loss固定lambda38/inter.05/intra.85，lambda未进入optimizer；未重新实现|\n"
    text += "\n未解决：作者历史环境、不同GPU非确定性、单seed波动及官方精确数据副本身份。未执行参数搜索，也不将上述不确定性当作已证明原因。\n"
    target = run / "reproduction_debug_report.md"
    with target.open("x", encoding="utf-8") as stream:
        stream.write(text)
    return str(target.relative_to(ROOT))


def complete_reports(run, output, results, metrics, debug_report, elapsed):
    a = json.loads((PHASE_A / "data_manifest.json").read_text())
    profile = json.loads((run / "flops.json").read_text())
    parameters = json.loads((run / "parameter_summary.json").read_text())
    start_note = "从seed42重新初始化训练"
    if (run / "resume_input.json").exists():
        resumed = json.loads((run / "resume_input.json").read_text())
        start_note = f"从已完成{resumed['completed_epochs']}epochs的checkpoint续训"
    cells = [f"{r['accuracy']:.2f} ± {r['confidence_interval']:.2f}" for r in results]
    table = "# CUB result table\n\n"
    table += "Accuracy (%)，95% CI为官方正态区间半宽，每项2000 episodes、query16、seed42。\n\n"
    table += "|Method|5-way 1-shot|5-way 5-shot|50-way 1-shot|50-way 5-shot|\n|---|---:|---:|---:|---:|\n"
    table += "|SCEG reproduced|" + "|".join(cells) + "|\n|Paper|87.79|94.56|56.02|73.69|\n"
    table += "\n论文：[CVF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhao_From_Few-way_to_Many-way_Rethinking_Few-shot_Fine-grained_Image_Classification_CVPR_2026_paper.pdf)。论文CI未在此补填。\n"
    update_document("docs/cub_result_table.md", table)
    report = "# CUB baseline reproduction report\n\n"
    report += "## 实验目的\n\n复现官方SCEG CUB baseline；不提升指标，不修改已验证模型/loss/dataset，不调参。完成标准通过后记录 `SCEG CUB baseline reproduced`；这代表协议执行完成，不代表论文数字逐bit或统计一致。\n\n"
    report += f"## 数据来源\n\nFRN固定commit `64e43f3ca3a4953445788fc853e288e032d0083c`，原始ID偶数train、mod4=1 val、mod4=3 test。100/50/50类，5903/2927/2958图，总11788；tree SHA256 `{a['hash']['tree_sha256']}`。Phase A全图decode/RGB/hash与5885预缩放图逐像素对照通过；manifest见 `{PHASE_A.relative_to(ROOT)}/data_manifest.json`。\n\n"
    report += "## 配置\n\n官方commit `f7f130aeb5b1b49246150574c53a874d4ce8de80`；10-way5-shot15-query；SGD lr.1、momentum.9、Nesterov、weight_decay5e-4；StepLR400 gamma.1；800 epochs；lambda38/inter.05/intra.85；seed42。workers4是用户授权的唯一训练执行参数差异，官方48；AMP/TF32关闭，deterministic warn_only保留。所有参数在config_resolved.yaml显式记录。\n\n"
    report += f"## 训练细节\n\n通过现有 `tools.train.main -> trainers.trainer.train`，补充记录层没有替换训练算法。正式run `{run.relative_to(ROOT)}` {start_note}至800epochs；原始seed为42，前序记录通过continuation.json追溯，metrics.json包含合并的完整epoch历史。每20轮5-way5-shot验证1000episodes，以strictly improved accuracy选best：epoch {metrics['best_epoch']}，val {metrics['best_validation']:.4f}%。全部epoch保留轻量指标、日志及验证证据；完整权重仅保留rolling last和validation选出的best，RNG只保留当前状态，不逐轮归档。预检独立重启后loss34.962846→20.538388。参数 {parameters['total']}，trainable {parameters['requires_grad']}；官方闲置参数保留。200图eval支持算子前向FLOPs估算 {profile['forward_flops']}，非全算子/非backward计数。\n\n"
    report += "## 测试协议\n\n800epochs后不重训练，使用同一best checkpoint。5-way1/5-shot由既有trainer自动执行；50-way1/5-shot通过既有tools.test执行。各2000episodes、query16，mean episode accuracy与1.96×sqrt(var(ddof=0)/n)；原始indices/accuracies保留且独立重算通过。\n\n"
    report += f"## 结果\n\n结果见 `docs/cub_result_table.md` 与 `{output.relative_to(ROOT)}/results.json`，best SHA256 `{results[0]['checkpoint_sha256']}`。本次流水线耗时 {elapsed:.1f}秒；续训前的耗时保留在前序epoch记录中。\n\n"
    report += table.split("# CUB result table\n\n", 1)[1]
    if debug_report:
        report += f"\n部分setting低于论文，已按split→sampler→augmentation→optimizer→loss顺序核查并生成 `{debug_report}`；没有调整参数。\n"
    report += "\n## 完成标准\n\n- [x] CUB manifest与配置冻结\n- [x] 800epochs完成\n- [x] best checkpoint可加载\n- [x] 5-way1/5-shot各2000episode\n- [x] 50-way1/5-shot各2000episode\n- [x] result.json/results.json生成\n- [x] discrepancy记录完成\n\n限制与差异见 `docs/cub_discrepancy.md`。\n"
    update_document("docs/cub_reproduction_report.md", report)


def main():
    global RUN_ID, OWNER_MARKER
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    RUN_ID = args.run_id
    OWNER_MARKER = f"<!-- run_id: {RUN_ID} -->"
    verify_protected()
    execution = Path(__file__).resolve().parent
    # Each launch is exclusive; no automatic retry can overwrite a prior job.
    logs = ROOT / "experiments/sceg_original/logs" / RUN_ID
    output = ROOT / "experiments/sceg_original/results" / RUN_ID
    logs.mkdir(exist_ok=False)
    output.mkdir(exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(logs / "workflow.log", mode="x"), logging.StreamHandler()])
    start = time.monotonic()
    phase = "C"
    write_json(output / "process.json", {"supervisor_pid": os.getpid(), "run_id": RUN_ID,
        "command": sys.orig_argv, "cwd": str(ROOT), "workflow_sha256": sha(__file__),
        "resume_source": str(args.resume.resolve()) if args.resume else None})
    write_json(output / "status.json", {"status": "running", "phase": phase,
        "run_id": RUN_ID, "full_training_completed": False, "formal_evaluation_completed": False})
    run = ROOT / "experiments/sceg_original/runs" / RUN_ID
    try:
        for name in ("baseline.py", "evaluate.py", "workflow.py"):
            shutil.copy2(execution / name, output / name)
        train_command = [str(execution / "baseline.py"), "--run-id", RUN_ID]
        if args.resume:
            train_command += ["--resume", str(args.resume)]
            logging.info("Phase C: continue %s from %s to epoch800", RUN_ID, args.resume)
        else:
            logging.info("Phase C: start fresh 800-epoch run %s", RUN_ID)
        invoke(train_command, logs / "train_process.log")
        metrics = json.loads((run / "metrics.json").read_text())
        assert metrics["status"] == "completed" and metrics["completed_epochs"] == 800
        assert metrics["full_training_completed"] and metrics["best_epoch"] is not None
        shutil.copy2(run / "phase_c_checks.json", output / "phase_c_checks.json")
        phase = "D"
        checkpoint = run / "checkpoint_best.pth"
        checkpoint_sha = sha(checkpoint)
        assert checkpoint.resolve() == (run / "checkpoint.pth").resolve()
        commit = (run / "git_commit.txt").read_text().strip()
        write_json(output / "status.json", {"status": "running", "phase": phase,
            "run_id": RUN_ID, "full_training_completed": True, "formal_evaluation_completed": False})
        logging.info("Phase C completed: epochs=800 best_epoch=%s. Phase D: 5-way complete; start 50-way.", metrics["best_epoch"])
        many_id = RUN_ID + "_test50"
        invoke([str(execution / "evaluate.py"), "--run-id", many_id,
                "--checkpoint", str(checkpoint), "--way", "50"], logs / "test50_process.log")
        many = ROOT / "experiments/sceg_original/runs" / many_id
        assert json.loads((many / "evaluation_checks.json").read_text())["status"] == "passed"
        results = [verified_result(parent / f"test_{way}way_{shot}shot.json", checkpoint_sha, commit)
                   for way, parent in ((5, run), (50, many)) for shot in (1, 5)]
        verify_protected()
        for result in results:
            destination = output / f"{result['way']}way_{result['shot']}shot"
            destination.mkdir()
            write_json(destination / "results.json", result)
            shutil.copy2(destination / "results.json", destination / "result.json")
        write_json(output / "phase_d_checks.json", {"status": "passed", "settings": 4,
            "episodes_per_setting": 2000, "same_best_checkpoint": True,
            "checkpoint_sha256": checkpoint_sha, "raw_means_and_ci_recomputed": True})
        logging.info("Phase D completed: all four settings, 2000 episodes each; mean/CI independently verified.")
        phase = "E"
        config = json.loads((run / "execution_adapter.json").read_text())
        debug_report = write_debug_report(results, config, run)
        aggregate = {"dataset": "CUB", "run_id": RUN_ID, "status": "completed",
            "full_training_completed": True, "completed_epochs": 800, "seed": 42,
            "commit": commit, "checkpoint_sha256": checkpoint_sha,
            "evaluations": results, "reproduction_debug_report": debug_report,
            "initial_completed_epochs": metrics.get("initial_completed_epochs", 0),
            "resumed_from": metrics.get("resumed_from"),
            "baseline_protocol_completed": True}
        write_json(output / "results.json", aggregate)
        shutil.copy2(output / "results.json", output / "result.json")
        complete_reports(run, output, results, metrics, debug_report, time.monotonic() - start)
        write_json(output / "phase_e_checks.json", {"status": "passed",
            "documents": ["docs/cub_reproduction_report.md", "docs/cub_discrepancy.md", "docs/cub_result_table.md"],
            "protected_sources_and_existing_artifacts_unchanged": True})
        write_json(output / "status.json", {**aggregate, "phase": "E", "elapsed_seconds": time.monotonic() - start})
        logging.info("Phase E completed. SCEG CUB baseline reproduced. Results: %s", output / "results.json")
    except BaseException as error:
        write_json(output / "status.json", {"status": "failed", "phase": phase, "run_id": RUN_ID,
            "error": f"{type(error).__name__}: {error}", "full_training_completed": phase != "C",
            "formal_evaluation_completed": False, "elapsed_seconds": time.monotonic() - start})
        logging.exception("Workflow stopped; no tuning/retraining/retry performed")
        raise


if __name__ == "__main__":
    main()
