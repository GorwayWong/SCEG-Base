# 清理与发布审计

日期：2026-10-09。范围仅为本仓库本轮产物；未修改/root/projects/aris/fewshot。

|清理项|理由|保留的审计证据|
|---|---|---|
|debug_overfit_report_legacy.md|与主报告重复，已被FRN正式调试取代|独有配置、初始/最终数值、commit与产物链接合并至debug_overfit_report.md；原始运行文件保留|
|data/CUB_fewshot_raw/val_pre、test_pre|旧0-based划分的5,903个PNG不再用于正式流程|原始11,788个JPEG、旧class清单和pre_resized_legacy_closerlook.json保留；正式FRN/Cars的11,326个PNG保留|
|sceg_preflight/runs/full_batch_200/checkpoint.pth|与已提交源码预检文件完全相同的未训练权重|同尺寸67,328,504 bytes、同SHA256 83c96441201f24356f26ea387a8e6820d8fe19c6977b94945bbcf5a02db3e7fa；canonical文件保留，另有checkpoint_reference.json|
|项目__pycache__、.pytest_cache|可重新生成的执行缓存|Python源码、依赖锁定、测试和既有验证记录均保留|

清理前这批文件共149,722,465 bytes（约142.8MiB）；实际磁盘占用受文件系统影响。未删除训练原图、下载archive、划分与来源脚本、任何原始指标/日志、正式预缩放数据、虚拟环境、两个不同数据划分上的overfit checkpoint，亦未改写既有实验的config/command/metrics。

原始CUB副本虽与FRN正式副本有图像重复，仍是prepare_frn_cub.py的输入与历史调试的原路径，故保留。两次overfit的checkpoint哈希不同，故分别保留。所有Git历史commit保留，实验记录中的源码commit仍可查询；不重写历史。

## Git发布内容

轻量运行配置、源码commit/diff、命令、环境、数据清单、step history、指标、诊断、日志和checkpoint哈希清单纳入Git。数据、.venv和.pth保持本地排除；GitHub上的审计通过元数据和原始commit追踪，不能把未训练预检权重或overfit数字当作论文指标。

验证：确认重复checkpoint SHA256相同；检查保留checkpoint及数据哈希、报告链接、运行目录轻量记录和Git差异。模型/训练算法未变，本次不重复GPU训练。
