# CUB paused closeout cleanup

800轮训练完成，三组正式测试完成；按用户要求停止50-way 5-shot与自动后处理，再执行本次清理。完整baseline仍缺50-way 5-shot结果。

主清理脚本移除1614个过时文件/别名，释放1440374784 bytes；随后移除39份非best验证采样明细，释放55439360 bytes。合计Linux文件系统已分配空间减少1495814144 bytes（约1.39 GiB）。另删除未运行的family result占位文件和空自动发布日志。11788张重复CUB JPEG在项目内部硬链接去重，保持全部类目录、图像SHA256和历史debug入口可用；外部数据来源不受影响。

保留最终last/best checkpoint、当前RNG、checkpoint SHA256、完整800轮指标与日志、全部40次validation指标、best epoch480原始验证记录、三组原始2000-episode测试记录、配置/环境/source/data清单及原始执行脚本副本。数据和权重留本地，不上传Git。

移除旧中断run的last/best/RNG、重复单轮JSON、逐轮激活诊断、重复训练进程输出、未运行family result占位文件和已取消的自动发布脚本。冻结配置与已验证模型/loss/dataset/tests源码未修改。历史输出hash清单保留为时点记录，后续记录入口仅保护源码hash。

明细见cleanup_manifest.json、cleanup_result.json与validation_cleanup.json；权威实验结果见`../20261010_cub800_seed42_resume122/results.json`。50-way 5-shot指标为null，无中途episode结果被计入正式统计。
