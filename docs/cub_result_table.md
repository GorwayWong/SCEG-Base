# CUB result table

Accuracy (%)；95% CI为官方正态区间半宽。已完成项目各2000 episodes，query16，seed42。

|Method|5-way 1-shot|5-way 5-shot|50-way 1-shot|50-way 5-shot|
|---|---:|---:|---:|---:|
|SCEG reproduced (partial)|87.61 ± 0.38|94.53 ± 0.19|56.52 ± 0.13|—（暂停）|
|Paper|87.79|94.56|56.02|73.69|

50-way 5-shot尚无完成结果；不将已处理但未持久化的episodes计为正式结果。论文CI未补填。

论文：[CVF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhao_From_Few-way_to_Many-way_Rethinking_Few-shot_Fine-grained_Image_Classification_CVPR_2026_paper.pdf)。
