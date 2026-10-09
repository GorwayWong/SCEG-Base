# Data provenance and episode protocol

补充材料§7.1确认以下类别数；many-way等于novel类别数：

|数据集|base|validation|novel|本项目状态|
|---|---:|---:|---:|---|
|CUB|100|50|50|FRN公开规则的独立物理副本已准备；旧划分另保留|
|Dogs|70|20|30|原始20580图片已下载；精确类别身份待核验|
|Cars|130|17|49|已有独立物理副本、全图decode及哈希通过；原始像素等价性未核验|
|Flowers102|51|26|25|原始8189图片及标签已下载；class split待核验|

来源项目只读：`/root/projects/aris/fewshot/data/fsfg`。来源图像目录实际使用符号链接；首次`cp -a`被审计拒绝后，重新用`cp -aL --reflink=auto`物理复制，删除本项目中被拒绝的临时链接目录。原项目没有改动。当前目标不存在符号链接/指向来源的硬链接。

CUB 11,788图，目录树SHA256 `4e7ab7af7cb1d5922c9d5ab6ff724a03897ab55ef130ee9f4ac7b3cb2e48792c`；Cars16,185图，`90b98fb285bc73d000359e1223126c9e891f66b4c271847425dfa9d38157cd74`。逐图片SHA256、class IDs及原provenance均保存在data/provenance/copied_datasets.json；class split集合两两无交集。

CUB旧来源脚本按0-based索引分配类，不能称为FRN实际split。[FRN公开初始化脚本](https://github.com/Tsingularity/FRN/blob/64e43f3ca3a4953445788fc853e288e032d0083c/data/init_CUB_fewshot.py#L77)遍历原始class IDs1..200：even IDs为train，id%4==1为val，id%4==3为test。该规则与旧来源完全不同。正式根`data/frn/CUB_fewshot_raw`严格复制发布规则，100/50/50类，5903/2927/2958图；11,788图片bytes全部与已有来源一致，新tree SHA256 `a390d9ae0840bf6fd27311b937ee57a676942aa60c0598155a274fe7d21f0cb0`。旧副本、旧overfit和旧预缩放清单完整保留，不覆盖历史。

Cars来源脚本为DN4风格seed196抽类，但依赖os.listdir顺序；本项目保留已生成清单，不重新运行脚本。Cars来源清单记录镜像版本、标签对照与原始archive像素等价性未验证。

作者README要求参考FRN；CUB采用可核验的FRN脚本，不能再用“同样100/50/50”充当身份相同的证明。SCEG没有提供自身四数据集完整class清单，FRN也没有Dogs/Cars/Flowers初始化协议，后者仍有来源缺口。未知split不按seed猜测，不使用Flowers原有按图片划分的setid.mat代替few-shot按类划分。

训练：ImageFolder按排序class目录给全局标签；每轮各类图像shuffle，按剩余图像数加权抽way个类。先输出所有类support（每类shot），再所有类query（每类15）；只消耗本轮图片、不重复，剩余不足shot+query移除该类。epoch含有动态数量的episodes，不能随意换成固定episode数。

测试：随机shuffle class列表，取前way；类内shuffle，先shot support后16 query。官方query_shot被忽略，因此正式协议固定16。按每episode百分比accuracy，取均值与 `1.96*sqrt(var(ddof=0)/n)`；2000测试episodes。

训练变换：PIL RGB→RandomResizedCrop84→ColorJitter(.4,.4,.4)→RandomHorizontalFlip→ToTensor→ImageNet normalize。验证/测试预缩放由官方utils/util.py:21–52规定：Resize(shorter side92)→CenterCrop84→保存PNG；pre模式只ToTensor/normalize。已生成正式FRN CUB/Cars的11,326个PNG，每张均重算变换并对照保存像素，清单见data/provenance/pre_resized.json；旧划分预缩放另存pre_resized_legacy_closerlook.json。debug的确定性变换不等于正式训练aug。

Dogs的HTTPS端点证书不匹配，已从同一发布者HTTP端点下载793,579,520-byte images.tar及lists.tar，完整20580图/120breed headers通过。Flowers从Oxford官方站点下载344,862,509-byte tar和labels，8189图与8189个1..102标签通过。SHA256及失败/重试记录保存在data/provenance/downloads*.json，raw_archive_audit.json记录审计范围。下载完成不等于已核验SCEG class split；Dogs/Flowers未按猜测建立train/val/test。
