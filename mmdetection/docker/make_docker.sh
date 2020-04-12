#!/bin/bash

# large config
docker cp /mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/configs/chongqing/large/round2_large_cr_r50_fpn_1x_dcn_11.py mmdet:/mmdetection/configs/
# small config 
docker cp /mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/configs/chongqing/small/round2_small_cr_r50_dcn_1x.py mmdet:/mmdetection/configs/

docker cp /mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/configs/chongqing/small/reppoints_moment_r50_fpn_2x_mt.py mmdet:/mmdetection/configs/

docker cp /mnt/hdd/tangwei/chongqing/mmdetection/configs/chongqing/round2/small/round2_small_cr_r50_dcn_1x_gn_ws.py mmdet:/mmdetection/mmdetection/configs/round2/small/

# large pth
docker cp /mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/work_dirs/r2_large_cr_dcn_r50_fpn_1x_20200226153657/large.pth mmdet:/mmdetection/work_dirs/
# small pth 
docker cp /mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/work_dirs/r2_small_cr_r50_fpn_1x_dcn_aug_20200225212246/small.pth mmdet:/mmdetection/work_dirs/
docker cp "/mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/work_dirs/reppoints_moment_r50_fpn_2x_mt_20200302171016/epoch_24.pth" mmdet:/mmdetection/work_dirs/
"/mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/work_dirs/reppoints_moment_r50_fpn_2x_mt_20200301161147/epoch_24.pth"
# run.sh
docker cp /mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/docker/run.sh mmdet:/mmdetection/


# submit.py
docker cp /mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/docker/submit.py mmdet:/mmdetection/
docker cp /mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/tools/sub_seqnms.py mmdet:/mmdetection/
"/mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/tools/sub_seqnms.py"

# test.py
docker cp /mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/tools/my_test.py mmdet:/mmdetection/tools/

# small 
python tools/publish_model.py ./code/work_dirs/r2_small_cr_r50_fpn_1x_dcn_aug_20200225005930/epoch_12.pth ./code/work_dirs/r2_small_cr_r50_fpn_1x_dcn_aug_20200225005930/small.pth
# large
python tools/publish_model.py ./code/work_dirs/round2_large_cr_dcn_r50_fpn_1x_20200220161254/epoch_12.pth ./code/work_dirs/round2_large_cr_dcn_r50_fpn_1x_20200220161254/large.pth

# small dataset
# docker cp /mnt/hdd/tangwei/chongqing/mmdetection/mmdet/datasets/chongqingSmallDet.py mmdet:/mmdetection/mmdetection/mmdet/datasets/
docker cp /mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/mmdet/datasets/chongqing11Det.py mmdet:/mmdetection/mmdet/datasets/
docker cp /mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/mmdet/datasets/__init__.py mmdet:/mmdetection/mmdet/datasets/

# testA small annotations
# docker cp /mnt/hdd/tangwei/chongqing/tcsubmit/instances_testA_small.json mmdet:/mmdetection/coco/annotations/

docker cp /mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/coco/annotations/instances_testA_11.json mmdet:/mmdetection/coco/annotations/

docker cp /mnt/hdd/tangwei/chongqing/tcdata/ mmdet:/tcdata/

docker container exec -it mmdet /bin/bash
nvidia-docker run -v /mnt/hdd/tangwei/chongqing/tcdata/:/tcdata/ registry.cn-shenzhen.aliyuncs.com/cqimg/cqsubmit:0219.2 sh run.sh
