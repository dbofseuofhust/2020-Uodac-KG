#!/bin/bash

# python mmdetection/code/coco_test.py
# python mmdetection/tools/test.py mmdetection/configs/chongqing/small/cr_r50_dcn_1x_small.py mmdetection/code/work_dirs/submi-950b6417.pth --json_out mmdetection/code/json/testA.json
# python mmdetection/code/submit.py mmdetection/code/json/testA.bbox.json --mode small
# python inference_small.py
# python inference_large.py

python tools/my_test.py configs/reppoints_moment_r50_fpn_2x_mt.py work_dirs/epoch_24.pth --json_out small.json
python submit.py small.bbox.json --mode small

# python tools/my_test.py configs/round2_large_cr_r50_fpn_1x_dcn_11.py work_dirs/large.pth --json_out large.json
# python submit.py large.bbox.json --mode large
# python sub_seqnms.py
# python mmdetection/tools/test.py mmdetection/configs/round2/large/round2_large_cr_r50_fpn_1x_dcn_aug.py mmdetection/code/work_dirs/large-aa78fc51.pth --json_out large.json
# python submit.py large.bbox.json --mode large
# python merge_json.py --large_json result_large.json --small_json result_small.json
