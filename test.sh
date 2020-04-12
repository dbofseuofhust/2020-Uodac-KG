#!/bin/bash
cd mmdetection/
# tools/dist_test.sh undersonar/configs/all_cascade_r50.py undersonar/work_dirs/r50_gc_epoch_ensemble.pth 2 --out undersonar/json/r50_gc.pkl
# tools/dist_test.sh undersonar/configs/all_cascade_r50.py undersonar/work_dirs/r50_db_epoch_ensemble.pth 2 --out undersonar/json/r50_db.pkl
# tools/dist_test.sh undersonar/configs/all_cascade_r50_speckle.py undersonar/work_dirs/r50_speckle_epoch_ensemble.pth 2 --out undersonar/json/r50_speckle.pkl
# tools/dist_test.sh undersonar/configs/all_cascade_r101.py undersonar/work_dirs/r101_epoch_ensemble.pth 2 --out undersonar/json/r101.pkl
# python tools/my_test.py undersonar/configs/all_cascade_r50.py undersonar/work_dirs/r50_gc_epoch_ensemble.pth --out undersonar/json/r50_gc.pkl
# python tools/my_test.py undersonar/configs/all_cascade_r50.py undersonar/work_dirs/r50_db_epoch_ensemble.pth --out undersonar/json/r50_db.pkl
# python tools/my_test.py undersonar/configs/all_cascade_r50_speckle.py undersonar/work_dirs/r50_speckle_epoch_ensemble.pth  --out undersonar/json/r50_speckle.pkl
# python tools/my_test.py undersonar/configs/all_cascade_r101.py undersonar/work_dirs/r101_epoch_ensemble.pth --out undersonar/json/r101.pkl

tools/dist_test.sh undersonar/configs/all_cascade_r50.py undersonar/work_dirs/r50_gc_epoch_ensemble.pth 2 --json_out undersonar/json/r50_gc.json
tools/dist_test.sh undersonar/configs/all_cascade_r50.py undersonar/work_dirs/r50_db_epoch_ensemble.pth 2 --json_out undersonar/json/r50_db.json
tools/dist_test.sh undersonar/configs/all_cascade_r50_speckle.py undersonar/work_dirs/r50_speckle_epoch_ensemble.pth 2 --json_out undersonar/json/r50_speckle.json
tools/dist_test.sh undersonar/configs/all_cascade_r101.py undersonar/work_dirs/r101_epoch_ensemble.pth 2 --json_out undersonar/json/r101.json