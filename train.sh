#!/bin/bash
cd mmdetection/
tools/dist_train.sh undersonar/configs/all_cascade_r50.py 2
tools/dist_train.sh undersonar/configs/all_cascade_r50_speckle.py 2
tools/dist_train.sh undersonar/configs/all_cascade_r50_101.py 2
