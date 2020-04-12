#!/bin/bash
# # prepare training data
cd mmdetection/undersonar/data/
# unzip train.zip
# mv train/前视声呐图像 train/qian
# mv train/侧扫声呐图像 train/ce
# mv train/负样本 train/fu
# mv train/ce/image/ss-436.JPG train/ce/image/ss-436.jpg

# # prepare testB data
# rar x image-B-test.rar
# mv image/前视声呐 image/qian
# mv image/侧扫声呐 image/ce
# mkdir image/all
# cp image/qian/* image/all
# cp image/ce/* image/all

cd ..
# convert xml to json
python src/xml2json.py

# generate coco format anno
python src/coco_trian.py
python src/coco_test.py


