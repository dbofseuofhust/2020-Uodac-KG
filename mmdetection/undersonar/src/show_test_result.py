import json
import os
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import mmcv

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

'''
name - 检测类别 - string
image_id - 图像编号 - string
confidence - 置信度 - float
xmin - 检测结果左上角x值 - int
ymin - 检测结果左上角y值 - int
xmax - 检测结果右下角x值 - int
ymax - 检测结果右下角y值 - int
'''

def show_test_object():

    mask_dir ="/mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/undersonar/data/a-test-image/image/mask_all/"
    mkdir(mask_dir)

    img_list = os.listdir(img_dir)

    anno_df = pd.read_csv(anno_dir)
    print(anno_df.head())

    prog_bar = mmcv.ProgressBar(len(img_list))
    for img_name in img_list:
        prog_bar.update()
        img_save_path = mask_dir + img_name

        # if os.path.exists(img_save_path):
        #     continue
        img = Image.open(img_dir + img_name).convert('RGB')
        draw = ImageDraw.Draw(img)
        image_id = img_name.replace('.jpg', '.xml')
        imgindex = anno_df[anno_df.image_id == image_id].index.tolist()

        if len(imgindex)== 0:
            continue   

        for i in range(len(imgindex)): 
            index = imgindex[i]
            xmin, ymin, xmax, ymax = anno_df.xmin[index], anno_df.ymin[index], anno_df.xmax[index], anno_df.ymax[index]
            defect_name = anno_df.name[index]
            score = anno_df.confidence[index]
            Font = ImageFont.truetype("/usr/share/fonts/truetype/ttf-dejavu/DejaVuSansCondensed-BoldOblique.ttf",20)
            draw.rectangle((xmin, ymin, xmax, ymax), outline=(255, 0, 0))
            draw.text((xmin, ymin), str(defect_name) + '|' + str(round(score,2)), font=Font, fill=(0, 255, 0))
        
        img.save(img_save_path)
        
img_dir = "/mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/undersonar/data/a-test-image/image/all/"
anno_dir = "/mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/undersonar/src/Weighted-Boxes-Fusion-master/submit.csv"
show_test_object()
    
