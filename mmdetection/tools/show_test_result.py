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




def show_test_object():

    mask_dir = "/mnt/hdd/tangwei/chongqing/new_mmdet/data/" + 'large_0208/'
    mkdir(mask_dir)

    with open(anno_dir) as f:
        data = json.load(f)

    images_df = pd.DataFrame(data['images']) # file_name, id
    anno_df = pd.DataFrame(data['annotations']) # image_id, category_id, bbox, score    

    prog_bar = mmcv.ProgressBar(images_df.shape[0])
    for i in range(images_df.shape[0]):
        prog_bar.update()
        image_id = images_df.id.iloc[i]
        img_name = images_df.file_name.iloc[i]
        img_save_path = mask_dir + img_name
        if os.path.exists(img_save_path):
            continue
        img = Image.open(img_dir + img_name)
        draw = ImageDraw.Draw(img)
        imgindex = anno_df[anno_df.image_id == image_id].index.tolist()

        if len(imgindex)== 0:
            continue   

        for i in range(len(imgindex)): 
            index = imgindex[i]
            bbox = anno_df.bbox[index]
            xmin, ymin, w, h = bbox
            defect_name = anno_df.category_id[index]
            score = anno_df.score[index]
            Font = ImageFont.truetype("/usr/share/fonts/truetype/ttf-dejavu/DejaVuSansCondensed-BoldOblique.ttf",20)
            draw.rectangle((xmin, ymin, xmin+w, ymin+h), outline=(255, 0, 0))
            draw.text((xmin, ymin), str(defect_name) + '|' + str(round(score,2)), font=Font, fill=(0, 255, 0))
        
        img.save(img_save_path)
        
        # break

def txt2json(txt_path):
    f = open(txt_path, 'r')
    res = []
    for line in f.readlines():
        line = line.strip()
        print(line.split(' '))
        image_name, score, x1, y1, x2, y2 = line.split(' ')
        w, h = float(x2)-float(x1), float(y2)-float(y1)
        res.append({'name':image_name, 'defect_name': '11', 'bbox':[float(x1), float(y1), w, h], 'score':float(score)})
    f.close()
    json_name = txt_path.replace('txt', 'json')
    with open(json_name, 'w') as fp:
        json.dump(res, fp, indent=4, separators=(',', ': '), cls=NpEncoder)
    print('over!')

def show_seqnms_object(img_base_name):
    anns_all = pd.read_json(anno_dir)
    mask_dir = "/mnt/hdd/tangwei/chongqing/new_mmdet/data/" + 'large_0207_seq/'
    mkdir(mask_dir)
    frame_names = []
    for frame_ind in range(5): #帧
        # frame_ind -> 图片索引 -> 0/1/2/3/4
        image_name = img_base_name + '_' + str(frame_ind) + '.jpg'
        img_path = img_dir + image_name
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)

        imgindex = anns_all[anns_all.name == image_name].index.tolist()
        if len(imgindex)== 0:
            continue

        for i in range(len(imgindex)): 
            index = imgindex[i]
            bbox = anns_all.bbox[index]
            xmin, ymin, w, h = bbox
            defect_name = anns_all.defect_name[index]
            score = anns_all.score[index]
            img_save_path = mask_dir + image_name
            Font = ImageFont.truetype("/usr/share/fonts/truetype/ttf-dejavu/DejaVuSansCondensed-BoldOblique.ttf",20)
            draw.rectangle((xmin, ymin, xmin+w, ymin+h), outline=(255, 0, 0))
            draw.text((xmin, ymin), str(defect_name) + '|' + str(round(score,2)), font=Font, fill=(0, 255, 0))
            img.save(img_save_path) 

def show_seqnms_object2(image_name, anno_dir):
    anns_all = pd.read_json(anno_dir)
    mask_dir = "/mnt/hdd/tangwei/chongqing/new_mmdet/data/" + 'large_0208_seq/'
    mkdir(mask_dir)

    img_path = img_dir + image_name
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)

    imgindex = anns_all[anns_all.name == image_name].index.tolist()
    # if len(imgindex)== 0:
    #     continue

    for i in range(len(imgindex)): 
        index = imgindex[i]
        bbox = anns_all.bbox[index]
        xmin, ymin, w, h = bbox
        defect_name = anns_all.defect_name[index]
        score = anns_all.score[index]
        img_save_path = mask_dir + image_name
        Font = ImageFont.truetype("/usr/share/fonts/truetype/ttf-dejavu/DejaVuSansCondensed-BoldOblique.ttf",20)
        draw.rectangle((xmin, ymin, xmin+w, ymin+h), outline=(255, 0, 0))
        draw.text((xmin, ymin), str(defect_name) + '|' + str(round(score,2)), font=Font, fill=(0, 255, 0))
        img.save(img_save_path)             

img_dir = "/mnt/hdd/tangwei/chongqing/mmdetection/code/data/jiuye/"
# anno_dir = "/mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/result.json"
# img_list = glob(img_dir + '*.jpg')

# txt2json('11.txt')
# anno_dir = '11.json'
# img_base_name = 'imgs_0150039'
# show_seqnms_object(img_base_name)
# show_test_object()
anno_dir = "/mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/imgs_0150039_0.json"
img_base_name = 'imgs_0150039'
img_name = img_base_name + '_0.jpg'
show_seqnms_object2(img_name, anno_dir)
    
