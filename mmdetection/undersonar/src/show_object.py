import os
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def show_bbox(img_dir, anno_dir, mask_dir):

    img_list = glob(img_dir + '*.jpg')
    anns_all = pd.read_json(anno_dir)
    mkdir(mask_dir)

    for j, img_path in enumerate(tqdm(img_list)):
        image_name = os.path.basename(img_path)
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
            img_save_path = mask_dir + image_name
            Font = ImageFont.truetype("/usr/share/fonts/truetype/ttf-dejavu/DejaVuSansCondensed-BoldOblique.ttf", size=20)
            draw.rectangle((xmin, ymin, xmin+w, ymin+h), outline=(255, 0, 0))
            draw.text((xmin, ymin), str(defect_name), font=Font, fill=(0, 255, 0))
        img.save(img_save_path)    
        if j > 10 :
            break

if __name__ == '__main__':
    # img_dir = "/mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/undersonar/data/train/qian/qian_hflip/"
    # anno_dir = "/mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/undersonar/json/anno_qian_hflip.json"
    # mask_dir = "/mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/undersonar/data/train/qian/qian_hflip_mask/"
    # show_bbox(img_dir, anno_dir, mask_dir)

    # img_dir = "/mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/undersonar/data/train/ce/half_right/"
    # anno_dir = "/mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/undersonar/json/anno_ce/anno_ce_half_right.json"
    # mask_dir = "/mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/undersonar/data/train/ce/half_right_mask/"
    # show_bbox(img_dir, anno_dir, mask_dir)    

    img_dir = "./data/train/ce/right_fu/"
    anno_dir = "./json/anno_ce/anno_ce_right_fu.json"
    mask_dir = "./data/train/ce/half_right_fu_mask/"
    show_bbox(img_dir, anno_dir, mask_dir)   

    img_dir = "./data/train/ce/left_fu/"
    anno_dir = "./json/anno_ce/anno_ce_left_fu.json"
    mask_dir = "./data/train/ce/half_left_fu_mask/"
    show_bbox(img_dir, anno_dir, mask_dir)      

  