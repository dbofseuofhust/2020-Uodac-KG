import argparse
import os, json
from PIL import Image, ImageDraw
from tqdm import tqdm
import pandas as pd
import numpy as np

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

def savejson(df, filename):
    result=[]
    for i in tqdm(range(df.shape[0])):
        name = df.name.iloc[i]
        defect_name = df.defect_name.iloc[i]
        xmin = df.bbox.iloc[i][0]
        ymin = df.bbox.iloc[i][1]
        w = df.bbox.iloc[i][2]
        h = df.bbox.iloc[i][3]
        bbox = [xmin, ymin, w, h]
        result.append({'name': name,'defect_name': defect_name,'bbox':bbox})
    #print(result)

    with open(filename, 'w') as fp:
        json.dump(result, fp, indent=4, separators=(',', ': '), cls=NpEncoder)
    print('over!')

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)    

def vertical_flip(anns_df, mode, json_name, org_img_dir, save_base_path):
    print("**************************************")
    print("Vertical flip...")

    save_dir = save_base_path + mode
    mkdirs(save_dir)

    print("**************************************")
    print("Save the pic...")

    for name in tqdm(anns_df.name):
        if not os.path.exists(os.path.join(save_dir, name)):
            img = Image.open(os.path.join(org_img_dir, name)).convert('RGB')
            width, height = img.size
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img.save(os.path.join(save_dir, name))
    print("Save Successful.")

    bbox_list = np.array(anns_df.bbox)
    print("**************************************")
    print("Save the json...")
    for i in tqdm(range(anns_df.shape[0])):
        name = anns_df.name.iloc[i]
        img = Image.open(os.path.join(org_img_dir, name))
        width, height = img.size
        ymin = bbox_list[i][1]
        ymax = ymin + bbox_list[i][3]
        bbox_list[i][1] = height - ymax
    anns_df.bbox = bbox_list
    savejson(anns_df, json_name)
    print("Save Successful.")


def horizontal_flip(anns_df, mode, json_name, org_img_dir, save_base_path):
    print("**************************************")
    print("Horizontal flip...")

    save_dir = save_base_path + mode
    mkdirs(save_dir)

    print("**************************************")
    print("Save the pic...")

    for name in tqdm(anns_df.name):
        if not os.path.exists(os.path.join(save_dir, name)):
            img = Image.open(os.path.join(org_img_dir, name)).convert('RGB')
            width, height = img.size
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img.save(os.path.join(save_dir, name))
            
    print("Save Successful.")
    bbox_list = np.array(anns_df.bbox)
    for i in tqdm(range(anns_df.shape[0])):
        name = anns_df.name.iloc[i]
        img = Image.open(os.path.join(org_img_dir, name))
        width, height = img.size
        xmin = bbox_list[i][0]
        xmax = xmin + bbox_list[i][2]
        bbox_list[i][0] = width - xmax	# width - xmax
    anns_df.bbox = bbox_list
    savejson(anns_df, json_name)

def rotate_180(anns_df):
    print("**************************************")
    print("Rotate 180...")

    # save_dir = path + 'defect_Images_r180'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    # print("**************************************")
    # print("Save the pic...")
    # for name in tqdm(anns_df.name):
    #     img = Image.open(os.path.join(base_dir, name))
    #     width, height = img.size
    #     img = img.transpose(Image.ROTATE_180)
    #     if not os.path.exists(os.path.join(save_dir, name)):
    #         img.save(os.path.join(save_dir, name))
    # print("Save Successful.")

    bbox_list = np.array(anns_df.bbox)
    for i in tqdm(range(anns_df.shape[0])):
        name = anns_df.name.iloc[i]
        img = Image.open(os.path.join(base_dir, name))
        width, height = img.size
        xmin = bbox_list[i][0]
        xmax = bbox_list[i][2]
        ymin = bbox_list[i][1]
        ymax = bbox_list[i][3]
        w = xmax - xmin
        h = ymax - ymin
        bbox_list[i][0] = round(width - xmax, 2)  # x1'=w-x2
        bbox_list[i][1] = round(height - ymax, 2) # y1'=h-y2
        bbox_list[i][2] = round(width - xmax + w, 2)  # x2'=w-x1
        bbox_list[i][3] = round(height - ymax + h, 2) # y2'=h-y1
    anns_df.bbox = bbox_list
    filename = "/home/tw/GD/Code/ks/mmdetection/baseline/json/round2/original/train_rotate_180_p1.json"
    savejson(anns_df, filename)

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('json_dir', help='mmdet test json_out file path')
    parser.add_argument('mode', help='mmdet test json_out file path')
    args = parser.parse_args()

    return args

# def main():
#     args = parse_args()
#     anns_df = pd.read_json(args.json_dir)
#     org_img_dir = "./data/train/qian/image/"
#     save_base_path = './data/train/qian/'
#     json_name = "./json/anno_qian_vflip.json"
#     mode = args.mode
#     vertical_flip(anns_df, mode, json_name, org_img_dir, save_base_path)  

def main():
    args = parse_args()
    anns_df = pd.read_json(args.json_dir)
    org_img_dir = "./data/train/qian/image/"
    save_base_path = './data/train/qian/'
    json_name = "./json/anno_qian_hflip.json"
    mode = args.mode
    horizontal_flip(anns_df, mode, json_name, org_img_dir, save_base_path) 

if __name__ == '__main__':
    main()