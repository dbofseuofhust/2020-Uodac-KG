import os
import json 
from PIL import Image 
import numpy as np 
from glob import glob 
from tqdm import tqdm 
import pandas as pd 
import ipdb
import random

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


def img_mirror(img_path, mode):
    '''
    img: np.array
    '''

    img_array = np.array(Image.open(img_path).convert('RGB'))
    h, w, _ = img_array.shape 
    if mode == 'left':
        half_img_array_left = img_array[:, :w//2, :]
        new_img_array= np.concatenate((half_img_array_left, np.fliplr(half_img_array_left)), axis=1)
    elif mode == 'right':
        half_img_array_right = img_array[:, w//2:, :]   
        new_img_array = np.concatenate((np.fliplr(half_img_array_right), half_img_array_right), axis=1)

    return new_img_array

def img_half(img_path, mode):
    '''
    img: np.array
    '''

    img_array = np.array(Image.open(img_path).convert('RGB'))
    h, w, _ = img_array.shape 
    if mode == 'left':
        new_img_array = img_array[:, :w//2, :]
    elif mode == 'right':
        new_img_array = img_array[:, w//2:, :]

    return new_img_array

def bbox_mirror(bbox, img_array):
    width = img_array.shape[1]
    x, y, w, h = bbox
    x_new = width - w - x -1
    return [x_new, y, w, h]

def bbox_half(bbox, img_array):
    width = img_array.shape[1]
    x, y, w, h = bbox
    x_new = width - x - 1
    return [x_new, y, w, h]

def aug(img_listdir, mode, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for img_path in tqdm(img_listdir):
        img_name = os.path.basename(img_path)
        new_img_array = img_mirror(img_path, mode)
        img = Image.fromarray(new_img_array)
        img_mirrored_name = img_name[:-4] + '_' + mode + 'mirror.jpg'
        img.save(os.path.join(save_dir, img_mirrored_name))
        # break

def aug_half(img_listdir, mode, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for img_path in tqdm(img_listdir):
        img_name = os.path.basename(img_path)
        new_img_array = img_half(img_path, mode)
        img = Image.fromarray(new_img_array)
        img_half_name = img_name[:-4] + '_' + mode + 'half.jpg'
        img.save(os.path.join(save_dir, img_half_name))
        # break

def savejson(anno_dir, mode, img_listdir, res_dir):
    anno_df = pd.read_json(anno_dir)
    result=[]
    for img_path in tqdm(img_listdir):
        img_array = np.array(Image.open(img_path).convert('RGB'))
        img_name = os.path.basename(img_path)
        width = img_array.shape[1]

        df = anno_df[anno_df.name == img_name]

        if df.shape[0] == 0:
            continue

        for j in range(df.shape[0]):          
            bbox = df['bbox'].iloc[j]
            if mode == 'left' and  bbox[0] > width / 2:
                continue
            elif mode == 'right' and bbox[0]+bbox[2] < width / 2:
                continue

            flipped_img_name = img_name[:-4] + '_' + mode + 'mirror.jpg'
            result.append({'name': flipped_img_name, 'defect_name': 'target', 'bbox':bbox})  

            flipped_img_array = img_mirror(img_path, mode)
            bbox = bbox_mirror(bbox, flipped_img_array)
            result.append({'name': flipped_img_name, 'defect_name': 'target', 'bbox':bbox})   

    with open(res_dir, 'w') as fp:
        json.dump(result, fp, indent=4, separators=(',', ': '), cls=NpEncoder)
    print('over!')

def savejson_half(anno_dir, mode, img_listdir, res_dir):
    anno_df = pd.read_json(anno_dir)
    result=[]
    for img_path in tqdm(img_listdir):
        img_array = np.array(Image.open(img_path).convert('RGB'))
        img_name = os.path.basename(img_path)
        width = img_array.shape[1]

        df = anno_df[anno_df.name == img_name]

        if df.shape[0] == 0:
            continue

        for j in range(df.shape[0]):          
            bbox = df['bbox'].iloc[j]
            if mode == 'left':
                if bbox[0] > width / 2:
                    continue
                else:
                    flipped_img_name = img_name[:-4] + '_' + mode + 'half.jpg'
                    result.append({'name': flipped_img_name, 'defect_name': 'target', 'bbox':bbox})                 
            elif mode == 'right':
                if bbox[0]+bbox[2] < width / 2:
                    continue
                else:
                    flipped_img_name = img_name[:-4] + '_' + mode + 'half.jpg'
                    bbox[0] = bbox[0] - width/2
                    result.append({'name': flipped_img_name, 'defect_name': 'target', 'bbox':bbox})            

    with open(res_dir, 'w') as fp:
        json.dump(result, fp, indent=4, separators=(',', ': '), cls=NpEncoder)
    print('over!')



def savejson_fu(anno_dir, mode, img_listdir, res_dir):
    anno_df = pd.read_json(anno_dir)
    result=[]
    for img_path in tqdm(img_listdir):
        img_array = np.array(Image.open(img_path).convert('RGB'))
        img_name = os.path.basename(img_path)
        org_img_name = img_name.split('_')[0] + '_' + mode + 'half.jpg'

        width = img_array.shape[1]

        df = anno_df[anno_df.name == org_img_name]

        if df.shape[0] == 0:
            continue

        for j in range(df.shape[0]):          
            bbox = df['bbox'].iloc[j]
            if mode == 'left':
                result.append({'name': img_name, 'defect_name': 'target', 'bbox':bbox})                 
            elif mode == 'right':
                bbox[0] = bbox[0] + width/2
                result.append({'name': img_name, 'defect_name': 'target', 'bbox':bbox})            

    with open(res_dir, 'w') as fp:
        json.dump(result, fp, indent=4, separators=(',', ': '), cls=NpEncoder)
    print('over!')

# ipdb.set_trace()
def fu_aug(mode, save_dir, times=1):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)    

    img_dir = "./data/train/ce/half_{}/".format(mode)
    img_listdir = glob(img_dir + '*.jpg')
    ss_img_listdir = glob(img_dir + 'ss*.jpg')  
    gx_img_listdir = glob(img_dir + 'gx*.jpg')  
    flv_img_listdir = glob(img_dir + 'flv*.jpg')  

    if mode == 'left':
        fu_img_dir = "./data/train/fu/half_{}/".format('right')
    else:
        fu_img_dir = "./data/train/fu/half_{}/".format('left')
    fu_img_listdir = glob(fu_img_dir + '*.jpg')  
    fu_ss_img_listdir = glob(fu_img_dir + 'ss*.jpg')  
    fu_gx_img_listdir = glob(fu_img_dir + 'gx*.jpg')  
    fu_flv_img_listdir = glob(fu_img_dir + 'flv*.jpg')  

    for img_path in tqdm(img_listdir):
        img_name = os.path.basename(img_path)
        img = Image.open(img_path)
        # height, width = img.size
        if img_name[:2] == 'ss':
            fu_list = fu_ss_img_listdir
        elif img_name[:2] == 'gx':
            fu_list = fu_gx_img_listdir
        else:
            fu_list = fu_flv_img_listdir

        random_img_listdirs = random.sample(fu_list, times)
        for fu_img_path in random_img_listdirs:

            # print(random_img)
            fu_img_name = os.path.basename(fu_img_path)
            fu_img = Image.open(fu_img_path)
            # fu_height, fu_width = fu_img.size

            if fu_img.size != img.size:
                fu_img = fu_img.resize(img.size)

            if mode =='left':
                new_img_array = np.concatenate((np.array(img), np.array(fu_img)), axis=1)
            else:
                new_img_array = np.concatenate((np.array(fu_img), np.array(img)), axis=1)

            new_img = Image.fromarray(new_img_array)
            new_img_name = img_name[:-4] + '_' + 'fu_' + fu_img_name
            new_img.save(os.path.join(save_dir, new_img_name))
        # break

# mode = 'left'
# save_dir = "./data/train/ce/left_fu/"
# fu_aug(mode, save_dir)

# mode = 'right'
# save_dir = "./data/train/ce/right_fu2/"
# fu_aug(mode, save_dir)






            


anno_dir = "./json/anno_ce/anno_train_fold0_half_left.json"
mode = 'left'
img_dir = "./data/train/ce/left_fu/"
img_listdir = glob(img_dir + '*.jpg')
res_dir = "./json/anno_ce/anno_ce_left_fu.json"
savejson_fu(anno_dir, mode, img_listdir, res_dir)



anno_dir = "./json/anno_ce/anno_train_fold0_half_right.json"
mode = 'right'
img_dir = "./data/train/ce/right_fu/"
img_listdir = glob(img_dir + '*.jpg')
res_dir = "./json/anno_ce/anno_ce_right_fu.json"
savejson_fu(anno_dir, mode, img_listdir, res_dir)




'''

if __name__ == "__main__":
    img_dir = "./data/train/ce/image/"
    img_listdir = glob(img_dir + '*.jpg')

    fu_img_dir = "./data/train/fu/image/"
    fu_img_listdir = glob(img_dir + '*.jpg')    

    # mode = 'right'
    # save_dir = "./data/train/ce/right/"
    # aug(img_listdir, mode, save_dir)
    # anno_dir = "./json/anno_ce.json"
    # res_dir = "./json/anno_ce/anno_ce_right.json"
    # savejson(anno_dir, mode, img_listdir, res_dir)

    # mode = 'left'
    # save_dir = "./data/train/ce/left/"
    # aug(img_listdir, mode, save_dir)
    # anno_dir = "./json/anno_ce.json"
    # res_dir = "./json/anno_ce/anno_ce_left.json"
    # savejson(anno_dir, mode, img_listdir, res_dir)

    # mode = 'left'
    # save_dir = "./data/train/ce/half_left/"
    # aug_half(img_listdir, mode, save_dir)
    # anno_dir = "./json/anno_ce.json"
    # res_dir = "./json/anno_ce/anno_ce_half_left.json"
    # savejson_half(anno_dir, mode, img_listdir, res_dir)    

    # mode = 'right'
    # save_dir = "./data/train/ce/half_right/"
    # # aug_half(img_listdir, mode, save_dir)
    # anno_dir = "./json/anno_ce.json"
    # res_dir = "./json/anno_ce/anno_ce_half_right.json"
    # savejson_half(anno_dir, mode, img_listdir, res_dir)    

    mode = 'left'
    save_dir = "./data/train/fu/half_left/"
    aug_half(fu_img_listdir, mode, save_dir)
    # anno_dir = "./json/anno_ce.json"
    # res_dir = "./json/anno_ce/anno_ce_half_left.json"
    # savejson_half(anno_dir, mode, img_listdir, res_dir)    

    mode = 'right'
    save_dir = "./data/train/fu/half_right/"
    aug_half(fu_img_listdir, mode, save_dir)
    # anno_dir = "./json/anno_ce.json"
    # res_dir = "./json/anno_ce/anno_ce_half_right.json"
    savejson_half(anno_dir, mode, img_listdir, res_dir)        
'''