import os
import json
import time
import argparse
import pandas as pd
import numpy as np
# from tqdm import tqdm
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

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('json_dir', help='mmdet test json_out file path')
    parser.add_argument('--mode', help='mmdet test json_out file path')
    args = parser.parse_args()

    return args

def main(args):
    
    mmdet_res_df = pd.read_json(args.json_dir)
    mode = args.mode
    test_coco_json_file = './coco/annotations/instances_testA_{}.json'.format(mode)
    # test_coco_json_file = './coco/annotations/instances_testA_11.json'
    with open(test_coco_json_file) as f:
        data = json.load(f)
    image_df = pd.DataFrame(data['images'])

    print('\nConverting images info ...')
    images = []
    img_id_list = mmdet_res_df.image_id.unique()
    prog_bar = mmcv.ProgressBar(len(img_id_list))
    for i in img_id_list:
        temp_df = image_df[image_df.id == i]
        file_name = temp_df.file_name.iloc[0]
        image_id = temp_df.id.iloc[0]
        if mode == 'large':
            image_id = image_id + 1232
        images.append({'file_name':file_name, 'id':image_id})
        prog_bar.update()

    print('\nConverting annotations info ...')
    annotations = []
    prog_bar = mmcv.ProgressBar(mmdet_res_df.shape[0])
    for i in range(mmdet_res_df.shape[0]):
        prog_bar.update()
        image_id = mmdet_res_df['image_id'].iloc[i]
        bbox = mmdet_res_df['bbox'].iloc[i]
        score = mmdet_res_df['score'].iloc[i]
        # if score < 0.01:
        #     continue
        category_id = mmdet_res_df['category_id'].iloc[i]
        df = image_df[image_df.id == image_id]
        file_name = df.file_name.iloc[0]
        # if mode == 'small' and category_id == 11:
        #     continue
        if mode == 'large' and file_name[3] != 's' and category_id == 11: 
            # pingshen
            continue
        if mode == 'large':
            image_id = image_id + 1232
        annotations.append({'image_id': image_id, 'bbox':bbox, 'category_id': category_id, 'score':float(score)})
        
        
    print('\nGenerating final file ...')
    # file_name = 'result_{}.json'.format(mode)
    file_name = 'result.json'
    predictions = {"images":images, "annotations":annotations}
    with open(file_name, 'w') as fp:
        json.dump(predictions, fp, indent = 4, separators=(',', ': '), cls=NpEncoder)
        # json.dump(predictions, fp, cls=NpEncoder)
    print('\nConvert sucessfully.')  

if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()
    main(args)
    end_time = time.time()
    print('\nSubmit cost time: {} s \n'.format(end_time - start_time))


