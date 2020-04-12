import os
import json
import time
import argparse
import pandas as pd
import numpy as np
import mmcv

'''
name - 检测类别 - string
image_id - 图像编号 - string
confidence - 置信度 - float
xmin - 检测结果左上角x值 - int
ymin - 检测结果左上角y值 - int
xmax - 检测结果右下角x值 - int
ymax - 检测结果右下角y值 - int
'''

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
    parser.add_argument('mode', help='mmdet test json_out file path')
    args = parser.parse_args()

    return args


def main3(args):
    
    mmdet_res_df = pd.read_json(args.json_dir)
    print(args.json_dir)
    test_coco_json_file = './coco/annotations/instances_testB_{}.json'.format(args.mode)    
    with open(test_coco_json_file) as f:
        data = json.load(f)
    image_df = pd.DataFrame(data['images'])


    label_list = ['target']
    label_dict = {'target':1}
    id2cat = {1:'target'}

    print('\nConverting annotations info ...')
    annotations = []
    prog_bar = mmcv.ProgressBar(image_df.shape[0])
    for i in range(image_df.shape[0]):
        prog_bar.update()
        image_id = image_df['id'].iloc[i]
        file_name = image_df['file_name'].iloc[i]
        df = mmdet_res_df[mmdet_res_df.image_id == image_id]

        if df.shape[0] == 0:
            continue

        for j in range(df.shape[0]):           
            bbox = df['bbox'].iloc[j]
            score = df['score'].iloc[j]
            category_id = df['category_id'].iloc[j]

            # annotations.append({'name': id2cat[category_id], 'image_id':file_name.replace('.jpg', '.xml'), 
            #     'confidence': float(score), 'xmin':int(bbox[0]), 'ymin':int(bbox[1]),
            #     'xmax':int(bbox[0]+bbox[2]), 'ymax':int(bbox[1]+bbox[3]),

            #     })
            annotations.append({'name': id2cat[category_id], 'image_id':file_name.replace('.jpg', '.xml'), 
                'confidence': float(score), 'xmin':float(bbox[0]), 'ymin':float(bbox[1]),
                'xmax':float(bbox[0]+bbox[2]), 'ymax':float(bbox[1]+bbox[3]),

                })            
    res = pd.DataFrame(annotations)
    print(res)
        
    print('\nGenerating final file ...')
    # file_name = 'result2.csv'
    file_name = os.path.basename(args.json_dir).replace('.bbox.json', '.csv')
    res.to_csv(file_name, index=0)
    print('\nConvert sucessfully.')      

if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()
    main3(args)
    end_time = time.time()
    print('\nSubmit cost time: {} s \n'.format(end_time - start_time))


