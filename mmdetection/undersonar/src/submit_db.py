import json
import argparse
import pandas as pd 
from tqdm import tqdm

parser = argparse.ArgumentParser(description='test ')
parser.add_argument('--jsonpath', type=str, default=None,
                    help='model names')
parser.add_argument('--testpath', type=str, default='/data/deeplearning/chongqing/chongqing1_round1_train1_20191223/instances_test2017.json',
                    help='training data directory')
parser.add_argument('--savepath', type=str, default=None,
                    help='whether to combine shanghaiTech and UCF dataset.')
args = parser.parse_args()

results = json.load(open(args.jsonpath,'r'))
testjson = json.load(open(args.testpath,'r'))

id2imginfo = {}
for img_info in testjson["images"]:
    id2imginfo[img_info['id']] = img_info

classes_dict = {1:'target'}

columns = ['name','image_id','confidence','xmin','ymin','xmax','ymax']
outputs = {col:[] for col in columns}

for u in tqdm(results):
    img_id = u['image_id']
    score = u['score']
    x1, y1, w, h = u['bbox']
    cat = classes_dict[u['category_id']]
    xmin, ymin, xmax, ymax = x1, y1, x1+w, y1+h
    filename = '{}.xml'.format(id2imginfo[img_id]['file_name'].split('.')[0])
    outputs['name'].append(cat)
    outputs['image_id'].append(filename)
    outputs['confidence'].append(score)
    outputs['xmin'].append(xmin)
    outputs['ymin'].append(ymin)
    outputs['xmax'].append(xmax)
    outputs['ymax'].append(ymax)

pd.DataFrame(outputs).to_csv(args.savepath,index=False,columns=columns)
