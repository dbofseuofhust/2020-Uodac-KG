import json
import time
import argparse
import numpy as np
from tqdm import tqdm

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
    parser.add_argument('--large_json', help='large json file path')
    parser.add_argument('--small_json', help='small json file path')
    args = parser.parse_args()

    return args

def merge_json(args):

    large_json_file = args.large_json
    with open(large_json_file) as f:
        large_data = json.load(f)

    small_json_file = args.small_json
    with open(small_json_file) as f:
        small_data = json.load(f)   
        
    images = []
    images.extend(small_data['images'])
    images.extend(large_data['images'])

    annotations = []
    annotations.extend(small_data['annotations'])
    annotations.extend(large_data['annotations'])

    print('Generating final file ...')
    file_name = 'result.json'
    predictions = {"images":images, "annotations":annotations}
    with open(file_name, 'w') as fp:
        json.dump(predictions, fp, indent = 4, separators=(',', ': '), cls=NpEncoder)

    print('Convert sucessfully.')

if __name__ == '__main__':
    args = parse_args()
    merge_json(args)

