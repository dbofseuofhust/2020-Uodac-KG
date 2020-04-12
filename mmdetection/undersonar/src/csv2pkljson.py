import os 
import json
import csv
from tqdm import tqdm


def main(path, test_json_path, save_test_json_path):
    name2id = {}

    testjson = json.load(open(test_json_path,'r'))
    iamges = testjson["images"]
    for v in iamges:
        name2id['{}.xml'.format(v['file_name'].split('.')[0])] = v['id']

    outputs = []
    with open(path,'r') as f:
        reader = csv.reader(f)
        results = list(reader)
        for line in tqdm(results[1:]):
            # {"image_id": 0, "bbox": [574.6636962890625, 2.546943187713623, 85.7412109375, 262.4199147224426], "score": 0.7574860453605652, "category_id": 1}
            name,image_id,confidence,xmin,ymin,xmax,ymax = line
            temp = {}
            temp["image_id"] = name2id[image_id]
            temp["bbox"] = [float(xmin),float(ymin),float(xmax)-float(xmin),float(ymax)-float(ymin)]
            temp["score"] = float(confidence)
            temp["category_id"] = 1
            outputs.append(temp)

    with open(save_test_json_path,"w") as f:
        json.dump(outputs,f)

if __name__ == '__main__':
    path = 'r101.csv'
    test_json_path = "./coco/annotations/instances_testB_all.json"
    save_test_json_path = 'json/r101.pkl.bbox.json'
    main(path, test_json_path, save_test_json_path)

    path = 'r50_gc.csv'
    test_json_path = "./coco/annotations/instances_testB_all.json"
    save_test_json_path = 'json/r50_gc.pkl.bbox.json'
    main(path, test_json_path, save_test_json_path)    

    path = 'r50_db.csv'
    test_json_path = "./coco/annotations/instances_testB_all.json"
    save_test_json_path = 'json/r50_db.pkl.bbox.json'
    main(path, test_json_path, save_test_json_path)       

    path = 'r50_speckle.csv'
    test_json_path = "./coco/annotations/instances_testB_all.json"
    save_test_json_path = 'json/r50_speckle.pkl.bbox.json'
    main(path, test_json_path, save_test_json_path)           