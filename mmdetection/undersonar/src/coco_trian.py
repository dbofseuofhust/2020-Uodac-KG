import os
import cv2
import json
import numpy as np
import shutil
import pandas as pd
from tqdm import tqdm 
from glob import glob 

class Fabric2COCO:

    def __init__(self, label_list=[], mode="trainall"):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.mode =mode
        self.label_list = label_list
        # if not os.path.exists("coco/images/{}".format(self.mode)):
        #     os.makedirs("coco/images/{}".format(self.mode))

    def to_coco(self, anno_file,img_dir):
        self._init_categories()
        anno_result= pd.read_json(open(anno_file,"r"))
        name_list=anno_result["name"].unique()
        for img_name in tqdm(name_list):
            img_anno = anno_result[anno_result["name"] == img_name]
            bboxs = img_anno["bbox"].tolist()
            defect_names = img_anno["defect_name"].tolist()
            assert img_anno["name"].unique()[0] == img_name

            img_path = os.path.join(img_dir,img_name)
            # print(img_path)
            img = cv2.imread(img_path)
            h,w,c = img.shape
            # h,w=1000,2446
            self.images.append(self._image(img_path,h, w))

            # self._cp_img(img_path)  # do not copy image 

            for bbox, defect_name in zip(bboxs, defect_names):
                # label= defect_name2label[defect_name]
                label= defect_name
                annotation = self._annotation(label, bbox)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'fabric defect'
        instance['license'] = ['none']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    def _init_categories(self):

        for i, v in enumerate(self.label_list):  # must change the number equal to (number of classes + 1)
            # print(v)
            category = {}
            category['id'] = int(i)+1
            category['name'] = str(v)
            category['supercategory'] = 'defect_name'
            self.categories.append(category)            

    def _image(self, path,h,w):
        image = {}
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path)
        return image

    def _annotation(self,label,bbox):
        area=bbox[2]*bbox[3]
        points=[
                [bbox[0], bbox[1]], # x1, y1
                [bbox[0]+bbox[2], bbox[1]], # x2, y1
                [bbox[0]+bbox[2], bbox[1]+bbox[3]], # x2, y2
                [bbox[0], bbox[1]+bbox[3]]] # x1, y2
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = label_dict[label]
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = bbox
        annotation['iscrowd'] = 0
        annotation['area'] = area
        return annotation

    def _cp_img(self, img_path):
        shutil.copy(img_path, os.path.join("coco/images/{}".format(self.mode), os.path.basename(img_path)))

    def save_coco_json(self, instance, save_path):
        import json
        with open(save_path, 'w') as fp:
            json.dump(instance, fp, indent=1, separators=(',', ': '))

def main(anno_dir, img_dir, label_list, mode):
	fabric2coco = Fabric2COCO(label_list=label_list, mode=mode)
	train_instance = fabric2coco.to_coco(anno_dir,img_dir)
	if not os.path.exists("coco/annotations/"):
	    os.makedirs("coco/annotations/")
	fabric2coco.save_coco_json(train_instance, "coco/annotations/"+'instances_{}.json'.format(mode))
	print('over!')

# convert the txt to json 
def txt2json(txt_path):
    f = open(txt_path, 'r')
    res = []
    for line in f.readlines():
        line = line.strip()
        image_name = line.split(' ')[0]
        seg_name = line.split(' ')[1]
        for item in line.split(' ')[2:]:
            x1, y1, x2, y2, label, confidence = [int(x) for x in item.split(",")]
            res.append({'image_name':image_name, 'defect_name': str(label), 'bbox':[x1, y1, x2, y2]})
    f.close()
    json_name = txt_path.replace('txt', 'json')
    with open(json_name, 'w') as fp:
        json.dump(res, fp, indent=4, separators=(',', ': '))
    print('over!')

if __name__ == '__main__':

    label_list = ['target']
    label_dict = {'target':1}    
    anno_dir = "./json/anno_qian.json"
    img_dir = "./data/train/qian/image/"
    mode = 'train_qian'
    main(anno_dir, img_dir, label_list, mode)   

    anno_dir = "./json/anno_ce.json"
    img_dir = "./data/train/ce/image/"
    mode = 'train_ce'
    main(anno_dir, img_dir, label_list, mode)    


  

