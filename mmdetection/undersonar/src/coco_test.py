import os
import cv2
import json
import numpy as np
import shutil
import mmcv 
from glob import glob 

class Fabric2COCO:

    def __init__(self, label_list=[],mode="train"):
        self.images = []
        self.categories = []
        self.img_id = 0
        self.mode =mode
        self.label_list = label_list
        # if not os.path.exists("coco/images/{}".format(self.mode)):
        #     os.makedirs("coco/images/{}".format(self.mode))

    def to_coco(self, img_dir):
        self._init_categories()

        name_list = os.listdir(img_dir)
        # name_list = glob(img_dir + '*/*.jpg')
        prog_bar = mmcv.ProgressBar(len(name_list))
        for img_name in name_list:
            prog_bar.update()
            img_path = os.path.join(img_dir,img_name)
            # img_path = img_name
            img = cv2.imread(img_path)
            h,w,c = img.shape
            self.images.append(self._image(img_path,h, w))
            self.img_id += 1

        instance = {}
        instance['info'] = 'fabric defect'
        instance['license'] = ['none']
        instance['images'] = self.images
        instance['categories'] = self.categories
        return instance

    def _init_categories(self):
        for i, v in enumerate(self.label_list):  
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


    def save_coco_json(self, instance, save_path):
        import json
        with open(save_path, 'w') as fp:
            json.dump(instance, fp, indent=1, separators=(',', ': '))

def main(img_dir, label_list, mode):
	fabric2coco = Fabric2COCO(label_list=label_list,mode=mode)
	train_instance = fabric2coco.to_coco(img_dir)
	if not os.path.exists("coco/annotations/"):
	    os.makedirs("coco/annotations/")
	fabric2coco.save_coco_json(train_instance, "coco/annotations/"+'instances_{}.json'.format(mode))
	print('over!')


if __name__ == '__main__':

    label_list = ['target']

    # img_dir = "/mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/undersonar/data/a-test-image/image/all/"
    # mode = 'testA_all'
    # main(img_dir,  label_list, mode)   

    img_dir = "./data/image/all/"
    mode = 'testB_all'
    main(img_dir,  label_list, mode)                     