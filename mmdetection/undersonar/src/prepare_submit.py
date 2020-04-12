# 此脚本用于生成mmdetection测试的json文件
import json
import os
from PIL import Image
import argparse
from tqdm import tqdm

if 0:
    modes = ['side','front']
    for mode in modes:
        savetestjsonpath = '/data/deeplearning/heqiongshuixia/sonar/a-test-image/image/instances_{}testA2017.json'.format(mode)
        imgroot = '/data/deeplearning/heqiongshuixia/sonar/a-test-image/image/{}'.format(mode)

        testpath = imgroot

        imagename = []
        for filename in os.listdir(testpath):
            imagename.append(filename)
        imagedict = {}
        for name in imagename:
            path = os.path.join(testpath,name)
            img = Image.open(path)
            imagedict[name] = img.size

        images_list = []
        images_name_id_dic = {}
        for i in tqdm(range(len(imagename))):
            temp_dict =  {'coco_url': '',
            'data_captured': '',
            'file_name': None,
            'flickr_url': '',
            'id': None,
            'height': None,
            'width': None,
            'license': 1}
            images_name_id_dic[imagename[i]]=i
            temp_dict["file_name"] = str(imagename[i])
            temp_dict["id"] = i
            temp_dict["width"] = imagedict[imagename[i]][0]
            temp_dict["height"] = imagedict[imagename[i]][1]
            images_list.append(temp_dict)

        categories_list = []
        classes_name = ['target']
        for i in range(len(classes_name)):
            temp_dict =  {
            "id": None,
            "name": None,
            "supercategory": None,}
            temp_dict["id"] = i+1
            temp_dict["name"] = classes_name[i]
            categories_list.append(temp_dict)

        test_dict = {}
        test_dict["info"] = None
        test_dict["licenses"] = None
        test_dict["categories"] = categories_list
        test_dict["images"] = images_list

        with open(savetestjsonpath,"w") as f:
            json.dump(test_dict,f)

if 1:
    modes = ['all']
    for mode in modes:
        # savetestjsonpath = '/data/deeplearning/heqiongshuixia/sonar/a-test-image/image/instances_{}testA2017.json'.format(mode)
        # imgroot = '/data/deeplearning/heqiongshuixia/sonar/a-test-image/image/{}'.format(mode)
        savetestjsonpath = "./coco/annotations/instances_testB_{}.json".format(mode)
        imgroot = "./data/image/all/"

        testpath = imgroot

        imagename = []
        for filename in os.listdir(testpath):
            imagename.append(filename)
        imagedict = {}
        for name in imagename:
            path = os.path.join(testpath,name)
            img = Image.open(path)
            imagedict[name] = img.size

        images_list = []
        images_name_id_dic = {}
        for i in tqdm(range(len(imagename))):
            temp_dict =  {'coco_url': '',
            'data_captured': '',
            'file_name': None,
            'flickr_url': '',
            'id': None,
            'height': None,
            'width': None,
            'license': 1}
            images_name_id_dic[imagename[i]]=i
            temp_dict["file_name"] = str(imagename[i])
            temp_dict["id"] = i
            temp_dict["width"] = imagedict[imagename[i]][0]
            temp_dict["height"] = imagedict[imagename[i]][1]
            images_list.append(temp_dict)

        categories_list = []
        classes_name = ['target']
        for i in range(len(classes_name)):
            temp_dict =  {
            "id": None,
            "name": None,
            "supercategory": None,}
            temp_dict["id"] = i+1
            temp_dict["name"] = classes_name[i]
            categories_list.append(temp_dict)

        test_dict = {}
        test_dict["info"] = None
        test_dict["licenses"] = None
        test_dict["categories"] = categories_list
        test_dict["images"] = images_list

        with open(savetestjsonpath,"w") as f:
            json.dump(test_dict,f)