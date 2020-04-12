#!/usr/bin/python
# -*- coding: UTF-8 -*-
# get annotation object bndbox location
import os
import cv2
try:
    import xml.etree.cElementTree as ET 
except ImportError:
    import xml.etree.ElementTree as ET
from glob import glob
from tqdm import tqdm
import pandas as pd 
import json

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
                                                     
def GetAnnotBoxLoc(AnotPath):#AnotPath VOC标注文件路径
    tree = ET.ElementTree(file=AnotPath)  #打开文件，解析成一棵树型结构
    root = tree.getroot()#获取树型结构的根
    ObjectSet=root.findall('object')#找到文件中所有含有object关键字的地方，这些地方含有标注目标
    ObjBndBoxSet={} #以目标类别为关键字，目标框为值组成的字典结构
    
    for Object in ObjectSet:
        ObjName=Object.find('name').text
        BndBox=Object.find('bndbox')
        x1 = int(BndBox.find('xmin').text)
        y1 = int(BndBox.find('ymin').text)
        x2 = int(BndBox.find('xmax').text)
        y2 = int(BndBox.find('ymax').text)

        BndBoxLoc=[x1,y1,x2-x1+1,y2-y1+1]
        if ObjName in ObjBndBoxSet:
            ObjBndBoxSet[ObjName].append(BndBoxLoc)#如果字典结构中含有这个类别了，那么这个目标框要追加到其值的末尾
        else:
            ObjBndBoxSet[ObjName]=[BndBoxLoc]#如果字典结构中没有这个类别，那么这个目标框就直接赋值给其值吧
    return ObjBndBoxSet

def display(objBox,pic):
    img = cv2.imread(pic)
    
    for key in objBox.keys():
        for i in range(len(objBox[key])):
            cv2.rectangle(img, (objBox[key][i][0],objBox[key][i][1]), (objBox[key][i][2], objBox[key][i][3]), (0, 0, 255), 2)        
            cv2.putText(img, key, (objBox[key][i][0],objBox[key][i][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)
    # cv2.imshow('img',img)
    cv2.imwrite('display.jpg',img)
    # cv2.waitKey(0)

def main(mode):
    xml_listdir = glob("./data/train/{}/box/*.xml".format(mode))
    res = []
    for xml in tqdm(xml_listdir):
        ObjBndBoxSet = GetAnnotBoxLoc(xml)
        if len(ObjBndBoxSet) == 0:
            continue
        name = os.path.basename(xml).replace('xml', 'jpg')

        for key in ObjBndBoxSet.keys():
            for i in range(len(ObjBndBoxSet[key])):
                bbox = ObjBndBoxSet[key][i]

                res.append({'name':name, 'bbox':bbox, 'defect_name':'target'})

                
    json_name = 'json/anno_{}.json'.format(mode)
    with open(json_name,'w') as fp:
        json.dump(res, fp, indent=4, separators=(',', ': '), cls=NpEncoder)

if __name__ == '__main__':
    mode = 'qian'
    main(mode)
    mode = 'ce'
    main(mode)
