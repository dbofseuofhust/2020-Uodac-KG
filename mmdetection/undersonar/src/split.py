import os
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
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

def create_train_val_df(img_list, train_val_index):
    index = []
    for img_name in img_list[train_val_index]:
        ind  = anno_df[anno_df.name == img_name].index.to_list()
        index.extend(ind)
    df = anno_df.loc[index]
    return df

def savejson(df, filename):
    result=[]
    for i in tqdm(range(df.shape[0])):
        name = df.name.iloc[i]
        defect_name = df.defect_name.iloc[i]
        bbox = df.bbox.iloc[i]
        result.append({'name': name,'defect_name': defect_name,'bbox':bbox})

    with open(filename, 'w') as fp:
        json.dump(result, fp, indent=4, separators=(',', ': '), cls=NpEncoder)
    print('over!')

def create_train_val_mirror_df(img_list, train_val_index, mode):
    anno_df = pd.read_json("./json/anno_ce/anno_ce_{}.json".format(mode))
    index = []
    for name in img_list[train_val_index]:
        img_name = name[:-4] + '_' + mode + 'mirror.jpg'
        ind  = anno_df[anno_df.name == img_name].index.to_list()
        index.extend(ind)
    df = anno_df.loc[index]
    return df

def create_train_val_half_df(img_list, train_val_index, mode):
    anno_df = pd.read_json("./json/anno_ce/anno_ce_half_{}.json".format(mode))
    index = []
    for name in img_list[train_val_index]:
        img_name = name[:-4] + '_' + mode + 'half.jpg'
        ind  = anno_df[anno_df.name == img_name].index.to_list()
        index.extend(ind)
    df = anno_df.loc[index]
    return df

json_dir = 'json/anno_ce.json'
anno_df = pd.read_json(json_dir)
img_list = anno_df.name.unique()

X = img_list
y = np.ones(img_list.shape)

floder = KFold(n_splits=5,random_state=0,shuffle=False)
train4, val1 = [], []
for train, test in floder.split(X,y):
    train4.append(train)
    val1.append(test)

df = create_train_val_df(img_list, train4[0])
savejson(df, 'json/anno_ce/anno_train_fold0.json')


# df = create_train_val_mirror_df(img_list, train4[0], 'left')
# savejson(df, 'json/anno_ce/anno_train_fold0_left.json')

# df = create_train_val_mirror_df(img_list, train4[0], 'right')
# savejson(df, 'json/anno_ce/anno_train_fold0_right.json')

df = create_train_val_half_df(img_list, train4[0], 'left')
savejson(df, 'json/anno_ce/anno_train_fold0_half_left.json')

df = create_train_val_half_df(img_list, train4[0], 'right')
savejson(df, 'json/anno_ce/anno_train_fold0_half_right.json')

# df = create_train_val_df(img_list, train4[1])
# savejson(df, 'json/anno/anno_train_fold1.json')

# df = create_train_val_df(img_list, train4[2])
# savejson(df, 'json/anno/anno_train_fold2.json')

# df = create_train_val_df(img_list, train4[3])
# savejson(df, 'json/anno/anno_train_fold3.json')

# df = create_train_val_df(img_list, train4[4])
# savejson(df, 'json/anno/anno_train_fold4.json')



# df = create_train_val_df(img_list, val1[0])
# savejson(df, 'json/anno_ce/anno_val_fold0.json')

# df = create_train_val_mirror_df(img_list,val1[0], 'left')
# savejson(df, 'json/anno_ce/anno_val_fold0_left.json')

# df = create_train_val_mirror_df(img_list,val1[0], 'right')
# savejson(df, 'json/anno_ce/anno_val_fold0_right.json')


# df = create_train_val_df(img_list, val1[1])
# savejson(df, 'json/anno/anno_val_fold1.json')

# df = create_train_val_df(img_list, val1[2])
# savejson(df, 'json/anno/anno_val_fold2.json')

# df = create_train_val_df(img_list, val1[3])
# savejson(df, 'json/anno/anno_val_fold3.json')

# df = create_train_val_df(img_list, val1[4])
# savejson(df, 'json/anno/anno_val_fold4.json')