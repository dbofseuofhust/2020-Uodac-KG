import os
import json
import numpy as np

SOFT_NMS_ENABLED = True
BBOX_VOTE_ENABLED = True

paths = [
    'json/r50_speckle.pkl.bbox.json',
    'json/r50_gc.pkl.bbox.json',
    'json/r50_db.pkl.bbox.json',
    'json/r101.pkl.bbox.json'
]
save_file = 'json/testB_ensemble.pkl.bbox.json'

iou_thr = 0.7
isSoft = True

def load_dets(det_fname):
    with open(det_fname,'r',encoding='utf-8') as fid:
        det_boxes = json.load(fid)
    results = {}
    for det in det_boxes:
        ID = det['image_id']
        if ID not in results.keys():
            results[ID] = {'ID':ID,\
                            'scores':[], \
                            'category_ids':[],\
                            'bboxes':[]
                        }
        results[ID]['scores'].append(det['score'])
        results[ID]['category_ids'].append(det['category_id'])
        results[ID]['bboxes'].append(det['bbox'])

    return results

def bbox_vote(det,iou_thr=0.75):
    if det.shape[0] <= 1:
        return det
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    # det = det[np.where(det[:, 4] > 0.2)[0], :]
    dets = []
    while det.shape[0] > 0:
        # if det.shape[0]
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these  det
        merge_index = np.where(o >= iou_thr)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        else:
            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
            det_accu_sum[:, 4] = max_score
            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum
    return dets

def soft_bbox_vote(det,iou_thr=0.75):
    if det.shape[0] <= 1:
        return det
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    dets = []
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these  det
        merge_index = np.where(o >= iou_thr)[0]
        det_accu = det[merge_index, :]
        det_accu_iou = o[merge_index]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        else:
            soft_det_accu = det_accu.copy()
            soft_det_accu[:, 4] = soft_det_accu[:, 4] * (1 - det_accu_iou)
            soft_index = np.where(soft_det_accu[:, 4] >= 0.01)[0]
            soft_det_accu = soft_det_accu[soft_index, :]

            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
            det_accu_sum[:, 4] = max_score

            if soft_det_accu.shape[0] > 0:
                det_accu_sum = np.row_stack((soft_det_accu, det_accu_sum))

            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum

    order = dets[:, 4].ravel().argsort()[::-1]
    dets = dets[order, :]
    return dets

# det
num_model = len(paths)
# load dets
all_dets = []
for i in range(num_model):
    all_dets.append(load_dets(paths[i]))

merged_results = []

all_img_idx = []
for i in range(num_model):
    all_img_idx.extend(list(all_dets[i].keys()))
all_img_idx = sorted(list(set(all_img_idx)))
for img_idx in all_img_idx:
    if img_idx%1000 == 0:
        print('Merging index:',img_idx)
    category_list = []
    concat_list = []

    # load dets for a image:
    for model_idx in range(num_model):
        if img_idx in all_dets[model_idx].keys():
            _dets = np.array(all_dets[model_idx][img_idx]['bboxes'])
            _scores = np.array(all_dets[model_idx][img_idx]['scores'])
            _dets[:,2:4] +=_dets[:,:2]
            _dets = np.hstack([_dets,_scores.reshape(-1,1)]) # x,y,x,y,score
            category_list.extend(all_dets[model_idx][img_idx]['category_ids'])  
            if len(_dets) > 0:
                concat_list.append(_dets)  

    category_set = set(category_list)
    category_list = np.array((category_list))
    # det = np.concatenate(concat_list, axis=1)
    # det = det.transpose()
    det = np.vstack(concat_list)
    for index in category_set:

        index_res = np.argwhere(category_list == index)[: ,0]

        det_per_cls = det[index_res]
        if isSoft:
            dets = soft_bbox_vote(det_per_cls,iou_thr)
        else:
            dets = bbox_vote(det_per_cls,iou_thr)
        for i in range(len(dets)):
            image_dict = {}
            image_dict["image_id"] = img_idx
            bounding_box = np.concatenate((dets[i][:2], (dets[i][2:4] - dets[i][:2])))
            image_dict["bbox"] = bounding_box.tolist()
            image_dict["score"] = dets[i][-1]
            image_dict["category_id"] = index
            merged_results.append(image_dict)

with open(save_file, 'w') as file:
    file.write(json.dumps(merged_results))
