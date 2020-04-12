import json

def check_json_with_error_msg(pred_json, num_classes=10):
    '''
    Args:
        pred_json (str): Json path
        num_classes (int): number of foreground categories
    Returns:
        Message (str)
    Example:
        >>> msg = check_json_with_error_msg('./submittion.json')
        >>> print(msg)
        missing key "annotations"
    '''
    try:
        if not pred_json.endswith('.json'):
            return "the prediction file should ends with .json"
        with open(pred_json) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return "the prediction data should be a dict"
        if not 'images' in data:
            return "missing key \"images\""
        if not 'annotations' in data:
            return "missing key \"annotations\""
        images = data['images']
        annotations = data['annotations']
        if not isinstance(images, (list, tuple)):
            return "\"images\" format error"
        if not isinstance(annotations, (list, tuple)):
            return "\"annotations\" format error"
        for image in images:
            if not 'file_name' in image:
                return "missing key \"file_name\" in \"images\""
            if not 'id' in image:
                return "missing key \"id\" in \"images\""
            if not isinstance(image['id'], int ):
                return "\"id\" should be int"
        for annotation in annotations:
            if not 'image_id' in annotation:
                return "missing key \"image_id\" in \"annotations\""
            if not 'category_id' in annotation:
                return "missing key \"category_id\" in \"annotations\""
            if not 'bbox' in annotation:
                return "missing key \"bbox\" in \"annotations\""
            if not 'score' in annotation:
                return "missing key \"score\" in \"annotations\""
            if not isinstance(annotation['bbox'], (tuple, list)):
                return "bbox format error"
            if len(annotation['bbox'])==0:
                return "empty bbox"
            if len(annotation['bbox'])!=4:
                return "bbox length error"
            if not isinstance(annotation['image_id'], int):
                return "\"image_id\" shoud be int"
            if not isinstance(annotation['category_id'], int):
                return "\"category_id\" shoud be int"
            if not isinstance(annotation['score'], (int, float)):
                return "\"score\" shoud be int or float"
            if annotation['category_id'] > num_classes or annotation['category_id'] < 0:
                return "\"category_id\" out of range"
    except:
        return "Unknown error"
    return "No error found"
anno_dir = "/mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/result2.json"
res = check_json_with_error_msg(anno_dir, num_classes=11)
print(res)