import torch
import os
import mmcv
from mmdet.models import build_detector

def get_model(config, model_dir):
    model = build_detector(config.model, test_cfg=config.test_cfg)
    checkpoint = torch.load(model_dir)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=True)
    return model


def model_average(modelA, modelB, alpha):
    # modelB占比 alpha
    for A_param, B_param in zip(modelA.parameters(), modelB.parameters()):
        A_param.data = A_param.data * (1 - alpha) + alpha * B_param.data
    return modelA

if __name__ == "__main__":
    config_dir = "/mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/undersonar/configs/all_cascade_r50.py"
    epoch_indices = [10, 11, 12]
    alpha = 0.5
    
    config = mmcv.Config.fromfile(config_dir)
    work_dir ="/mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/undersonar/work_dirs/all_cr_dcn_r50_fpn_1x_20200410030400/"
    model_dir_list = [os.path.join(work_dir, 'epoch_{}.pth'.format(epoch)) for epoch in epoch_indices]

    model_ensemble = None
    for model_dir in model_dir_list:
        if model_ensemble is None:
            model_ensemble = get_model(config, model_dir)
        else:
            model_fusion = get_model(config, model_dir)
            model_emsemble = model_average(model_ensemble, model_fusion, alpha)

    checkpoint = torch.load(model_dir_list[-1])
    checkpoint['state_dict'] = model_ensemble.state_dict()
    save_dir = os.path.join(work_dir, 'epoch_ensemble.pth')
    torch.save(checkpoint, save_dir)