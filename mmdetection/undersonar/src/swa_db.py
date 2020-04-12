import torch
import os
import mmcv
from mmdet.models import build_detector
import argparse

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

    parser = argparse.ArgumentParser(description='test ')
    parser.add_argument('--config-dir', type=str, default=None,
                        help='model names')
    parser.add_argument('--work-dir', type=str,
                        default='/data/deeplearning/chongqing/chongqing1_round1_train1_20191223/instances_test2017.json',
                        help='training data directory')
    args = parser.parse_args()

    # 注意，此py文件没有更新batchnorm层，所以只有在mmdetection默认冻住BN情况下使用，如果训练时BN层被解冻，不应该使用此py
    # 逻辑上会score会高一点不会太多，需要指定的参数是 [config_dir, epoch_indices, alpha]
    epoch_indices = [10, 11, 12]
    # alpha = 0.7
    alpha = 0.5

    config = mmcv.Config.fromfile(args.config_dir)
    work_dir = args.work_dir
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
