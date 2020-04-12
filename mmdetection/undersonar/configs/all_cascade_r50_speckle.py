# fp16 settings
fp16 = dict(loss_scale=512.)
num_classes=2
import time
# model settings
model = dict(
    type='CascadeRCNN',
    num_stages=3,
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',     
        dcn=dict(
            type='DCN', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),    
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8], #<---------------------8
        anchor_ratios=[0.5, 1.0, 2.0, 4.0, 6.0, 8.0],  #<---------------------8     
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        # add_context=True, #<---------------add_context
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=[
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2],
            reg_class_agnostic=True,           
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.05, 0.05, 0.1, 0.1],
            reg_class_agnostic=True,          
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.033, 0.033, 0.067, 0.067],
            reg_class_agnostic=True,           
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
    ])
# model training and testing settings
iou_thr = [0.55, 0.65, 0.75]        
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7, #<---------------------
            neg_iou_thr=0.3, #<---------------------
            min_pos_iou=0.3, #<---------------------
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=[
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=iou_thr[0],
                neg_iou_thr=iou_thr[0],
                min_pos_iou=iou_thr[0],
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=iou_thr[1],
                neg_iou_thr=iou_thr[1],
                min_pos_iou=iou_thr[1],
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=iou_thr[2],
                neg_iou_thr=iou_thr[2],
                min_pos_iou=iou_thr[2],
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)
    ],
    stage_loss_weights=[1, 0.5, 0.25])
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.001, 
        nms=dict(type='nms', iou_thr=0.5), 
        max_per_img=80))
# dataset settings
dataset_type = 'UnderSonarDataset'
data_root = "./undersonar/data/"
anno_root = "./undersonar/coco/annotations/"

imgs_per_gpu = 2
lr = 0.00125*2*imgs_per_gpu

# coco image info
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# albu_train_transforms = [
#     dict(type='Equalize', always_apply=True), 
# ]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(2000, 300), (2000, 1200)],
        multiscale_mode='range',
        keep_ratio=True), 
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='SpeckleNoise', severity=1, ratio=0.2),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(2000, 300), (2000, 600), (2000, 900), (2000, 1200)],
        # img_scale=[(2000, 300), (2000, 600), (2000, 900)],
        # img_scale=[(4096, 800), (4096, 1200)],
        # img_scale=[(4096, 1200)],        
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=imgs_per_gpu,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=[
            anno_root + 'instances_train_ce.json',
            anno_root + 'instances_train_qian.json',
            # anno_root + 'instances_train_ce_fold0_left_fu.json',
            # anno_root + 'instances_train_ce_fold0_right_fu.json',            
        ],

        img_prefix=[
            data_root + 'train/ce/image/',
            data_root + 'train/qian/image/',
            # data_root + "train/ce/left_fu/",
            # data_root + "train/ce/right_fu/",            
        ],
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=anno_root + 'instances_testB_all.json',
        img_prefix=data_root + "/image/all/",
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './undersonar/work_dirs/all_cascade_r50_speckle/'     
# load_from = None         
load_from = "./checkpoints/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-dfa53166.pth"
resume_from = None 
workflow = [('train', 1)]

# baseline ------> 0.41578698 ------>train/test img_scale=(1500, 844) 
# del w<=10 and h <=10 ------> 0.41462205 ------>train/test img_scale=(1500, 844) 
# ColorJitter 0.41705803
# baseline 0.47766863 cr_dcn_r50_fpn_1x_20200310170344
# boxjitter 0.47703111

# "/mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/undersonar/work_dirs/all_cr_dcn_r50_fpn_1x_20200408001310/epoch_ensemble.pth"  0.29157740

# "/mnt/hdd/tangwei/chongqing/new_mmdet/mmdetection/undersonar/work_dirs/all_cr_dcn_r50_fpn_1x_20200408165845/epoch_ensemble.pth" 0.2913