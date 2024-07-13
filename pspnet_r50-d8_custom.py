log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)
custom_hooks = []
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

checkpoint_config = dict(
    by_epoch=False,
    interval=4000,
    meta=dict(
        config='configs/custom/pspnet_r50-d8_custom.py',
        mmseg_version='1.2.2'
    )
)

data_root = 'data/escasymptoms/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(512, 512), pad_val=0),
    dict(type='ImageToTensor', keys=['img', 'gt_semantic_seg']),
    dict(type='PackSegInputs'),
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        scales=[(2048, 512)],
        transforms=[
            dict(type='Resize', keep_ratio=True, scale=(2048, 512)),
            dict(type='RandomFlip', prob=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='PackSegInputs'),
        ]
    )
]


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CustomCocoDataset',
        data_root=data_root,
        data_prefix=dict(
            img_path='images',
            seg_map_path='labels'
        ),
        ann_file='coco/coco_annotations.json',
        pipeline=train_pipeline
    ),
    val=dict(
        type='CustomCocoDataset',
        data_root=data_root,
        data_prefix=dict(
            img_path='images',
            seg_map_path='labels'
        ),
        ann_file='coco/coco_annotations.json',
        pipeline=test_pipeline
    ),
    test=dict(
        type='CustomCocoDataset',
        data_root=data_root,
        data_prefix=dict(
            img_path='images',
            seg_map_path='labels'
        ),
        ann_file='coco/coco_annotations.json',
        pipeline=test_pipeline
    )
)

evaluation = dict(interval=1, metric='mIoU')
work_dir = 'work_dirs/pspnet_r50-d8_custom'


model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnet50_v1c')
    ),
    decode_head=dict(
        type='PSPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
    )
)


optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)
optimizer_config = dict()

optim_wrapper = dict(
    optimizer=optimizer,
    clip_grad=None,
    type='AmpOptimWrapper'
)

lr_config = dict(
    by_epoch=False,
    min_lr=0.0,
    policy='poly',
    power=1.0,
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001
)

runner = dict(type='IterBasedRunner', max_iters=40000)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=1000
    ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1000,
        end=40000,
        by_epoch=False
    )
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='CustomCocoDataset',
        data_root=data_root,
        ann_file='coco/coco_annotations.json',
        data_prefix=dict(img_path='images', seg_map_path='labels'),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler'),
    dataset=dict(
        type='CustomCocoDataset',
        data_root=data_root,
        ann_file='coco/coco_annotations.json',
        data_prefix=dict(img_path='images', seg_map_path='labels'),
        pipeline=test_pipeline
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler'),
    dataset=dict(
        type='CustomCocoDataset',
        data_root=data_root,
        ann_file='coco/coco_annotations.json',
        data_prefix=dict(img_path='images', seg_map_path='labels'),
        pipeline=test_pipeline
    )
)

val_evaluator = dict(type='IoUMetric', metric='mIoU')
test_evaluator = dict(type='IoUMetric', metric='mIoU')
