_base_ = [
    '../_base_/datasets/sunrgbd-3d-10class.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='VoteNet',
    backbone=dict(
        type='PointNet2SASSG',
        in_channels=4,
        num_points=(2048, 1024, 512, 256),
        radius=(0.2, 0.4, 0.8, 1.2),
        num_samples=(64, 32, 16, 16),
        sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                     (128, 128, 256)),
        fp_channels=((256, 256), (256, 256)),
        norm_cfg=dict(type='BN2d'),
        sa_cfg=dict(
            type='PointSAModule',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=True)),
    bbox_head=dict(
        type='SRHead',
        num_classes=10,
        num_proposal=128,
        vote_module='NoiseSuppressionModule',
        vote_module_cfg=dict(
            in_channels=256,
            vote_per_seed=1,
            gt_per_seed=3,
            conv_channels=(256, 256),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            cls_weight=3.,
            cls_bg_weight=0.2,
            norm_feats=True,
            vote_loss=dict(
                type='ChamferDistance',
                mode='l1',
                reduction='none',
                loss_dst_weight=10.0)),
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',
            num_sizes=10,
            num_dir_bins=12,
            with_rot=True,
            mean_sizes=[
                [2.114256, 1.620300, 0.927272], [0.791118, 1.279516, 0.718182],
                [0.923508, 1.867419, 0.845495], [0.591958, 0.552978, 0.827272],
                [0.699104, 0.454178, 0.75625], [0.69519, 1.346299, 0.736364],
                [0.528526, 1.002642, 1.172878], [0.500618, 0.632163, 0.683424],
                [0.404671, 1.071108, 1.688889], [0.76584, 1.398258, 0.472728]
            ]),
        init_head_module='SRInitHead',
        init_head_cfg=dict(
            num_proposal=160,
            vote_aggregation_cfg=dict(
                type='PointSAModule',
                radius=0.3,
                num_sample=16,
                mlp_channels=[256, 256, 256, 256],
                use_xyz=True,
                normalize_xyz=True),
            pred_layer_cfg=dict(
                in_channels=256, shared_conv_channels=(256, 128), bias=True),
            objectness_loss=dict(
                type='CrossEntropyLoss',
                class_weight=[0.2, 0.8],
                reduction='sum',
                loss_weight=5.0),
            center_loss=dict(
                type='ChamferDistance',
                mode='l2',
                reduction='sum',
                loss_src_weight=10.0,
                loss_dst_weight=10.0),
            dir_class_loss=dict(
                type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
            dir_res_loss=dict(
                type='SmoothL1Loss', reduction='sum', loss_weight=10.0),
            size_class_loss=dict(
                type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
            size_res_loss=dict(
                type='SmoothL1Loss', reduction='sum', loss_weight=10.0 / 3.0)),
        roi_head_module='SRRoIHead',
        use_nms=False,
        roi_head_cfg=dict(
            num_heads=3,
            cost_weights=(1., 1., 1.),
            proposal_dim=128,
            proposal_att_dim=128,
            init_feat_dim=256,
            roi_agg_pool_layer_cfg=dict(
                 radii=0.8,
                 sample_num=64,
                 mlp_channels=[256, 256, 256, 256],
                 normalize_xyz=True,
                 use_agg=False,
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 pre_roi_conv_cfg=dict(
                     in_channels=256,
                     conv_channels=(128, 128),
                     connect_from=None,
                     conv_cfg=dict(type='Conv1d'),
                     norm_cfg=dict(type='BN1d'),
                     act_cfg=dict(type='ReLU')),
                 roi_pooler='MultiRoIPool3d',
                 roi_pool_cfg=dict(
                     out_size=[1, 3, 5],
                     max_pts_per_voxel=[96, 96, 96],
                     mode='max')),
            seed_roi_agg_update=dict(use_agg=False),
            rcnn_cfg=dict(
                att_layer_cfg=dict(
                    dim_feedforward=2048,
                    nhead=8,
                    dropout=0.1,
                    activation="relu",
                    dim_dynamic=64,
                    num_dynamic=2,
                    dy_conv='DynamicMultiConvOneSimple',
                    dy_params=dict(roi_out=[1, 3, 5], proposal_feat_dim=128)),
                pred_layer_cfg=dict(
                    num_cls=1,
                    num_reg=1)),
            bbox_weights=(2., 2., 2., 1., 1., 1.),
            match_cost_class=3.,
            match_cost_bbox=0.9,
            match_center_weight=3.,
            match_iou_weight=0.,
            cost_class=1.5,
            cost_bbox=0.45,
            cost_dir_cls=1.,
            cost_dir_res=10.,
            center_weight=3.,
            iou_weight=2.,
            corner_weight=.25)))
# model training and testing settings
train_cfg = dict(pos_distance_thr=0.3, neg_distance_thr=0.6, sample_mod='vote')
test_cfg = dict(
    sample_mod='seed', nms_thr=0.25, score_thr=0.01, per_class_proposal=True)

data = dict(
    samples_per_gpu=4,
)
# optimizer
# yapf:disable
log_config = dict(
    interval=30,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
optimizer = dict(type='AdamW', lr=0.008 * 0.4, weight_decay=1e-2)
