_base_ = [
    '../_base_/datasets/scannet-3d-18class.py',
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
        num_classes=18,
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
            num_sizes=18,
            num_dir_bins=1,
            with_rot=False,
            mean_sizes=[[0.76966727, 0.8116021, 0.92573744],
                        [1.876858, 1.8425595, 1.1931566],
                        [0.61328, 0.6148609, 0.7182701],
                        [1.3955007, 1.5121545, 0.83443564],
                        [0.97949594, 1.0675149, 0.6329687],
                        [0.531663, 0.5955577, 1.7500148],
                        [0.9624706, 0.72462326, 1.1481868],
                        [0.83221924, 1.0490936, 1.6875663],
                        [0.21132214, 0.4206159, 0.5372846],
                        [1.4440073, 1.8970833, 0.26985747],
                        [1.0294262, 1.4040797, 0.87554324],
                        [1.3766412, 0.65521795, 1.6813129],
                        [0.6650819, 0.71111923, 1.298853],
                        [0.41999173, 0.37906948, 1.7513971],
                        [0.59359556, 0.5912492, 0.73919016],
                        [0.50867593, 0.50656086, 0.30136237],
                        [1.1511526, 1.0546296, 0.49706793],
                        [0.47535285, 0.49249494, 0.5802117]]),
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
            match_iou_weight=4.,
            match_point_weight=0.03,
            cost_class=1.5,
            cost_bbox=0.45,
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
optimizer = dict(type='AdamW', lr=0.008 * 0.9, weight_decay=1e-2)
# yapf:disable
log_config = dict(
    interval=30,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable