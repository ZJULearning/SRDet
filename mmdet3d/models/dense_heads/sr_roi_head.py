import copy
import math
import numpy as np
import torch
import torch.distributed as dist
from torch import nn as nn
from torch.nn import functional as F
from mmcv.cnn import normal_init

from mmdet3d.models.model_utils.utils import (
    is_dist_avail_and_initialized, get_world_size, roi_head_matchv2,
    get_src_permutation_idx, sigmoid_focal_loss,
    RoIAggPool3d, get_activation_fn, get_clones, DynamicMultiConvOneSimple)
from mmdet3d.models.losses.axis_aligned_iou_loss import AxisAlignedIoULoss
from mmdet3d.models.losses import SmoothL1Loss
from mmdet.core import multi_apply


class SRRoIHead(nn.Module):

    def __init__(self,
                 bbox_coder,
                 num_classes,
                 num_proposal,
                 train_cfg,
                 num_heads=6,
                 share_head=True,
                 cost_weights=(1., 1., 1., 1., 1., 1.),
                 init_feat_dim=256,
                 proposal_dim=128,
                 proposal_att_dim=128,
                 roi_agg_pool_layer_cfg=dict(
                     radii=0.3,
                     sample_num=16,
                     mlp_channels=[256, 256],
                     normalize_xyz=True,
                     norm_cfg=dict(type='BN2d'),
                     bias='auto',
                     pre_roi_conv_cfg=dict(
                         in_channels=256,
                         conv_channels=(128, 128),
                         connect_from=None,
                         conv_cfg=dict(type='Conv1d'),
                         norm_cfg=dict(type='BN1d'),
                         act_cfg=dict(type='ReLU')),
                     roi_pooler='RoIFPPool3d',
                     roi_pool_cfg=dict(
                         out_size=5,
                         max_pts_per_voxel=64,
                         mode='max')),
                 seed_roi_agg_update=dict(),
                 rcnn_cfg=dict(
                     att_layer_cfg=dict(
                         dim_feedforward=2048,
                         nhead=8,
                         dropout=0.1,
                         activation="relu",
                         dim_dynamic=64,
                         num_dynamic=2,
                         dy_conv='DynamicConvOneSimple',
                         dy_params=dict(),
                         use_mlp_att=False,
                         num_proposal=None),
                     pred_layer_cfg=dict(
                         num_cls=1,
                         num_reg=1)),
                 bbox_weights=(2., 2., 2., 1., 1., 1.),
                 match_cost_class=3.,
                 match_cost_bbox=0.9,
                 match_center_weight=3.,
                 match_iou_weight=2.,
                 match_point_weight=1.,
                 cost_class=1.5,
                 cost_bbox=0.45,
                 cost_dir_cls=0.,
                 cost_dir_res=0.,
                 center_weight=3.,
                 iou_weight=2.,
                 corner_weight=1.):
        super(SRRoIHead, self).__init__()
        self.bbox_coder = bbox_coder
        self.num_dir_bins = self.bbox_coder.num_dir_bins
        self.num_classes = num_classes
        self.num_proposal = num_proposal
        self.train_cfg = train_cfg
        self.num_heads = num_heads
        self.cost_weights = cost_weights
        assert self.num_heads == len(self.cost_weights)
        self.share_head = share_head
        self.init_feat_dim = init_feat_dim

        assert proposal_dim == roi_agg_pool_layer_cfg['pre_roi_conv_cfg']['conv_channels'][-1]
        rcnn_cfg['att_layer_cfg']['in_channels'] = proposal_dim
        rcnn_cfg['att_layer_cfg']['dy_params']['proposal_feat_dim'] = proposal_dim
        self.proposal_dim = proposal_dim
        self.proposal_att_dim = proposal_att_dim

        self.roi_agg_pool_layer_cfg = roi_agg_pool_layer_cfg
        self.seed_roi_agg_update = seed_roi_agg_update
        self.rcnn_cfg = rcnn_cfg

        self.bbox_weights = bbox_weights
        self.iou_loss = AxisAlignedIoULoss(reduction='sum', loss_weight=iou_weight)
        self.corner_loss = SmoothL1Loss(reduction='sum', loss_weight=corner_weight)

        self.match_cost_class = match_cost_class
        self.match_cost_bbox = match_cost_bbox
        self.match_center_weight = match_center_weight
        self.match_iou_weight = match_iou_weight
        self.match_point_weight = match_point_weight
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_dir_cls = cost_dir_cls
        self.cost_dir_res = cost_dir_res
        self.center_weight = center_weight

        self.mean_sizes = torch.tensor(self.bbox_coder.mean_sizes)
        self.supervise_suffix = [''] + [f'_aux_{i}' for i in range(num_heads - 1)]
        suffix_cost = [self.cost_weights[-1]] + list(self.cost_weights[:-1])
        self.suffix_cost_dict = dict(zip(self.supervise_suffix, suffix_cost))

        self.build_model()
        self.init_weights()

    def build_model(self):
        self.init_feat_mlp = nn.Sequential(
            nn.Linear(self.init_feat_dim, self.init_feat_dim),
            nn.LayerNorm(self.init_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.init_feat_dim, self.proposal_dim))

        self.vs_roi_agg_pool = RoIAggPool3d(**self.roi_agg_pool_layer_cfg)
        rcnn_head = RCNNHead(self.proposal_dim, self.proposal_att_dim, self.num_classes,
                             self.num_dir_bins, self.bbox_weights,
                             self.roi_agg_pool_layer_cfg['roi_pool_cfg'],
                             **self.rcnn_cfg)

        if self.share_head:
            self.head_series = nn.ModuleList([rcnn_head for _ in range(self.num_heads)])
        else:
            self.head_series = get_clones(rcnn_head, self.num_heads)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, inputs, init_proposal_boxes=None, init_proposal_feats=None):
        points_xyz = inputs['points_xyz']  # (N, point-num, 3)
        feat = inputs['features']  # (N, C, point-num)
        seeds_xyz = inputs['seeds_xyz']
        seeds_feat = inputs['seeds_features']
        vs_xyz = torch.cat([points_xyz, seeds_xyz], dim=1)
        vs_feat = torch.cat([feat, seeds_feat], dim=2)

        N = points_xyz.shape[0]
        proposal_boxes = init_proposal_boxes

        foreground_emb = init_proposal_feats
        foreground_emb = self.init_feat_mlp(foreground_emb)
        foreground_emb = foreground_emb.view(1, -1, self.proposal_dim)

        inter_class_logits = []
        inter_pred_bboxes = []
        inter_last_angle = []
        inter_pred_dirs = []
        inter_pred_angle = []
        for i, rcnn_head in enumerate(self.head_series):
            # proposal_boxes <=> (xc, yc, zc, l, w, h), which is different from sparse-rcnn
            # since the roi-pool requires different formats of boxes
            rois = proposal_boxes.clone().detach()
            rois[..., 2] = rois[..., 2] - rois[..., 5] / 2.
            inter_last_angle.append(rois[:, :, [-1]])

            if i == 0:
                vs_feat, vs_proposal_feat = self.vs_roi_agg_pool(vs_xyz, vs_feat, rois)
            else:
                vs_proposal_feat = self.vs_roi_agg_pool.roi_pooler(vs_xyz, vs_feat, rois)

            proposal_feat = vs_proposal_feat.flatten(2).permute(2, 0, 1)

            class_logits, pred_bboxes, pred_dirs, foreground_emb = rcnn_head(
                proposal_feat, foreground_emb, proposal_boxes)

            pred_angle = self.pred_dirs_to_angle(pred_dirs) + proposal_boxes[:, :, [-1]]
            proposal_boxes = torch.cat([pred_bboxes, pred_angle], dim=-1).detach()

            inter_class_logits.append(class_logits)
            inter_pred_bboxes.append(pred_bboxes)
            inter_pred_dirs.append(pred_dirs)
            inter_pred_angle.append(pred_angle)

        return inter_class_logits, inter_pred_bboxes, inter_last_angle, \
               inter_pred_dirs, inter_pred_angle

    def split_pred(self, inter_class_logits, inter_pred_bboxes, inter_last_angle,
                   inter_pred_dirs, inter_pred_angle):
        results = {'sem_scores': inter_class_logits[-1], 'bboxes': inter_pred_bboxes[-1],
                   'last_angles': inter_last_angle[-1], 'dirs': inter_pred_dirs[-1],
                   'angles': inter_pred_angle[-1]}

        for i in range(len(inter_class_logits) - 1):
            results[f'sem_scores_aux_{i}'] = inter_class_logits[i]
            results[f'bboxes_aux_{i}'] = inter_pred_bboxes[i]
            results[f'last_angles_aux_{i}'] = inter_last_angle[i]
            results[f'dirs_aux_{i}'] = inter_pred_dirs[i]
            results[f'angles_aux_{i}'] = inter_pred_angle[i]

        return results

    def pred_dirs_to_angle(self, dirs):
        dir_class = dirs[:, :, :self.num_dir_bins]
        dir_res = dirs[:, :, self.num_dir_bins:]
        dir_class = torch.argmax(dir_class, -1)
        dir_res = torch.gather(dir_res, 2, dir_class.unsqueeze(-1))
        dir_res.squeeze_(2)
        dir_angle = self.bbox_coder.class2angle(dir_class, dir_res).reshape(*dirs.shape[:2], 1)
        return dir_angle

    def decode(self, bbox_preds):
        sem_scores = torch.sigmoid(bbox_preds['sem_scores'])
        obj_scores = sem_scores.max(-1)[0]
        bbox_preds['size_class'] = bbox_preds['sem_scores']

        bboxes = bbox_preds['bboxes']
        if self.bbox_coder.with_rot:
            dir_angle = bbox_preds['angles']
        else:
            dir_angle = bboxes.new_zeros(*bboxes.shape[:2], 1)
        bbox3d = torch.cat([bboxes, dir_angle], dim=-1)
        return sem_scores, obj_scores, bbox3d

    def get_losses_single_head(self,
                               suffix,
                               sem_scores,
                               bboxes,
                               dirs,
                               class_targets,
                               center_targets,
                               size_res_norm_targets,
                               dir_class_targets,
                               dir_res_targets,
                               angles,
                               gt_corner3d,
                               idx,
                               num_boxes,
                               img_metas=None):
        pred_center = bboxes[:, :, :3][idx]  # (gt-num, 3)
        pred_size = bboxes[:, :, 3:6][idx]
        pred_angles = angles[idx]
        mean_sizes = self.mean_sizes.mean(0).to(bboxes.device)
        pred_size_res_norm = (pred_size - mean_sizes) / mean_sizes

        center_loss = F.l1_loss(pred_center, center_targets, reduction='none').sum() / num_boxes
        size_loss = F.l1_loss(pred_size_res_norm, size_res_norm_targets,
                              reduction='none').sum() / num_boxes

        # calculate class loss
        src_logits = sem_scores.flatten(0, 1)  # (batch * proposal_num, num_classes)
        target_classes = class_targets.flatten(0, 1)  # (batch * proposal_num,)
        pos_inds = torch.nonzero(target_classes != self.num_classes, as_tuple=True)[0]
        labels = torch.zeros_like(src_logits)
        labels[pos_inds, target_classes[pos_inds]] = 1  # (batch * proposal_num, num_classes)

        semantic_loss = sigmoid_focal_loss(
            src_logits,
            labels,
            alpha=0.25,
            gamma=2.0,
            reduction="sum",
        ) / num_boxes

        pred_dir_class = dirs[:, :, :self.num_dir_bins][idx]  # (gt-num, self.num_dir_bins)
        dir_cls_loss = F.cross_entropy(pred_dir_class, dir_class_targets, reduction='mean')

        heading_label_one_hot = torch.zeros_like(pred_dir_class)  # (gt-num, self.num_dir_bins)
        heading_label_one_hot.scatter_(1, dir_class_targets.unsqueeze(-1), 1)
        pred_dir_res_norm = dirs[:, :, self.num_dir_bins:][idx]
        dir_res_norm = torch.sum(
            pred_dir_res_norm * heading_label_one_hot, -1)
        dir_res_loss = F.l1_loss(dir_res_norm, dir_res_targets,
                                 reduction='none').sum() / num_boxes

        cost_weight = self.suffix_cost_dict[suffix]
        losses = {
            f'semantic_loss{suffix}': semantic_loss * self.cost_class * cost_weight,
            f'center_loss{suffix}': center_loss * self.cost_bbox * self.center_weight * cost_weight,
            f'size_loss{suffix}': size_loss * self.cost_bbox * cost_weight,
            f'dir_cls_loss{suffix}': dir_cls_loss * self.cost_bbox * self.cost_dir_cls,
            f'dir_res_loss{suffix}': dir_res_loss * self.cost_bbox * self.cost_dir_res}

        pred_size = torch.clamp(pred_size, 0)
        half_pred_size = pred_size / 2.
        pred_corners = torch.cat([pred_center - half_pred_size,
                                  pred_center + half_pred_size], dim=-1)
        half_size_targets = (size_res_norm_targets * mean_sizes + mean_sizes) / 2.
        target_corners = torch.cat([center_targets - half_size_targets,
                                    center_targets + half_size_targets], dim=-1)
        iou_loss = self.iou_loss(pred_corners, target_corners) / num_boxes
        losses.update({f'iou_loss{suffix}': iou_loss * self.cost_bbox * cost_weight})

        # calculate corner loss
        pred_bbox3d = torch.cat([pred_center, pred_size, pred_angles], dim=-1)
        pred_bbox3d = pred_bbox3d.reshape(-1, pred_bbox3d.shape[-1])
        pred_bbox3d = img_metas[0]['box_type_3d'](
            pred_bbox3d.clone(),
            box_dim=pred_bbox3d.shape[-1],
            with_yaw=self.bbox_coder.with_rot,
            origin=(0.5, 0.5, 0.5))
        pred_corners3d = pred_bbox3d.corners.reshape(-1, 8, 3)
        corner_loss = self.corner_loss(
            pred_corners3d,
            gt_corner3d.reshape(-1, 8, 3)) / num_boxes
        losses.update({f'corner_loss{suffix}': corner_loss * self.cost_bbox * cost_weight})

        return losses, None

    def get_losses(self, bbox_preds, img_metas, class_targets, size_res_norm_targets,
                   dir_class_targets, dir_res_targets, center_targets,
                   gt_corner3d, box_loss_weights, num_boxes, idx):
        sem_scores = [bbox_preds[f'sem_scores{n}'] for n in self.supervise_suffix]
        bboxes = [bbox_preds[f'bboxes{n}'] for n in self.supervise_suffix]
        dirs = [bbox_preds[f'dirs{n}'] for n in self.supervise_suffix]
        angles = [bbox_preds[f'angles{n}'] for n in self.supervise_suffix]

        loss_list, _ = multi_apply(
            self.get_losses_single_head, self.supervise_suffix, sem_scores, bboxes, dirs,
            class_targets, center_targets, size_res_norm_targets,
            dir_class_targets, dir_res_targets, angles, gt_corner3d, idx, num_boxes=num_boxes,
            img_metas=img_metas)

        losses = dict()
        for loss_dict in loss_list:
            losses.update(loss_dict)

        return losses

    def get_targets_single_head(self,
                                pred_logits,
                                pred_bboxes,
                                last_angles,
                                pred_angles,
                                gt_labels_3d,
                                center_targets,
                                size_res_norm_targets,
                                angle_targets,
                                gt_bboxes_3d,
                                bbox_preds):
        # example: indices => [(tensor(14, 29, 90), tensor(2, 0, 1)), (tensor(82), tensor(0))]
        # batchsize = 2. for 1-th and 2-th batch, we have 3 gt and 1 gt.
        pred_centers = pred_bboxes[:, :, :3]
        pred_size = pred_bboxes[:, :, 3:6]
        mean_sizes = self.mean_sizes.mean(0).to(pred_centers.device)
        pred_size_res_norm = (pred_size - mean_sizes) / mean_sizes
        gt_corner3d = [gt_bbox_3d.corners.cuda() for gt_bbox_3d in gt_bboxes_3d]
        feat_points = None
        indices = roi_head_matchv2(
            pred_logits, pred_centers, pred_size_res_norm, pred_angles,
            [{'labels': gt_labels_3d[i].cuda(), 'centers': center_targets[i].cuda(),
              'sizes_res_norm': size_res_norm_targets[i],
              'angles': angle_targets[i]} for i in range(len(gt_labels_3d))],
            self.match_cost_class, self.match_cost_bbox, self.match_center_weight,
            use_focal=True, use_iou=True, iou_weight=self.match_iou_weight,
            mean_size=self.mean_sizes.mean(0).to(pred_logits.device),
            with_roi=self.bbox_coder.with_rot, feat_points=feat_points,
            point_weight=self.match_point_weight)

        # example: idx => [tensor(0, 0, 0, 1), tensor(14, 29, 90, 82)]
        idx = get_src_permutation_idx(indices)
        batch = len(gt_labels_3d)

        # 1. class_targets
        # example: class_targets_o: (4,)
        class_targets_o = torch.cat([t[J] for t, (_, J) in zip(gt_labels_3d, indices)])
        # size_class_targets: (batch, pred). the last class is background
        class_targets = torch.full((batch, self.num_proposal), self.num_classes,
                                   dtype=torch.int64, device=pred_logits.device)
        class_targets[idx] = class_targets_o.cuda()

        # 2. center_targets
        center_targets = torch.cat([t[J] for t, (_, J) in zip(center_targets, indices)]).cuda()

        # 3. size_res_targets
        size_res_norm_targets = torch.cat([t[J] for t, (_, J) in zip(
            size_res_norm_targets, indices)]).cuda()

        # 4. dir_class_targets
        angle_targets = torch.cat([t[J] for t, (_, J) in zip(angle_targets, indices)]).cuda()
        last_angles = last_angles[idx]
        angle_targets = angle_targets - last_angles.squeeze(-1)
        dir_class_targets, dir_res_targets = self.bbox_coder.angle2class(angle_targets)

        # 5. gt_corner3d
        gt_corner3d = torch.cat([t[J] for t, (_, J) in zip(gt_corner3d, indices)]).cuda()

        objectness_targets = class_targets != self.num_classes
        pos_weights = objectness_targets

        box_loss_weights = pos_weights.float() / (torch.sum(pos_weights).float() + 1e-6)
        return (class_targets, center_targets, size_res_norm_targets, dir_class_targets,
                dir_res_targets, gt_corner3d, box_loss_weights, idx)

    def get_targets(self, points, gt_bboxes_3d, gt_labels_3d, bbox_preds):
        (center_targets, _, size_res_targets, _, _) = multi_apply(
            self.bbox_coder.encode, gt_bboxes_3d, gt_labels_3d)

        mean_size = self.mean_sizes.mean(0).cuda()
        size_res_norm_targets = [(b.dims.cuda() - mean_size) / mean_size for b in gt_bboxes_3d]
        angle_targets = [b.yaw.cuda() for b in gt_bboxes_3d]

        pred_logits = [bbox_preds[f'sem_scores{n}'] for n in self.supervise_suffix]
        pred_bboxes = [bbox_preds[f'bboxes{n}'] for n in self.supervise_suffix]
        last_angles = [bbox_preds[f'last_angles{n}'] for n in self.supervise_suffix]
        pred_angles = [bbox_preds[f'angles{n}'] for n in self.supervise_suffix]
        (class_targets, center_targets, size_res_norm_targets, dir_class_targets,
         dir_res_targets, gt_corner3d, box_loss_weights, idx) = multi_apply(
            self.get_targets_single_head, pred_logits, pred_bboxes, last_angles,
            pred_angles, gt_labels_3d=gt_labels_3d, center_targets=center_targets,
            size_res_norm_targets=size_res_norm_targets,
            angle_targets=angle_targets, gt_bboxes_3d=gt_bboxes_3d, bbox_preds=bbox_preds)

        num_boxes = sum(len(t) for t in gt_labels_3d)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float,
                                    device=next(iter(bbox_preds.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        return (class_targets, size_res_norm_targets, dir_class_targets, dir_res_targets,
                center_targets, gt_corner3d, box_loss_weights, num_boxes, idx)


class RCNNHead(nn.Module):

    def __init__(self,
                 proposal_dim,
                 proposal_att_dim,
                 num_classes,
                 num_dir_bins,
                 bbox_weights,
                 roi_pool_cfg,
                 att_layer_cfg,
                 pred_layer_cfg):
        super(RCNNHead, self).__init__()
        self.proposal_dim = proposal_dim
        self.proposal_att_dim = proposal_att_dim
        self.num_classes = num_classes
        self.num_dir_bins = num_dir_bins
        self.bbox_weights = bbox_weights
        self.roi_pool_cfg = roi_pool_cfg
        self.pooler_resolution = self.roi_pool_cfg['out_size']
        if isinstance(self.pooler_resolution, (list, tuple)):
            self.pooler_grid_num = sum(p ** 3 for p in self.pooler_resolution)
        self.att_layer_cfg = att_layer_cfg
        self.pred_layer_cfg = pred_layer_cfg

        self.scale_clamp = math.log(100000.0 / 16)

        self.build_att_model()
        self.build_pred_model()

    def build_att_model(self):
        nhead = self.att_layer_cfg['nhead']
        dim_feedforward = self.att_layer_cfg['dim_feedforward']
        dropout = self.att_layer_cfg['dropout']
        activation = self.att_layer_cfg['activation']

        dim_dynamic = self.att_layer_cfg['dim_dynamic']
        num_dynamic = self.att_layer_cfg['num_dynamic']
        dy_conv = self.att_layer_cfg['dy_conv']
        dy_params = self.att_layer_cfg['dy_params']
        attention_layer = nn.MultiheadAttention

        self.attn_down = nn.Sequential(
            nn.Linear(self.proposal_dim, self.proposal_att_dim),
            nn.LayerNorm(self.proposal_att_dim))

        self.self_attn = attention_layer(self.proposal_att_dim, nhead, dropout=dropout)

        self.attn_up = nn.Sequential(
            nn.Linear(self.proposal_att_dim, self.proposal_dim),
            nn.LayerNorm(self.proposal_dim))

        # new options
        self.inst_interact = eval(dy_conv)(self.proposal_dim, dim_dynamic, num_dynamic,
                                            self.pooler_resolution, **dy_params)

        # roi-use-mlp

        self.emb_norm = nn.Identity()

        self.norm2 = nn.LayerNorm(self.proposal_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(self.proposal_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(self.proposal_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, self.proposal_dim)
        self.norm3 = nn.LayerNorm(self.proposal_dim)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    def build_pred_model(self):
        # cls.
        num_cls = self.pred_layer_cfg['num_cls']
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(self.proposal_dim, self.proposal_dim, False))
            cls_module.append(nn.LayerNorm(self.proposal_dim))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        num_reg = self.pred_layer_cfg['num_reg']
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(self.proposal_dim, self.proposal_dim, False))
            reg_module.append(nn.LayerNorm(self.proposal_dim))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)

        # pred.
        self.class_logits = nn.Linear(self.proposal_dim, self.num_classes)
        self.bboxes_delta = nn.Linear(self.proposal_dim, 6 + self.num_dir_bins * 2)

    def forward(self, proposal_feats, foreground_emb, proposal_boxes):
        # proposal_feats: (7*7*7, N * nr_boxes, C)
        N, nr_boxes = proposal_boxes.shape[:2]

        # inst_interact.
        pro_features2, _ = self.inst_interact(foreground_emb, proposal_feats)

        pro_features2 = pro_features2.view(1, N * nr_boxes, -1)
        pro_features = foreground_emb + self.dropout2(pro_features2)
        pro_features = self.norm2(pro_features)  # (1, N*roi-num, C)

        # (nr-boxes, N, self.proposal-feat-dim)
        pro_features = pro_features.view(N, nr_boxes, -1).transpose(0, 1)
        pro_features_down = self.attn_down(pro_features)
        pro_features2 = self.self_attn(pro_features_down, pro_features_down,
                                        value=pro_features_down)[0]
        pro_features2 = self.attn_up(pro_features2)

        pro_features = pro_features + self.dropout1(pro_features2)
        obj_features = self.norm1(pro_features).transpose(0, 1).reshape(1, N * nr_boxes, -1)

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)  # (1, N*roi-num, C)

        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)

        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)

        class_logits = self.class_logits(cls_feature)  # (N*num-roi, num-class+1)
        bboxes_deltas = self.bboxes_delta(reg_feature)  # (N*num-roi, 6 + 2*bin-num)
        pred_bboxes = self.apply_deltas(bboxes_deltas, proposal_boxes[..., :6].view(-1, 6))
        pred_dirs = bboxes_deltas[:, 6:]

        return class_logits.view(N, nr_boxes, -1), pred_bboxes.view(N, nr_boxes, -1), \
               pred_dirs.view(N, nr_boxes, -1), obj_features

    def apply_deltas(self, deltas, boxes):
        ctr_x, ctr_y, ctr_z, l, w, h = boxes.transpose(0, 1)

        wx, wy, wz, wl, ww, wh = self.bbox_weights
        dx = deltas[:, [0]] / wx
        dy = deltas[:, [1]] / wy
        dl = deltas[:, [3]] / wl
        dz = deltas[:, [2]] / wz
        dw = deltas[:, [4]] / ww
        dh = deltas[:, [5]] / wh

        # Prevent sending too large values into torch.exp()
        dl = torch.clamp(dl, max=self.scale_clamp)
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * l[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * w[:, None] + ctr_y[:, None]
        pred_ctr_z = dz * h[:, None] + ctr_z[:, None]
        pred_l = torch.exp(dl) * l[:, None]
        pred_w = torch.exp(dw) * w[:, None]
        pred_h = torch.exp(dh) * h[:, None]

        pred_boxes = torch.cat([pred_ctr_x, pred_ctr_y, pred_ctr_z, pred_l, pred_w, pred_h], dim=-1)
        return pred_boxes


def box_mapping(points_xyz, proposal_boxes):
    points_np = points_xyz.cpu().detach().numpy()
    point_min = torch.tensor(
        np.percentile(points_np, 5, axis=1, keepdims=True)).to(proposal_boxes.device)
    point_max = torch.tensor(
        np.percentile(points_np, 95, axis=1, keepdims=True)).to(proposal_boxes.device)
    proposal_boxes[:, :, 0:3] = proposal_boxes[:, :, 0:3] * (point_max - point_min) + point_min
    proposal_boxes[:, :, 3:6] = proposal_boxes[:, :, 3:6] * (point_max - point_min)

    return proposal_boxes
