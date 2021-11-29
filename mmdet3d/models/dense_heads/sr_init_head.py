import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from mmcv.cnn import normal_init
from mmdet3d.models.builder import build_loss
from mmdet3d.models.losses import chamfer_distance
from mmdet3d.ops import build_sa_module
from mmdet.core import multi_apply
from .base_conv_bbox_head import BaseConvBboxHead


class SRInitHead(nn.Module):

    def __init__(self,
                 bbox_coder,
                 num_classes,
                 train_cfg,
                 num_proposal=1024,
                 vote_aggregation_cfg=None,
                 pred_layer_cfg=None,
                 objectness_loss=None,
                 center_loss=None,
                 dir_class_loss=None,
                 dir_res_loss=None,
                 size_class_loss=None,
                 size_res_loss=None):
        super(SRInitHead, self).__init__()
        self.bbox_coder = bbox_coder
        self.num_sizes = self.bbox_coder.num_sizes
        self.num_dir_bins = self.bbox_coder.num_dir_bins
        self.num_classes = num_classes
        self.num_proposal = num_proposal
        self.vote_aggregation_cfg = vote_aggregation_cfg
        self.pred_layer_cfg = pred_layer_cfg
        self.train_cfg = train_cfg

        self.objectness_loss = build_loss(objectness_loss)
        self.center_loss = build_loss(center_loss)
        self.dir_res_loss = build_loss(dir_res_loss)
        self.dir_class_loss = build_loss(dir_class_loss)
        self.size_res_loss = build_loss(size_res_loss)
        self.size_class_loss = build_loss(size_class_loss)

        self.build_model()
        self.init_weights()

    def build_model(self):
        self.vote_aggregation_cfg['num_point'] = self.num_proposal
        self.vote_aggregation = build_sa_module(self.vote_aggregation_cfg)

        self.head_pred = BaseConvBboxHead(
            **self.pred_layer_cfg,
            num_cls_out_channels=self._get_cls_out_channels(),
            num_reg_out_channels=self._get_reg_out_channels())

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                normal_init(m, std=0.01)

    def _get_cls_out_channels(self):
        """Return the channel number of classification outputs."""
        # Class numbers (k) + objectness (2)
        return self.num_classes + 2

    def _get_reg_out_channels(self):
        """Return the channel number of regression outputs."""
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_dir_bins*2),
        # size class+residual(num_sizes*4)
        return 3 + self.num_dir_bins * 2 + self.num_sizes * 4

    def forward(self, inputs):
        aggregation_inputs = dict(points_xyz=inputs['points_xyz'], features=inputs['features'])
        aggregated_points, aggregated_features, _ = self.vote_aggregation(**aggregation_inputs)

        head_pred = self.head_pred(aggregated_features)
        return aggregated_points, aggregated_features, head_pred

    def split_pred(self, cls_preds, reg_preds, head_inputs):
        results = {}
        start, end = 0, 0

        base_xyz = head_inputs['aggregated_points']
        cls_preds_trans = cls_preds.transpose(1, 2)
        reg_preds_trans = reg_preds.transpose(1, 2)

        # decode center
        end += 3
        # (batch_size, num_proposal, 3)
        results['init_center'] = base_xyz + \
                                 reg_preds_trans[..., start:end].contiguous()
        start = end

        # decode direction
        end += self.num_dir_bins
        results['init_dir_class'] = reg_preds_trans[..., start:end].contiguous()
        start = end

        end += self.num_dir_bins
        dir_res_norm = reg_preds_trans[..., start:end].contiguous()
        start = end

        results['init_dir_res_norm'] = dir_res_norm
        results['init_dir_res'] = dir_res_norm * (np.pi / self.num_dir_bins)

        # decode size
        end += self.num_sizes
        results['init_size_class'] = reg_preds_trans[..., start:end].contiguous()
        start = end

        end += self.num_sizes * 3
        size_res_norm = reg_preds_trans[..., start:end]
        batch_size, num_proposal = reg_preds_trans.shape[:2]
        size_res_norm = size_res_norm.view(
            [batch_size, num_proposal, self.num_sizes, 3])
        start = end

        results['init_size_res_norm'] = size_res_norm.contiguous()
        mean_sizes = reg_preds.new_tensor(self.bbox_coder.mean_sizes)
        results['init_size_res'] = (
                size_res_norm * mean_sizes.unsqueeze(0).unsqueeze(0))

        # decode objectness score
        start = 0
        end = 2
        results['init_obj_scores'] = cls_preds_trans[..., start:end].contiguous()
        start = end

        # decode semantic score
        results['init_sem_scores'] = cls_preds_trans[..., start:].contiguous()

        return results

    def bbox_decode(self, bbox_out, suffix=''):
        """Decode predicted parts to bbox3d.

        Args:
            bbox_out (dict): Predictions from model, should contain keys below.

                - center: predicted bottom center of bboxes.
                - dir_class: predicted bbox direction class.
                - dir_res: predicted bbox direction residual.
                - size_class: predicted bbox size class.
                - size_res: predicted bbox size residual.
            suffix (str): Decode predictions with specific suffix.

        Returns:
            torch.Tensor: Decoded bbox3d with shape (batch, n, 7).
        """
        center = bbox_out['init_center' + suffix]
        batch_size, num_proposal = center.shape[:2]

        # decode heading angle
        if self.bbox_coder.with_rot:
            dir_class = torch.argmax(bbox_out['init_dir_class' + suffix], -1)
            dir_res = torch.gather(bbox_out['init_dir_res' + suffix], 2,
                                   dir_class.unsqueeze(-1))
            dir_res.squeeze_(2)
            dir_angle = self.bbox_coder.class2angle(dir_class, dir_res).reshape(
                batch_size, num_proposal, 1)
        else:
            dir_angle = center.new_zeros(batch_size, num_proposal, 1)

        # decode bbox size
        size_class = torch.argmax(
            bbox_out['init_size_class' + suffix], -1, keepdim=True)
        size_res = torch.gather(bbox_out['init_size_res' + suffix], 2,
                                size_class.unsqueeze(-1).repeat(1, 1, 1, 3))
        mean_sizes = center.new_tensor(self.bbox_coder.mean_sizes)
        size_base = torch.index_select(mean_sizes, 0, size_class.reshape(-1))
        bbox_size = size_base.reshape(batch_size, num_proposal,
                                      -1) + size_res.squeeze(2)

        bbox3d = torch.cat([center, bbox_size, dir_angle], dim=-1)
        return bbox3d

    def decode(self, bbox_preds):
        obj_scores = F.softmax(bbox_preds['init_obj_scores'], dim=-1)[..., -1]
        sem_scores = F.softmax(bbox_preds['init_sem_scores'], dim=-1)
        bbox3d = self.bbox_decode(bbox_preds)
        return sem_scores, obj_scores, bbox3d

    def get_init_proposals(self, cls_preds, reg_preds, head_inputs, out_proposal_num):
        init_results = self.split_pred(cls_preds, reg_preds, head_inputs)
        sem_scores, obj_scores, bbox3d = self.decode(init_results)
        feats = head_inputs['aggregated_features'].transpose(1, 2)  # (N, proposal-num, C)

        proposal_boxes = list()
        proposal_feats = list()
        batch_size = bbox3d.shape[0]
        for b in range(batch_size):
            _, topk_indices = obj_scores[b].topk(out_proposal_num, sorted=False)
            box_pred_per_img = bbox3d[b][topk_indices]
            feat_pred_per_img = feats[b][topk_indices]
            proposal_boxes.append(box_pred_per_img)
            proposal_feats.append(feat_pred_per_img)
        proposal_boxes = torch.stack(proposal_boxes, dim=0)
        proposal_feats = torch.stack(proposal_feats, dim=0)
        return proposal_boxes, proposal_feats, init_results

    def get_losses(self, bbox_preds, size_class_targets, size_res_targets,
                   dir_class_targets, dir_res_targets, center_targets, mask_targets,
                   objectness_targets, objectness_weights, box_loss_weights, valid_gt_weights):
        # calculate objectness loss
        objectness_loss = self.objectness_loss(
            bbox_preds['init_obj_scores'].transpose(2, 1),
            objectness_targets,
            weight=objectness_weights)

        # calculate center loss
        source2target_loss, target2source_loss = self.center_loss(
            bbox_preds['init_center'],
            center_targets,
            src_weight=box_loss_weights,
            dst_weight=valid_gt_weights)
        center_loss = source2target_loss + target2source_loss

        # calculate direction class loss
        dir_class_loss = self.dir_class_loss(
            bbox_preds['init_dir_class'].transpose(2, 1),
            dir_class_targets,
            weight=box_loss_weights)

        # calculate direction residual loss
        batch_size, proposal_num = size_class_targets.shape[:2]
        heading_label_one_hot = bbox_preds['init_center'].new_zeros(
            (batch_size, proposal_num, self.num_dir_bins))
        heading_label_one_hot.scatter_(2, dir_class_targets.unsqueeze(-1), 1)
        dir_res_norm = torch.sum(
            bbox_preds['init_dir_res_norm'] * heading_label_one_hot, -1)
        dir_res_loss = self.dir_res_loss(
            dir_res_norm, dir_res_targets, weight=box_loss_weights)

        # calculate size class loss
        size_class_loss = self.size_class_loss(
            bbox_preds['init_size_class'].transpose(2, 1),
            size_class_targets,
            weight=box_loss_weights)

        # calculate size residual loss
        one_hot_size_targets = bbox_preds['init_center'].new_zeros(
            (batch_size, proposal_num, self.num_sizes))
        one_hot_size_targets.scatter_(2, size_class_targets.unsqueeze(-1), 1)
        one_hot_size_targets_expand = one_hot_size_targets.unsqueeze(
            -1).repeat(1, 1, 1, 3).contiguous()
        size_residual_norm = torch.sum(
            bbox_preds['init_size_res_norm'] * one_hot_size_targets_expand, 2)
        box_loss_weights_expand = box_loss_weights.unsqueeze(-1).repeat(
            1, 1, 3)
        size_res_loss = self.size_res_loss(
            size_residual_norm,
            size_res_targets,
            weight=box_loss_weights_expand)

        # calculate semantic loss

        return dict(
            init_objectness_loss=objectness_loss,
            init_center_loss=center_loss,
            init_dir_class_loss=dir_class_loss,
            init_dir_res_loss=dir_res_loss,
            init_size_class_loss=size_class_loss,
            init_size_res_loss=size_res_loss)

    def get_targets(self, points, gt_bboxes_3d, gt_labels_3d, max_gt_num,
                    valid_gt_masks, bbox_preds):
        if self.vote_aggregation_cfg is None:
            agg_points = bbox_preds['vote_points']
        else:
            agg_points = bbox_preds['aggregated_points']
        (size_class_targets, size_res_targets, dir_class_targets, dir_res_targets, center_targets,
         mask_targets, objectness_targets, objectness_masks, pos_weights) = multi_apply(
            self.get_targets_single, points, gt_bboxes_3d, gt_labels_3d, agg_points)

        # pad targets as original code of votenet.
        center_targets, valid_gt_masks = self.pad_targets(
            gt_labels_3d, max_gt_num, center_targets, valid_gt_masks)
        center_targets = torch.stack(center_targets)
        valid_gt_masks = torch.stack(valid_gt_masks)
        objectness_targets = torch.stack(objectness_targets)
        objectness_masks = torch.stack(objectness_masks)
        pos_weights = torch.stack(pos_weights)

        # all weights are normalized
        objectness_weights = objectness_masks / (torch.sum(objectness_masks) + 1e-6)
        box_loss_weights = pos_weights.float() / (torch.sum(pos_weights).float() + 1e-6)
        valid_gt_weights = valid_gt_masks.float() / (torch.sum(valid_gt_masks.float()) + 1e-6)

        dir_class_targets = torch.stack(dir_class_targets)
        dir_res_targets = torch.stack(dir_res_targets)
        size_class_targets = torch.stack(size_class_targets)
        size_res_targets = torch.stack(size_res_targets)
        mask_targets = torch.stack(mask_targets)

        return (size_class_targets, size_res_targets, dir_class_targets, dir_res_targets,
                center_targets, mask_targets, objectness_targets, objectness_weights,
                box_loss_weights, valid_gt_weights)

    def get_targets_single(self,
                           points,
                           gt_bboxes_3d,
                           gt_labels_3d,
                           aggregated_points):
        """Generate targets of vote head for single batch.

        Args:
            points (torch.Tensor): Points of each batch.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth \
                boxes of each batch.
            gt_labels_3d (torch.Tensor): Labels of each batch.
            aggregated_points (torch.Tensor): Aggregated points from
                vote aggregation layer.

        Returns:
            tuple[torch.Tensor]: Targets of vote head.
        """
        gt_bboxes_3d = gt_bboxes_3d.to(points.device)

        (center_targets, size_class_targets, size_res_targets,
         dir_class_targets, dir_res_targets) = self.bbox_coder.encode(gt_bboxes_3d, gt_labels_3d)

        # distance1: (1, proposal_num). assignment: (1, proposal_num)
        distance1, _, assignment, _ = chamfer_distance(
            aggregated_points.unsqueeze(0),
            center_targets.unsqueeze(0),
            reduction='none')
        assignment = assignment.squeeze(0)  # assign the nearest gt to each proposal

        objectness_targets, objectness_masks, pos_weights = \
            self.get_vote_objectness_and_pos_weights(distance1)

        dir_class_targets = dir_class_targets[assignment]
        dir_res_targets = dir_res_targets[assignment]
        dir_res_targets /= (np.pi / self.num_dir_bins)
        size_class_targets = size_class_targets[assignment]
        size_res_targets = size_res_targets[assignment]

        one_hot_size_targets = gt_bboxes_3d.tensor.new_zeros(
            (self.num_proposal, self.num_sizes))
        one_hot_size_targets.scatter_(1, size_class_targets.unsqueeze(-1), 1)
        one_hot_size_targets = one_hot_size_targets.unsqueeze(-1).repeat(
            1, 1, 3)
        mean_sizes = size_res_targets.new_tensor(
            self.bbox_coder.mean_sizes).unsqueeze(0)
        pos_mean_sizes = torch.sum(one_hot_size_targets * mean_sizes, 1)
        size_res_targets /= pos_mean_sizes

        mask_targets = gt_labels_3d[assignment]

        return (size_class_targets, size_res_targets, dir_class_targets, dir_res_targets,
                center_targets, mask_targets.long(), objectness_targets, objectness_masks,
                pos_weights)

    def get_vote_objectness_and_pos_weights(self, distance):
        euclidean_distance = torch.sqrt(distance.squeeze(0) + 1e-6)

        objectness_targets = distance.new_zeros(self.num_proposal, dtype=torch.long)
        objectness_targets[
            euclidean_distance < self.train_cfg['pos_distance_thr']] = 1

        objectness_masks = distance.new_zeros(self.num_proposal)
        objectness_masks[
            euclidean_distance < self.train_cfg['pos_distance_thr']] = 1.0
        objectness_masks[
            euclidean_distance > self.train_cfg['neg_distance_thr']] = 1.0
        return objectness_targets, objectness_masks, objectness_targets

    def pad_targets(self, gt_labels_3d, max_gt_num, center_targets, valid_gt_masks):
        for index in range(len(gt_labels_3d)):
            pad_num = max_gt_num - gt_labels_3d[index].shape[0]
            center_targets[index] = F.pad(center_targets[index], (0, 0, 0, pad_num))
            valid_gt_masks[index] = F.pad(valid_gt_masks[index], (0, pad_num))
        return center_targets, valid_gt_masks
