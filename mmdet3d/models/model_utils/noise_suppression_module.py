import numpy as np
import torch
from mmcv import is_tuple_of
from torch import nn as nn
from torch.nn import functional as F

from mmdet.core import multi_apply
from mmdet3d.models.builder import build_loss
from mmdet3d.models.model_utils.utils import sigmoid_focal_loss
from .utils import ConvModulev2


class NoiseSuppressionModule(nn.Module):

    def __init__(self,
                 in_channels,
                 vote_per_seed=1,
                 gt_per_seed=3,
                 num_points=-1,
                 conv_channels=(16, 16),
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU'),
                 norm_feats=True,
                 with_res_feat=True,
                 cls_weight=1.,
                 vote_xyz_range=None,
                 vote_loss=None,
                 cls_bg_weight=0.2,):
        super().__init__()

        self.in_channels = in_channels
        assert vote_per_seed == 1
        self.gt_per_seed = gt_per_seed
        self.num_points = num_points
        self.norm_feats = norm_feats
        self.with_res_feat = with_res_feat

        assert vote_xyz_range is None or is_tuple_of(vote_xyz_range, float)
        self.vote_xyz_range = vote_xyz_range

        if vote_loss is not None:
            self.vote_loss = build_loss(vote_loss)

        prev_channels = in_channels
        vote_conv_list = list()
        for k in range(len(conv_channels)):
            vote_conv_list.append(
                ConvModulev2(
                    prev_channels,
                    conv_channels[k],
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    bias=True,
                    inplace=True))
            prev_channels = conv_channels[k]
        self.vote_conv = nn.Sequential(*vote_conv_list)
        self.pred_cls_len = 2

        # conv_out predicts coordinate and residual features
        if with_res_feat:
            out_channel = 3 + self.pred_cls_len + in_channels
        else:
            out_channel = 3 + self.pred_cls_len
        self.conv_out = nn.Conv1d(prev_channels, out_channel, 1)

        self.cls_weight = cls_weight
        empty_weight = torch.ones(2)
        empty_weight[0] = cls_bg_weight
        self.register_buffer('empty_weight', empty_weight)

    def forward(self, seed_points, seed_feats):
        if self.num_points != -1:
            assert self.num_points < seed_points.shape[1], \
                f'Number of vote points ({self.num_points}) should be ' \
                f'smaller than seed points size ({seed_points.shape[1]})'
            seed_points = seed_points[:, :self.num_points]
            seed_feats = seed_feats[..., :self.num_points]

        batch_size, feat_channels, num_seed = seed_feats.shape
        num_vote = num_seed

        x = self.vote_conv(seed_feats)
        # (batch_size, (3+out_dim)*vote_per_seed, num_seed)
        votes = self.conv_out(x)

        votes = votes.transpose(2, 1).view(batch_size, num_seed,
                                           1, -1)

        offset = votes[:, :, :, 0:3]
        cls = votes[:, :, :, 3:3 + self.pred_cls_len]
        cls_norm = torch.softmax(cls, -1)

        if self.vote_xyz_range is not None:
            limited_offset_list = []
            for axis in range(len(self.vote_xyz_range)):
                limited_offset_list.append(offset[..., axis].clamp(
                    min=-self.vote_xyz_range[axis],
                    max=self.vote_xyz_range[axis]))
            offset = torch.stack(limited_offset_list, -1)

        vote_points = (seed_points.unsqueeze(2) + offset * cls_norm[..., [-1]]).contiguous()
        vote_points = vote_points.view(batch_size, num_vote, 3)
        offset = offset.reshape(batch_size, num_vote, 3).transpose(2, 1)
        cls = cls.reshape(batch_size, num_vote, self.pred_cls_len).transpose(2, 1)
        vote_preds = dict(offset=offset, cls=cls)

        if self.with_res_feat:
            res_feats = votes[:, :, :, 3 + self.pred_cls_len:]
            vote_feats = (seed_feats.transpose(2, 1).unsqueeze(2) + res_feats).contiguous()
            vote_feats = vote_feats.view(batch_size, num_vote, feat_channels).transpose(
                2, 1).contiguous()

            if self.norm_feats:
                features_norm = torch.norm(vote_feats, p=2, dim=1)
                vote_feats = vote_feats.div(features_norm.unsqueeze(1))
        else:
            vote_feats = seed_feats
        return vote_points, vote_feats, vote_preds

    def get_targets(self, points, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask,
                    pts_instance_mask, bbox_coder, gt_per_seed, num_classes):
        vote_targets, vote_target_masks, vote_target_ids = multi_apply(
            self.get_targets_single, points, gt_bboxes_3d, gt_labels_3d,
            pts_semantic_mask, pts_instance_mask,
            bbox_coder=bbox_coder, gt_per_seed=gt_per_seed, num_classes=num_classes)

        vote_targets = torch.stack(vote_targets)
        vote_target_masks = torch.stack(vote_target_masks)
        vote_target_ids = torch.stack(vote_target_ids)
        return vote_targets, vote_target_masks, vote_target_ids

    def get_targets_single(self, points, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask,
                           pts_instance_mask, bbox_coder, gt_per_seed, num_classes):
        assert bbox_coder.with_rot or pts_semantic_mask is not None

        gt_bboxes_3d = gt_bboxes_3d.to(points.device)
        num_points = points.shape[0]
        if bbox_coder.with_rot:
            vote_targets = points.new_zeros([num_points, 3 * gt_per_seed])
            vote_target_masks = points.new_zeros([num_points],
                                                 dtype=torch.long)
            vote_target_idx = points.new_zeros([num_points], dtype=torch.long)
            vote_target_ids = points.new_zeros([num_points], dtype=torch.long)
            # box_indices_all, (20000, box num), reflecting that each point is in which box
            box_indices_all = gt_bboxes_3d.points_in_boxes(points)
            for i in range(gt_labels_3d.shape[0]):
                box_indices = box_indices_all[:, i]
                indices = torch.nonzero(
                    box_indices, as_tuple=False).squeeze(-1)
                # points which are in the i-th box
                selected_points = points[indices]
                vote_target_masks[indices] = 1
                vote_targets_tmp = vote_targets[indices]
                votes = gt_bboxes_3d.gravity_center[i].unsqueeze(
                    0) - selected_points[:, :3]

                for j in range(gt_per_seed):
                    column_indices = torch.nonzero(
                        vote_target_idx[indices] == j,
                        as_tuple=False).squeeze(-1)
                    vote_targets_tmp[column_indices,
                    int(j * 3):int(j * 3 +
                                   3)] = votes[column_indices]
                    if j == 0:
                        vote_targets_tmp[column_indices] = votes[
                            column_indices].repeat(1, gt_per_seed)

                vote_targets[indices] = vote_targets_tmp
                vote_target_idx[indices] = torch.clamp(
                    vote_target_idx[indices] + 1, max=2)
                vote_target_ids[indices] = i
        elif pts_semantic_mask is not None:
            vote_targets = points.new_zeros([num_points, 3])
            vote_target_masks = points.new_zeros([num_points], dtype=torch.long)
            vote_target_ids = points.new_zeros([num_points], dtype=torch.long)

            for i in torch.unique(pts_instance_mask):
                indices = torch.nonzero(
                    pts_instance_mask == i, as_tuple=False).squeeze(-1)
                if pts_semantic_mask[indices[0]] < num_classes:
                    selected_points = points[indices, :3]
                    center = 0.5 * (
                            selected_points.min(0)[0] + selected_points.max(0)[0])
                    vote_targets[indices, :] = center - selected_points
                    vote_target_masks[indices] = 1
                    vote_target_ids[indices] = i
            vote_targets = vote_targets.repeat((1, gt_per_seed))
        else:
            raise NotImplementedError
        return vote_targets, vote_target_masks, vote_target_ids

    def get_loss(self, bbox_preds, vote_targets, vote_targets_mask, vote_target_ids):
        """Calculate loss of voting module.

        Args:
            bbox_preds (dict)
            vote_targets (torch.Tensor): Targets of votes.
            vote_targets_mask (torch.Tensor): Mask of valid vote targets.
            vote_target_ids (torch.Tensor)

        Returns:
            torch.Tensor: Weighted vote loss.
        """
        seed_points = bbox_preds['seed_points']
        vote_points = bbox_preds['vote_points']
        seed_indices = bbox_preds['seed_indices']
        vote_points_non_mult = bbox_preds.get('vote_points_non_mult', None)
        vote_cls = bbox_preds['vote_preds']['cls']  # (batch, cls-num, point-num)
        batch_size, num_seed = seed_points.shape[:2]

        # vote_targets_mask is bool, (B, 20000), reflecting whether a point is in boxes
        # seed_indices, (B, 1024), reflecting kept points (or seeds)
        # seed_gt_votes_mask, (B, 1024), reflecting whether a seed is in boxes. For those
        #   seeds in boxes, we'll calc a vote loss
        seed_gt_votes_mask = torch.gather(vote_targets_mask, 1,
                                          seed_indices).float()

        seed_indices_expand = seed_indices.unsqueeze(-1).repeat(
            1, 1, 3 * self.gt_per_seed)
        seed_gt_votes = torch.gather(vote_targets, 1, seed_indices_expand)
        seed_gt_votes += seed_points.repeat(1, 1, self.gt_per_seed)

        weight = seed_gt_votes_mask / (torch.sum(seed_gt_votes_mask) + 1e-6)
        distance = self.vote_loss(
            vote_points.view(batch_size * num_seed, -1, 3),
            seed_gt_votes.view(batch_size * num_seed, -1, 3),
            dst_weight=weight.view(batch_size * num_seed, 1))[1]
        vote_loss = torch.sum(torch.min(distance, dim=1)[0])

        if vote_points_non_mult is not None:
            distance = self.vote_loss(
                vote_points_non_mult.view(batch_size * num_seed, -1, 3),
                seed_gt_votes.view(batch_size * num_seed, -1, 3),
                dst_weight=weight.view(batch_size * num_seed, 1))[1]
            vote_loss = vote_loss / 2. + torch.sum(torch.min(distance, dim=1)[0]) / 2.

        cls_target = seed_gt_votes_mask.long()
        cls_loss = F.cross_entropy(vote_cls, cls_target, self.empty_weight)
        vote_loss = vote_loss + cls_loss * self.cls_weight

        return vote_loss
