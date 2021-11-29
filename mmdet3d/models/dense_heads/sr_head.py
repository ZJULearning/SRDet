import torch
import copy
from mmcv.runner import force_fp32
from torch import nn as nn

from mmdet.core import build_bbox_coder
from mmdet.models import HEADS
from mmdet3d.core.post_processing import aligned_3d_nms
from mmdet3d.models.model_utils import NoiseSuppressionModule
from .sr_init_head import SRInitHead
from .sr_roi_head import SRRoIHead


@HEADS.register_module()
class SRHead(nn.Module):

    def __init__(self,
                 num_classes,
                 num_proposal,
                 bbox_coder,
                 train_cfg=None,
                 test_cfg=None,
                 vote_module='NoiseSuppressionModule',
                 vote_module_cfg=None,
                 init_head_module='SRInitHead',
                 init_head_cfg=None,
                 roi_head_module='SRRoIHead',
                 roi_head_cfg=None,
                 vote_loss_weight=1.,
                 init_loss_weight=1.,
                 use_nms=False):
        super(SRHead, self).__init__()
        self.num_classes = num_classes
        self.num_proposal = num_proposal
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_sizes = self.bbox_coder.num_sizes
        self.num_dir_bins = self.bbox_coder.num_dir_bins

        self.vote_module = vote_module
        self.vote_module_cfg = vote_module_cfg
        self.init_head_module = init_head_module
        self.init_head_cfg = init_head_cfg
        self.roi_head_module = roi_head_module
        self.roi_head_cfg = roi_head_cfg
        self.vote_loss_weight = vote_loss_weight
        self.init_loss_weight = init_loss_weight
        self.use_nms = use_nms

        self.gt_per_seed = self.vote_module_cfg['gt_per_seed']
        self.vote_module = eval(self.vote_module)(**self.vote_module_cfg)

        self.init_head = eval(self.init_head_module)(
            self.bbox_coder, num_classes, train_cfg, **init_head_cfg)

        self.roi_head = eval(self.roi_head_module)(
            self.bbox_coder, num_classes, num_proposal, train_cfg, **roi_head_cfg)

        self.fp16_enabled = False

    def init_weights(self):
        pass

    def forward(self, feat_dict, sample_mod=None):
        seed_points = feat_dict['fp_xyz'][-1]
        seed_features = feat_dict['fp_features'][-1]
        seed_indices = feat_dict['fp_indices'][-1]

        vote_points, vote_features, vote_preds = self.vote_module(seed_points, seed_features)

        # vote_preds will be used for loss calculation in vote-module
        head_inputs = dict(points_xyz=vote_points, features=vote_features,
                           seeds_xyz=seed_points, seeds_features=seed_features,
                           vote_preds=vote_preds)

        agg_points, agg_features, init_preds = self.init_head(head_inputs)
        head_inputs['aggregated_points'] = agg_points
        head_inputs['aggregated_features'] = agg_features
        proposal_boxes, proposal_feats, init_results = self.init_head.get_init_proposals(
            *init_preds, head_inputs, self.num_proposal)

        roi_preds = self.roi_head(head_inputs, proposal_boxes, proposal_feats)
        results = self.roi_head.split_pred(*roi_preds)

        results.update(init_results)
        results.update(dict(
            seed_points=seed_points,
            seed_indices=seed_indices,
            vote_points=vote_points,
            vote_features=vote_features,
            aggregated_points=agg_points,
            aggregated_features=agg_features,
            vote_preds=vote_preds))

        return results

    @force_fp32(apply_to=('bbox_preds',))
    def loss(self,
             bbox_preds,
             points,
             gt_bboxes_3d,
             gt_labels_3d,
             pts_semantic_mask=None,
             pts_instance_mask=None,
             img_metas=None,
             gt_bboxes_ignore=None,
             ret_target=False):
        """Compute loss.

        Args:
            bbox_preds (dict): Predictions from forward of vote head.
            points (list[torch.Tensor]): Input points.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth \
                bboxes of each sample.
            gt_labels_3d (list[torch.Tensor]): Labels of each sample.
            pts_semantic_mask (None | list[torch.Tensor]): Point-wise
                semantic mask.
            pts_instance_mask (None | list[torch.Tensor]): Point-wise
                instance mask.
            img_metas (list[dict]): Contain pcd and img's meta info.
            gt_bboxes_ignore (None | list[torch.Tensor]): Specify
                which bounding.
            ret_target (Bool): Return targets or not.

        Returns:
            dict: Losses of Votenet.
        """
        vote_targets, init_head_targets, roi_head_targets = self.get_targets(
            points, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask, pts_instance_mask, bbox_preds)

        # calculate vote loss
        vote_loss = self.vote_module.get_loss(bbox_preds, *vote_targets)

        # calculate head loss
        init_losses = self.init_head.get_losses(bbox_preds, *init_head_targets)
        losses = self.roi_head.get_losses(bbox_preds, img_metas, *roi_head_targets)
        for k, v in init_losses.items():
            init_losses[k] = v * self.init_loss_weight
        losses.update(init_losses)
        losses.update(dict(vote_loss=vote_loss * self.vote_loss_weight))

        if ret_target:
            losses['vote_targets'] = vote_targets
            losses['init_head_targets'] = init_head_targets
            losses['roi_head_targets'] = roi_head_targets

        return losses

    def fill_empty_example(self, gt_bboxes_3d, gt_labels_3d):
        # gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth bboxes of each batch.
        # gt_labels_3d (list[torch.Tensor]): Labels of each batch.
        valid_gt_masks = list()
        gt_num = list()
        for batch_index in range(len(gt_labels_3d)):
            if len(gt_labels_3d[batch_index]) == 0:
                # no gt in a sample
                fake_box = gt_bboxes_3d[batch_index].tensor.new_zeros(
                    1, gt_bboxes_3d[batch_index].tensor.shape[-1])
                gt_bboxes_3d[batch_index] = gt_bboxes_3d[batch_index].new_box(fake_box)
                gt_labels_3d[batch_index] = gt_labels_3d[batch_index].new_zeros(1)
                valid_gt_masks.append(gt_labels_3d[batch_index].new_zeros(1))
                gt_num.append(1)
            else:
                valid_gt_masks.append(gt_labels_3d[batch_index].new_ones(
                    gt_labels_3d[batch_index].shape))
                gt_num.append(gt_labels_3d[batch_index].shape[0])
        max_gt_num = max(gt_num)  # max gt num in a batch
        return gt_bboxes_3d, gt_labels_3d, valid_gt_masks, max_gt_num

    def get_targets(self,
                    points,
                    gt_bboxes_3d,
                    gt_labels_3d,
                    pts_semantic_mask=None,
                    pts_instance_mask=None,
                    bbox_preds=None):
        """Generate targets of vote head.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth \
                bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): Labels of each batch.
            pts_semantic_mask (None | list[torch.Tensor]): Point-wise semantic
                label of each batch.
            pts_instance_mask (None | list[torch.Tensor]): Point-wise instance
                label of each batch.
            bbox_preds (torch.Tensor): Bounding box predictions of vote head.

        Returns:
            tuple[torch.Tensor]: Targets of vote head.
        """
        gt_bboxes_3d, gt_labels_3d, valid_gt_masks, max_gt_num = \
            self.fill_empty_example(gt_bboxes_3d, gt_labels_3d)

        if pts_semantic_mask is None:
            pts_semantic_mask = [None for i in range(len(gt_labels_3d))]
            pts_instance_mask = [None for i in range(len(gt_labels_3d))]

        # vote target will not be affected by one-one match
        vote_targets = self.vote_module.get_targets(
            points, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask, pts_instance_mask,
            self.bbox_coder, self.gt_per_seed, self.num_classes)

        init_head_targets = self.init_head.get_targets(
            points, gt_bboxes_3d, gt_labels_3d, max_gt_num, valid_gt_masks, bbox_preds)
        roi_head_targets = self.roi_head.get_targets(
            points, gt_bboxes_3d, gt_labels_3d, bbox_preds)

        return vote_targets, init_head_targets, roi_head_targets

    def get_bboxes(self,
                   points,
                   bbox_preds,
                   input_metas,
                   rescale=False,
                   use_nms=True):
        """Generate bboxes from vote head predictions.

        Args:
            points (torch.Tensor): Input points.
            bbox_preds (dict): Predictions from vote head.
            input_metas (list[dict]): Point cloud and image's meta info.
            rescale (bool): Whether to rescale bboxes.
            use_nms (bool): Whether to apply NMS, skip nms postprocessing
                while using vote head in rpn stage.

        Returns:
            list[tuple[torch.Tensor]]: Bounding boxes, scores and labels.
        """
        # decode boxes
        sem_scores, obj_scores, bbox3d = self.roi_head.decode(bbox_preds)

        if use_nms and self.use_nms:
            batch_size = bbox3d.shape[0]
            results = list()
            for b in range(batch_size):
                bbox_selected, score_selected, labels = \
                    self.multiclass_nms_single(obj_scores[b], sem_scores[b],
                                               bbox3d[b], points[b, ..., :3],
                                               input_metas[b])
                bbox = input_metas[b]['box_type_3d'](
                    bbox_selected,
                    box_dim=bbox_selected.shape[-1],
                    with_yaw=self.bbox_coder.with_rot)
                results.append((bbox, score_selected, labels))
        elif use_nms:
            batch_size = bbox3d.shape[0]
            results = list()
            labels = torch.arange(self.num_classes, device=sem_scores.device). \
                unsqueeze(0).repeat(self.num_proposal, 1).flatten(0, 1)
            for b in range(batch_size):
                scores_per_img, topk_indices = sem_scores[b].flatten().topk(
                    self.num_proposal, sorted=False)
                keep_idx = scores_per_img > self.test_cfg.score_thr
                scores_per_img = scores_per_img[keep_idx]
                topk_indices = topk_indices[keep_idx]

                labels_per_img = labels[topk_indices]
                box_pred_per_img = bbox3d[b].view(-1, 1, 7).repeat(
                    1, self.num_classes, 1).view(-1, 7)[topk_indices]

                bbox = input_metas[b]['box_type_3d'](
                    box_pred_per_img,
                    box_dim=box_pred_per_img.shape[-1],
                    with_yaw=self.bbox_coder.with_rot,
                    origin=(0.5, 0.5, 0.5))
                results.append((bbox, scores_per_img, labels_per_img))

        return results

    def multiclass_nms_single(self, obj_scores, sem_scores, bbox, points,
                              input_meta):
        """Multi-class nms in single batch.

        Args:
            obj_scores (torch.Tensor): Objectness score of bounding boxes.
            sem_scores (torch.Tensor): semantic class score of bounding boxes.
            bbox (torch.Tensor): Predicted bounding boxes.
            points (torch.Tensor): Input points.
            input_meta (dict): Point cloud and image's meta info.

        Returns:
            tuple[torch.Tensor]: Bounding boxes, scores and labels.
        """
        bbox = input_meta['box_type_3d'](
            bbox,
            box_dim=bbox.shape[-1],
            with_yaw=self.bbox_coder.with_rot,
            origin=(0.5, 0.5, 0.5))
        box_indices = bbox.points_in_boxes(points)

        corner3d = bbox.corners
        minmax_box3d = corner3d.new(torch.Size((corner3d.shape[0], 6)))
        minmax_box3d[:, :3] = torch.min(corner3d, dim=1)[0]
        minmax_box3d[:, 3:] = torch.max(corner3d, dim=1)[0]

        nonempty_box_mask = box_indices.T.sum(1) > 5

        bbox_classes = torch.argmax(sem_scores, -1)
        nms_selected = aligned_3d_nms(minmax_box3d[nonempty_box_mask],
                                      obj_scores[nonempty_box_mask],
                                      bbox_classes[nonempty_box_mask],
                                      self.test_cfg.nms_thr)

        # filter empty boxes and boxes with low score
        scores_mask = (obj_scores > self.test_cfg.score_thr)
        nonempty_box_inds = torch.nonzero(
            nonempty_box_mask, as_tuple=False).flatten()
        nonempty_mask = torch.zeros_like(bbox_classes).scatter(
            0, nonempty_box_inds[nms_selected], 1)
        selected = (nonempty_mask.bool() & scores_mask.bool())

        if self.test_cfg.per_class_proposal:
            bbox_selected, score_selected, labels = [], [], []
            for k in range(sem_scores.shape[-1]):
                bbox_selected.append(bbox[selected].tensor)
                score_selected.append(obj_scores[selected] *
                                      sem_scores[selected][:, k])
                labels.append(
                    torch.zeros_like(bbox_classes[selected]).fill_(k))
            bbox_selected = torch.cat(bbox_selected, 0)
            score_selected = torch.cat(score_selected, 0)
            labels = torch.cat(labels, 0)
        else:
            bbox_selected = bbox[selected].tensor
            score_selected = obj_scores[selected]
            labels = bbox_classes[selected]

        return bbox_selected, score_selected, labels
