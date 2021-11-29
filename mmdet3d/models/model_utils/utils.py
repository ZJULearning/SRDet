import copy
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from mmcv.cnn import ConvModule

from mmdet3d.ops import build_sa_module, spconv as spconv
from mmdet3d.ops.pointnet_modules.point_fp_module import PointFPModule
from mmdet3d.ops.roiaware_pool3d import RoIAwarePool3d
from mmdet3d.ops import QueryAndGroup
from mmdet3d.core.bbox import bbox_overlaps_3d, DepthInstance3DBoxes


def sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = -1,
        gamma: float = 2,
        reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def pool_roi(points_feat, points_xyz, rois, roi_pooler, rcnn_sp_conv=None):
    roi_feats = []
    for batch_i in range(rois.shape[0]):
        roi_feat = roi_pooler(  # [roi-num, out_x, out_y, out_z, C]
            rois[batch_i], points_xyz[batch_i],
            points_feat[batch_i].transpose(0, 1).contiguous())
        roi_feats.append(roi_feat)
    roi_feats = torch.cat(roi_feats, dim=0)  # [batch * roi-num, out_x, out_y, out_z, C]

    if rcnn_sp_conv is None:
        proposal_feats = roi_feats.permute(0, 4, 1, 2, 3)
    else:
        extend_batch_size = roi_feats.shape[0]
        sparse_shape = roi_feats.shape[1:4]
        sparse_idx = roi_feats.sum(dim=-1).nonzero(as_tuple=False)
        roi_features = roi_feats[sparse_idx[:, 0], sparse_idx[:, 1],
                                 sparse_idx[:, 2], sparse_idx[:, 3]]
        coords = sparse_idx.int()
        roi_features = spconv.SparseConvTensor(roi_features, coords, sparse_shape,
                                               extend_batch_size)
        proposal_feats = rcnn_sp_conv(roi_features).dense()
    return proposal_feats  # (batch * roi-num, C, out_x, out_y, out_z)


def roi_head_match(pred_logits, pred_centers, pred_size_res_norm, targets,
                   class_weight, bbox_weight, center_weight,
                   use_focal=False, use_iou=False, iou_weight=1.0, mean_size=None):
    with torch.no_grad():
        if use_focal:
            out_prob = pred_logits.flatten(0, 1).sigmoid()
        else:
            out_prob = pred_logits.flatten(0, 1).softmax(-1)
        out_center = pred_centers.flatten(0, 1)  # [batch_size * num_queries, 3]
        out_size_res_norm = pred_size_res_norm.flatten(0, 1)

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_cts = torch.cat([v["centers"] for v in targets])
        tgt_sizes_res_norm = torch.cat([v["sizes_res_norm"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if use_focal:
            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox_xyz = torch.cdist(out_center, tgt_cts, p=1)
        # both out_size_res and tgt_sizes are normed!
        cost_bbox_size = torch.cdist(out_size_res_norm, tgt_sizes_res_norm, p=1)
        cost_bbox = cost_bbox_xyz * center_weight + cost_bbox_size

        if use_iou:
            out_dir_res = torch.zeros([out_center.shape[0], 1]).to(out_center.device)
            out_size = out_size_res_norm * mean_size + mean_size
            pred_bbox = torch.cat([out_center, torch.clamp(out_size, 0), out_dir_res], dim=-1)
            tgt_sizes = tgt_sizes_res_norm * mean_size + mean_size
            tgt_dir_res = torch.zeros([tgt_cts.shape[0], 1]).to(out_center.device)
            target_bbox = torch.cat([tgt_cts, tgt_sizes, tgt_dir_res], dim=-1)
            iou_3d = bbox_overlaps_3d(pred_bbox, target_bbox, coordinate='lidar')
            cost_bbox = cost_bbox + iou_3d * iou_weight

        # Final cost matrix
        C = bbox_weight * cost_bbox + class_weight * cost_class

        bs, num_queries = pred_logits.shape[:2]
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["labels"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices]


def roi_head_matchv2(pred_logits, pred_centers, pred_size_res_norm, pred_angles, targets,
                     class_weight, bbox_weight, center_weight, use_focal=False,
                     use_iou=False, iou_weight=1.0, mean_size=None, with_roi=False,
                     feat_points=None, point_weight=1.):
    with torch.no_grad():
        if use_focal:
            out_prob = pred_logits.flatten(0, 1).sigmoid()
        else:
            out_prob = pred_logits.flatten(0, 1).softmax(-1)
        out_center = pred_centers.flatten(0, 1)  # [batch_size * num_queries, 3]
        out_size_res_norm = pred_size_res_norm.flatten(0, 1)
        out_angle = pred_angles.flatten(0, 1)

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_cts = torch.cat([v["centers"] for v in targets])
        tgt_sizes_res_norm = torch.cat([v["sizes_res_norm"] for v in targets])
        tgt_angles = torch.cat([v["angles"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        # Compute the classification cost.
        if use_focal:
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox_xyz = torch.cdist(out_center, tgt_cts, p=1)
        # both out_size_res and tgt_sizes are normed!
        cost_bbox_size = torch.cdist(out_size_res_norm, tgt_sizes_res_norm, p=1)
        cost_bbox = cost_bbox_xyz * center_weight + cost_bbox_size

        if use_iou:
            out_size = out_size_res_norm * mean_size + mean_size
            if not with_roi:
                out_angle = torch.zeros([out_center.shape[0], 1]).to(out_center.device)
            pred_bbox = torch.cat([out_center, torch.clamp(out_size, 0), out_angle], dim=-1)
            tgt_sizes = tgt_sizes_res_norm * mean_size + mean_size
            target_bbox = torch.cat([tgt_cts, tgt_sizes, tgt_angles.unsqueeze(-1)], dim=-1)
            iou_3d = bbox_overlaps_3d(pred_bbox, target_bbox, coordinate='lidar')
            cost_iou = -iou_3d
            cost_bbox = cost_bbox + cost_iou * iou_weight

        bs, num_queries = pred_logits.shape[:2]
        sizes = [len(v["labels"]) for v in targets]
        if feat_points is not None:
            assert use_iou
            pred_bbox = pred_bbox.view(bs, -1, 7)
            target_bbox = target_bbox.split(sizes, 0)
            cost_points = torch.zeros_like(cost_bbox)
            px, py = 0, 0
            for i in range(bs):
                pred_bbox3d = DepthInstance3DBoxes(pred_bbox[i], origin=(0.5, 0.5, 0.5))
                target_bbox3d = DepthInstance3DBoxes(target_bbox[i], origin=(0.5, 0.5, 0.5))
                p = feat_points[i]
                cost_point = -torch.mm(pred_bbox3d.points_in_boxes(p).transpose(0, 1).float(),
                                       target_bbox3d.points_in_boxes(p).float())
                dx, dy = cost_point.shape
                cost_points[px:px + dx, py:py + dy] = cost_point
                px += dx
                py += dy
            cost_bbox += cost_points * point_weight

        # Final cost matrix
        C = bbox_weight * cost_bbox + class_weight * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices]


def get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


class DynamicConv(nn.Module):

    def __init__(self,
                 hidden_dim=256,
                 dim_dynamic=64,
                 num_dynamic=2,
                 pooler_resolution=7):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dim_dynamic = dim_dynamic
        self.num_dynamic = num_dynamic
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        if isinstance(pooler_resolution, (list, tuple)):
            pooler_grid_num = sum(p ** 3 for p in pooler_resolution)
        else:
            pooler_grid_num = pooler_resolution ** 3

        num_output = self.hidden_dim * pooler_grid_num
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features, **kwargs):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (7*7*7, N * nr_boxes, self.d_model)
        '''
        features = roi_features.permute(1, 0, 2)  # (N*roi-num, 7*7*7, C)
        # (N*roi-num, 1, 2*hidden-dim*dim-dynamic)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features, None


class DynamicConvOneSimple(nn.Module):

    def __init__(self,
                 hidden_dim=256,
                 pooler_resolution=7,
                 pooler_grid_num=None,
                 proposal_feat_dim=None,
                 use_bmm=False,
                 remove_inter=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.proposal_feat_dim = proposal_feat_dim
        self.use_bmm = use_bmm
        self.remove_inter = remove_inter

        if self.proposal_feat_dim is None:
            self.proposal_feat_dim = self.hidden_dim

        if not self.remove_inter:
            dynamic_out = self.hidden_dim * self.hidden_dim if self.use_bmm else self.hidden_dim
            self.dynamic_layer = nn.Linear(self.proposal_feat_dim, dynamic_out)

        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.activation = nn.ReLU(inplace=True)

        if pooler_grid_num is None:
            if isinstance(pooler_resolution, (list, tuple)):
                pooler_grid_num = sum(p ** 3 for p in pooler_resolution)
            else:
                pooler_grid_num = pooler_resolution ** 3

        num_output = self.hidden_dim * pooler_grid_num
        self.out_layer = nn.Linear(num_output, self.proposal_feat_dim)
        self.norm2 = nn.LayerNorm(self.proposal_feat_dim)

    def forward(self, pro_features, roi_features, **kwargs):
        '''
        pro_features: (1,  N * nr_boxes, proposal-feat-dim)
        roi_features: (7*7*7, N * nr_boxes, C)
        '''
        features = roi_features.permute(1, 0, 2)  # (N*roi-num, 7*7*7, C)
        if not self.remove_inter:
            # (N*roi-num, 1, 2*hidden-dim*dim-dynamic)
            pro_features = self.dynamic_layer(pro_features)
            if self.use_bmm:
                pro_features = pro_features.view(-1, self.hidden_dim, self.hidden_dim)
                features = torch.bmm(features, pro_features)
            else:
                pro_features = pro_features.view(-1, 1, self.hidden_dim)
                features = features * pro_features

        mid_features = features
        features = self.norm1(features)
        features = self.activation(features)  # (N*roi-num, 7*7*7, C)

        features = features.flatten(1)
        features = self.out_layer(features)  # (N*roi-num, proposal-feat-dim)

        features = self.norm2(features)
        features = self.activation(features)

        return features, mid_features


class DynamicMultiConvOneSimple(nn.Module):

    def __init__(self,
                 hidden_dim=256,
                 dim_dynamic=64,
                 num_dynamic=2,
                 pooler_resolution=7,
                 roi_out=[3, 5],
                 proposal_feat_dim=None,
                 use_interact=True,
                 use_bmm=False,
                 share_norm=False,
                 share_with_act=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.roi_out = roi_out
        self.proposal_feat_dim = proposal_feat_dim
        self.use_interact = use_interact
        self.use_bmm = use_bmm
        self.share_norm = share_norm
        self.share_with_act = share_with_act

        if self.proposal_feat_dim is None:
            self.proposal_feat_dim = self.hidden_dim

        if self.use_interact:
            dynamic_out = self.hidden_dim * self.hidden_dim if self.use_bmm else self.hidden_dim
            self.dynamic_layer = nn.Linear(self.proposal_feat_dim, dynamic_out)

        self.activation = nn.ReLU(inplace=True)

        self.norm1 = nn.ModuleList()
        self.norm2 = nn.ModuleList()
        self.out_layer = nn.ModuleList()
        for roi_size in self.roi_out:
            self.norm1.append(nn.LayerNorm(self.hidden_dim))
            num_output = self.hidden_dim * roi_size ** 3
            self.out_layer.append(nn.Linear(num_output, self.proposal_feat_dim))
            self.norm2.append(nn.LayerNorm(self.proposal_feat_dim))

        if self.share_norm:
            self.norm2 = nn.LayerNorm(self.proposal_feat_dim)

    def forward(self, pro_features, roi_features, **kwargs):
        '''
        pro_features: (1,  N * nr_boxes, proposal-feat-dim)
        roi_features: (5*5*5 + 7*7*7, N * nr_boxes, C)
        '''
        roi_feats = []
        roi_grid_num_last = 0
        roi_grid_num = 0
        for roi_size in self.roi_out:
            roi_grid_num += roi_size ** 3
            roi_feats.append(roi_features[roi_grid_num_last:roi_grid_num])
            roi_grid_num_last = roi_grid_num

        # (N*roi-num, 1, 2*hidden-dim*dim-dynamic)
        if self.use_interact:
            pro_features = self.dynamic_layer(pro_features)
        feats = []
        for i, roi_feat in enumerate(roi_feats):
            feat = roi_feat.permute(1, 0, 2)  # (N*roi-num, 5*5*5, C)
            if self.use_interact:
                if self.use_bmm:
                    pro_features = pro_features.view(-1, self.hidden_dim, self.hidden_dim)
                    m_feat = torch.bmm(feat, pro_features)
                else:
                    pro_features = pro_features.view(-1, 1, self.hidden_dim)
                    m_feat = feat * pro_features
            else:
                m_feat = feat

            features = self.norm1[i](m_feat)
            features = self.activation(features)  # (N*roi-num, 7*7*7, C)

            features = features.flatten(1)
            features = self.out_layer[i](features)  # (N*roi-num, proposal-feat-dim)

            if not self.share_norm:
                features = self.norm2[i](features)
                features = self.activation(features)
            feats.append(features)

        features = sum(feats)
        if self.share_norm:
            features = self.norm2(features)
            if self.share_with_act:
                features = self.activation(features)

        return features, None


class DynamicConvOne(nn.Module):

    def __init__(self,
                 hidden_dim=256,
                 pooler_resolution=7,
                 simple_dy=False,
                 dy_add=False,
                 simple_out=False,
                 no_norm1=False,
                 no_norm2=False,
                 cls_mult=False,
                 cls_add=False,
                 cls_multd5=False,
                 use_dy_layer=True,
                 dy_use_norm=False,
                 dy_use_norm_act_lin=False,
                 proposal_feat_dim=None):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.simple_dy = simple_dy
        self.dy_add = dy_add
        if self.dy_add:
            assert self.simple_dy
        self.simple_out = simple_out
        self.no_norm1 = no_norm1
        self.no_norm2 = no_norm2
        self.cls_mult = cls_mult
        self.cls_add = cls_add
        self.cls_multd5 = cls_multd5
        self.use_dy_layer = use_dy_layer
        self.dy_use_norm = dy_use_norm
        self.dy_use_norm_act_lin = dy_use_norm_act_lin
        self.proposal_feat_dim = proposal_feat_dim

        if self.proposal_feat_dim:
            self.reduction = nn.Linear(self.proposal_feat_dim, self.hidden_dim)
        else:
            self.proposal_feat_dim = self.hidden_dim

        if not self.use_dy_layer or self.dy_use_norm or self.dy_use_norm_act_lin:
            assert self.simple_dy

        if not self.simple_dy:
            self.dynamic_layer = nn.Linear(self.hidden_dim, self.hidden_dim * self.hidden_dim)
        elif self.use_dy_layer:
            if self.dy_use_norm:
                self.dynamic_layer = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim))
            elif self.dy_use_norm_act_lin:
                self.dynamic_layer = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.hidden_dim, self.hidden_dim))
            else:
                self.dynamic_layer = nn.Linear(self.hidden_dim, self.hidden_dim)

        if not self.no_norm1:
            self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.activation = nn.ReLU(inplace=True)

        if isinstance(pooler_resolution, (list, tuple)):
            pooler_grid_num = sum(p ** 3 for p in pooler_resolution)
        else:
            pooler_grid_num = pooler_resolution ** 3

        if self.simple_out:
            num_output = pooler_grid_num
            self.out_layer = nn.Linear(num_output, 1)
        else:
            num_output = self.hidden_dim * pooler_grid_num
            self.out_layer = nn.Linear(num_output, self.hidden_dim)
        if not self.no_norm2:
            self.norm2 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features, **kwargs):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (7*7*7, N * nr_boxes, self.d_model)
        '''
        features = roi_features.permute(1, 0, 2)  # (N*roi-num, 7*7*7, C)
        # (N*roi-num, 1, 2*hidden-dim*dim-dynamic)
        if self.simple_dy:
            if self.use_dy_layer:
                pro_features = self.dynamic_layer(pro_features).view(-1, 1, self.hidden_dim)
            else:
                pro_features = pro_features.view(-1, 1, self.hidden_dim)
            if self.dy_add:
                features = features + pro_features
            else:
                features = features * pro_features
                if self.cls_mult or self.cls_add or self.cls_multd5:
                    seeds_cls_feat = kwargs.get('seeds_cls_feat', None)
                    assert seeds_cls_feat is not None
                    seeds_cls_feat = seeds_cls_feat.transpose(0, 1)  # (N*roi-num, 7*7*7, 1)
                    if self.cls_mult:
                        features = features * seeds_cls_feat
                    elif self.cls_add:
                        features = features + seeds_cls_feat
                    elif self.cls_multd5:
                        features = features * (seeds_cls_feat + 0.5)

        else:
            parameters = self.dynamic_layer(pro_features).permute(1, 0, 2).view(
                -1, self.hidden_dim, self.hidden_dim)  # (N*nr_boxes, C, C)
            features = torch.bmm(features, parameters)
        mid_features = features
        if not self.no_norm1:
            features = self.norm1(features)
        features = self.activation(features)  # (N*roi-num, 7*7*7, C)

        if self.simple_out:
            features = features.transpose(1, 2)
            features = self.out_layer(features).squeeze(-1)
        else:
            features = features.flatten(1)
            features = self.out_layer(features)

        if not self.no_norm2:
            features = self.norm2(features)
        features = self.activation(features)

        return features, mid_features


class DynamicConvOneAware(nn.Module):

    def __init__(self,
                 hidden_dim=256,
                 pooler_resolution=5,
                 proposal_feat_dim=None):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.pooler_resolution = pooler_resolution
        self.proposal_feat_dim = proposal_feat_dim

        if self.proposal_feat_dim is None:
            self.proposal_feat_dim = self.hidden_dim

        if isinstance(pooler_resolution, (list, tuple)):
            pooler_grid_num = sum(p ** 3 for p in pooler_resolution)
        else:
            pooler_grid_num = pooler_resolution ** 3
        self.pooler_grid_num = pooler_grid_num

        num_output = self.hidden_dim * pooler_grid_num
        self.dynamic_layer = nn.Linear(self.proposal_feat_dim, num_output)

        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.activation = nn.ReLU(inplace=True)

        self.out_layer = nn.Linear(num_output, self.proposal_feat_dim)
        self.norm2 = nn.LayerNorm(self.proposal_feat_dim)

    def forward(self, pro_features, roi_features, **kwargs):
        '''
        pro_features: (1, N * nr_boxes, proposal-feat-dim)
        roi_features: (5*5*5, N * nr_boxes, C)
        '''
        pro_features = self.dynamic_layer(pro_features)
        pro_features = pro_features.view(-1, self.hidden_dim,  # (5*5*5, N * nr_boxes, C)
                                         self.pooler_grid_num).permute(2, 0, 1)

        features = roi_features * pro_features
        mid_features = features

        features = self.norm1(features)
        features = self.activation(features)  # (5*5*5, N*nr_boxes, C)

        features = self.out_layer(features.transpose(0, 1).flatten(1))

        features = self.norm2(features)  # (N*roi-num, proposal-feat-dim)
        features = self.activation(features)

        return features, mid_features


class DynamicConvMHA(nn.Module):

    def __init__(self,
                 hidden_dim=256,
                 nhead=4,
                 dropout=0.,
                 use_normact=True,
                 dropoutnorm=False,
                 plus_pro_feat=False,
                 use_roi_att=False,
                 roi_dropoutnorm=False,
                 use_ldn=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_normact = use_normact
        self.dropoutnorm = dropoutnorm
        self.plus_pro_feat = plus_pro_feat
        self.use_roi_att = use_roi_att
        self.roi_dropoutnorm = roi_dropoutnorm
        self.use_ldn = use_ldn

        if self.use_roi_att:
            self.roi_att = nn.MultiheadAttention(self.hidden_dim, nhead, dropout=dropout)
            if self.roi_dropoutnorm:
                self.roi_dropout = nn.Dropout(dropout)
                self.roi_norm = nn.LayerNorm(self.hidden_dim)

        self.dynamic_layer = nn.MultiheadAttention(self.hidden_dim, nhead, dropout=dropout)
        if self.use_normact:
            self.norm = nn.LayerNorm(self.hidden_dim)
            self.activation = nn.ReLU(inplace=True)
        elif self.dropoutnorm:
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(self.hidden_dim)

        if self.use_ldn:
            self.linear1 = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.dropout1 = nn.Dropout(dropout)
            self.norm1 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features, **kwargs):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (7*7*7, N * nr_boxes, self.d_model)
        '''
        if self.use_roi_att:
            roi_features2 = self.roi_att(roi_features, roi_features, roi_features)[0]
            if self.roi_dropoutnorm:
                roi_features2 = roi_features + self.roi_dropout(roi_features2)
                roi_features2 = self.roi_norm(roi_features2)
            roi_features = roi_features2
        features = self.dynamic_layer(pro_features, roi_features, roi_features)[0]
        if self.use_normact:
            features = self.norm(features)
            features = self.activation(features)
        elif self.dropoutnorm:
            features = self.dropout(features)
            features = self.norm(features)
        if self.plus_pro_feat:
            features = features + pro_features

        if self.use_ldn:
            features2 = self.linear1(features)
            features2 = features + self.dropout1(features2)
            features = self.norm1(features2)

        return features.squeeze(0), None


class DynamicConvCLSMHA(nn.Module):

    def __init__(self,
                 hidden_dim=256,
                 nhead=4,
                 dropout=0.,
                 dropoutnorm=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropoutnorm = dropoutnorm

        self.dynamic_layer = nn.MultiheadAttention(self.hidden_dim, nhead, dropout=dropout)
        if self.dropoutnorm:
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features, **kwargs):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (7*7*7, N * nr_boxes, self.d_model)
        '''
        ex_features = torch.cat([pro_features, roi_features], dim=0)
        features = self.dynamic_layer(ex_features, ex_features, ex_features)[0][[0]]
        if self.dropoutnorm:
            features = self.dropout(features)
            features = self.norm(features)

        return features.squeeze(0), None


class MultiGroupMLPMerge(nn.Module):

    def __init__(self,
                 proposal_dim=256,
                 int_mlp_group=1,
                 roi_out=[3, 5]):
        super().__init__()
        self.proposal_dim = proposal_dim
        self.int_mlp_group = int_mlp_group
        self.roi_out = roi_out

        assert self.proposal_dim % self.int_mlp_group == 0
        dim_per_group = self.proposal_dim // self.int_mlp_group
        self.roi_feat_mlp = nn.ModuleList()
        for roi_size in self.roi_out:
            inc = roi_size ** 3 * dim_per_group
            outc = dim_per_group
            self.roi_feat_mlp.append(nn.Linear(inc, outc))

    def forward(self, pro_features, roi_features, **kwargs):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (5*5*5 + 7*7*7, N * nr_boxes, self.d_model)
        '''
        roi_feats = []
        roi_grid_num_last = 0
        roi_grid_num = 0
        for roi_size in self.roi_out:
            roi_grid_num += roi_size ** 3
            roi_feats.append(roi_features[roi_grid_num_last:roi_grid_num])
            roi_grid_num_last = roi_grid_num

        roi_feats_merge = []
        n = roi_features.shape[1]
        for roi_feat, mlp in zip(roi_feats, self.roi_feat_mlp):
            rf = roi_feat.permute(1, 2, 0).contiguous().view(n, self.int_mlp_group, -1)
            roi_feats_merge.append(mlp(rf).view(1, n, -1))

        pro_features = sum(roi_feats_merge) + pro_features
        return pro_features, None


class MultiConv3dMerge(nn.Module):

    def __init__(self,
                 proposal_dim=256,
                 roi_out=[3, 5],
                 use_ln=False):
        super().__init__()
        # assert list(roi_out) == [3] or list(roi_out) == [3, 5] or list(roi_out) == [1, 3, 5]
        self.proposal_dim = proposal_dim
        self.roi_out = roi_out
        self.use_ln = use_ln

        self.roi_feat_conv = nn.ModuleList()
        self.roi_feat_norm = nn.ModuleList()
        for roi_size in self.roi_out:
            if roi_size == 1:
                self.roi_feat_conv.append(nn.Conv3d(proposal_dim, proposal_dim, 1, 1, 0))
            else:
                self.roi_feat_conv.append(nn.Conv3d(proposal_dim, proposal_dim, 3, 1, 0))
            self.roi_feat_norm.append(nn.LayerNorm(proposal_dim))

    def forward(self, pro_features, roi_features, **kwargs):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (5*5*5 + 7*7*7, N * nr_boxes, self.d_model)
        '''
        C = roi_features.shape[-1]
        roi_feats = []
        roi_grid_num_last = 0
        roi_grid_num = 0
        for roi_size in self.roi_out:
            roi_grid_num += roi_size ** 3
            roi_feats.append(
                roi_features[roi_grid_num_last:roi_grid_num].permute(1, 2, 0).contiguous().view(
                    -1, C, roi_size, roi_size, roi_size))  # (N*nr_boxes, C, 5, 5, 5)
            roi_grid_num_last = roi_grid_num

        pre_feat = None
        for i, (roi_feat, conv, roi_size) in enumerate(zip(
                roi_feats[::-1], self.roi_feat_conv[::-1], self.roi_out[::-1])):
            # 5x5x5, 3x3x3 or 5x5x5, 3x3x3, 1x1x1
            if pre_feat is not None:
                feat = roi_feat + pre_feat
            else:
                feat = roi_feat
            feat = conv(feat)
            if self.use_ln:
                feat = feat.permute(0, 2, 3, 4, 1)
                feat = self.roi_feat_norm[i](feat).permute(0, 4, 1, 2, 3)
            pre_feat = feat

        feat = feat.flatten(2).permute(2, 0, 1)
        pro_features = feat + pro_features
        return pro_features, None


class ROIConv(nn.Module):

    def __init__(self,
                 in_channels=256,
                 conv_channels=(128, 128),
                 connect_from=None,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU')):
        super(ROIConv, self).__init__()
        if connect_from is not None:
            ex_conv_channels = [in_channels] + list(conv_channels)
            assert ex_conv_channels[connect_from] == ex_conv_channels[-1]

        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.connect_from = connect_from

        self.roi_conv_list = nn.ModuleList()
        for k in range(len(self.conv_channels)):
            self.roi_conv_list.append(
                ConvModulev2(
                    in_channels,
                    self.conv_channels[k],
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    bias=True,
                    inplace=True))
            in_channels = self.conv_channels[k]

    def forward(self, x):
        feats = [x]
        for conv in self.roi_conv_list:
            in_feat = feats[-1]
            feats.append(conv(in_feat))

        if self.connect_from is None:
            return feats[-1]
        return feats[-1] + feats[self.connect_from]


class RoIAggPool3d(nn.Module):

    def __init__(self,
                 use_agg=True,
                 use_pre_conv=True,
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
                 roi_pooler='RoIPool3d',
                 roi_pool_cfg=dict(
                     out_size=5,
                     max_pts_per_voxel=64,
                     mode='max')):
        super(RoIAggPool3d, self).__init__()
        self.use_agg = use_agg
        self.use_pre_conv = use_pre_conv

        if self.use_agg:
            self.grouper = QueryAndGroup(
                radii,
                sample_num,
                min_radius=0,
                use_xyz=True,
                normalize_xyz=normalize_xyz)

            mlp_spec = copy.deepcopy(mlp_channels)
            mlp_spec[0] += 3

            self.mlp = nn.Sequential()
            for i in range(len(mlp_spec) - 1):
                self.mlp.add_module(
                    f'layer{i}',
                    ConvModulev2(
                        mlp_spec[i],
                        mlp_spec[i + 1],
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        conv_cfg=dict(type='Conv2d'),
                        norm_cfg=norm_cfg,
                        bias=bias))
        if self.use_pre_conv:
            self.pre_roi_conv = ROIConv(**pre_roi_conv_cfg)
        self.roi_pooler = eval(roi_pooler)(**roi_pool_cfg)

    def forward(self, points_xyz, features, rois):
        if self.use_agg:
            # (B, C, num_point, nsample)
            new_features = self.grouper(points_xyz, points_xyz, features)
            # (B, mlp[-1], num_point, nsample)
            new_features = self.mlp(new_features)
            # (B, mlp[-1], num_point, 1)
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
            features = new_features.squeeze(-1)  # (B, C, num_point)

        if self.use_pre_conv:
            features = self.pre_roi_conv(features)
        # [batch * roi-num, C, out_x*out_y*out_z]
        roi_feats = self.roi_pooler(points_xyz, features, rois)
        return features, roi_feats


class RoIPool3d(nn.Module):

    def __init__(self,
                 out_size=5,
                 max_pts_per_voxel=64,
                 mode='max'):
        super(RoIPool3d, self).__init__()
        self.roi_pooler = RoIAwarePool3d(out_size, max_pts_per_voxel, mode)

    def forward(self, points_xyz, features, rois):
        roi_feats = []
        for batch_i in range(rois.shape[0]):
            roi_feat = self.roi_pooler(  # [roi-num, out_x, out_y, out_z, C]
                rois[batch_i], points_xyz[batch_i],
                features[batch_i].transpose(0, 1).contiguous())
            roi_feats.append(roi_feat)
        roi_feats = torch.cat(roi_feats, dim=0)  # [batch * roi-num, out_x, out_y, out_z, C]
        # [batch * roi-num, C, out_x*out_y*out_z]
        roi_feats = roi_feats.permute(0, 4, 1, 2, 3).flatten(2)
        return roi_feats


class MultiRoIPool3d(nn.Module):

    def __init__(self,
                 out_size=(3, 5),
                 max_pts_per_voxel=(96, 96),
                 mode='max'):
        super(MultiRoIPool3d, self).__init__()
        assert len(out_size) == len(max_pts_per_voxel)
        self.roi_poolers = nn.ModuleList()
        for o, pts in zip(out_size, max_pts_per_voxel):
            self.roi_poolers.append(RoIAwarePool3d(o, pts, mode))

    def forward(self, points_xyz, features, rois):
        roi_feats = []
        for batch_i in range(rois.shape[0]):
            roi_feat = []
            for pooler in self.roi_poolers:
                roi_f = pooler(  # [roi-num, out_x, out_y, out_z, C]
                    rois[batch_i], points_xyz[batch_i],
                    features[batch_i].transpose(0, 1).contiguous())
                roi_f = roi_f.permute(0, 4, 1, 2, 3).flatten(2)  # [roi-num, C, out_x*out_y*out_z]
                roi_feat.append(roi_f)
            roi_feat = torch.cat(roi_feat, dim=2)  # [roi-num, C, M]
            roi_feats.append(roi_feat)
        roi_feats = torch.cat(roi_feats, dim=0)  # [batch * roi-num, C, M]
        return roi_feats


class RoIFPPool3d(nn.Module):

    def __init__(self,
                 out_size=5,
                 fp_module_cfg=dict(
                     mlp_channels=(128, 128),
                     norm_cfg=dict(type='BN2d')),
                 use_ray=False):
        super(RoIFPPool3d, self).__init__()
        self.out_size = out_size
        # assert self.out_size % 2 == 1
        self.fp_module = PointFPModule(**fp_module_cfg)
        if use_ray:
            self.base_grid = get_ray_grid(self.out_size)
        else:
            self.base_grid = get_grid(self.out_size)

    def forward(self, pts, pts_feature, rois):
        """RoIAwarePool3d module forward.

        Args:
            pts (torch.Tensor): [B, npoints, 3]
            pts_feature (torch.Tensor): [B, C, npoints]
            rois (torch.Tensor): [B, N, 7],in LiDAR coordinate,
                (x, y, z) is the bottom center of rois
        """
        B, roi_num = rois.shape[:2]
        C = pts_feature.shape[1]
        rois[..., 2] = rois[..., 2] + rois[..., 5] / 2.
        base_grid = self.base_grid.to(rois.device)  # (out_x*out_y*out_z, 3)
        grid_rois = []
        flatten_rois = rois.view(-1, 7)
        for roi in flatten_rois:
            grid_roi = base_grid * roi[3:6] + roi[0:3]  # (out_x*out_y*out_z, 3)
            grid_rois.append(grid_roi)
        grid_rois = torch.stack(grid_rois, dim=0)  # (B*roi-num, out_x*out_y*out_z, 3)
        grid_rois = grid_rois.view(B, -1, 3)
        # (B, C, roi-num*out_x*out_y*out_z)
        rois_feats = self.fp_module(grid_rois, pts, None, pts_feature)
        rois_feats = rois_feats.transpose(1, 2).contiguous().view(
            B * roi_num, -1, C).transpose(1, 2)
        return rois_feats


class RoISAPool3d(nn.Module):

    def __init__(self,
                 out_size=5,
                 sa_module_cfg=dict(
                     type='PointSAModule',
                     num_point=None,
                     radius=0.3,
                     num_sample=16,
                     mlp_channels=[128, 128, 128, 128],
                     use_xyz=True,
                     normalize_xyz=True),
                 use_ray=False):
        super(RoISAPool3d, self).__init__()
        self.out_size = out_size
        self.use_ray = use_ray
        self.sa_module = build_sa_module(sa_module_cfg)
        if self.use_ray:
            self.base_grid = get_ray_grid(self.out_size)
        else:
            self.base_grid = get_grid(self.out_size)

    def forward(self, pts, pts_feature, rois):
        """RoIAwarePool3d module forward.

        Args:
            pts (torch.Tensor): [B, npoints, 3]
            pts_feature (torch.Tensor): [B, C, npoints]
            rois (torch.Tensor): [B, N, 7],in LiDAR coordinate,
                (x, y, z) is the bottom center of rois
        """
        B, roi_num = rois.shape[:2]
        C = pts_feature.shape[1]
        rois[..., 2] = rois[..., 2] + rois[..., 5] / 2.
        base_grid = self.base_grid.to(rois.device)  # (out_x*out_y*out_z, 3)
        grid_rois = []
        flatten_rois = rois.view(-1, 7)
        for roi in flatten_rois:
            grid_roi = base_grid * roi[3:6] + roi[0:3]  # (out_x*out_y*out_z, 3)
            grid_rois.append(grid_roi)
        grid_rois = torch.stack(grid_rois, dim=0)  # (B*roi-num, out_x*out_y*out_z, 3)
        grid_rois = grid_rois.view(B, -1, 3)  # (B, roi-num*out_x*out_y*out_z, 3)
        # (B, roi-num*out_x*out_y*out_z, C)
        rois_feats = self.sa_module(pts, features=pts_feature, target_xyz=grid_rois)[1]
        rois_feats = rois_feats.view(B * roi_num, -1, C).transpose(1, 2)
        return rois_feats


def get_grid(out_size):
    base_grid_x = torch.arange(0, 1, 1 / out_size) - (out_size - 1) / (2 * out_size)
    base_grid_y = torch.arange(0, 1, 1 / out_size) - (out_size - 1) / (2 * out_size)
    base_grid_z = torch.arange(0, 1, 1 / out_size) - (out_size - 1) / (2 * out_size)
    base_grid_x = base_grid_x.view(-1, 1, 1).repeat(1, out_size, out_size)
    base_grid_y = base_grid_y.view(1, -1, 1).repeat(out_size, 1, out_size)
    base_grid_z = base_grid_z.view(1, 1, -1).repeat(out_size, out_size, 1)
    base_grid = torch.stack([base_grid_x, base_grid_y, base_grid_z], dim=-1).view(-1, 3)
    return base_grid


def get_ray_grid(out_size):
    assert out_size == 13
    face_center = torch.tensor([[0.5, 0, 0], [-0.5, 0, 0], [0, 0.5, 0],
                                [0, -0.5, 0], [0, 0, 0.5], [0, 0, -0.5]])
    base_grid = torch.cat([face_center, face_center / 2., torch.zeros(1, 3, dtype=torch.float)],
                          dim=0)
    return base_grid


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class ConvModulev2(ConvModule):

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                if isinstance(self.norm, nn.LayerNorm):
                    if x.dim() == 3:
                        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
                    elif x.dim() == 4:
                        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                    else:
                        raise NotImplementedError
                else:
                    x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
        return x


class TransPool3d(nn.Module):

    def __init__(self,
                 use_agg=True,
                 use_pre_conv=True,
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
                 roi_pool_cfg=dict(
                     out_size=5,
                     max_pts_per_voxel=64,
                     mode='max')):
        super(RoIAggPool3d, self).__init__()
        self.use_agg = use_agg
        self.use_pre_conv = use_pre_conv

        if self.use_agg:
            self.grouper = QueryAndGroup(
                radii,
                sample_num,
                min_radius=0,
                use_xyz=True,
                normalize_xyz=normalize_xyz)

            mlp_spec = copy.deepcopy(mlp_channels)
            mlp_spec[0] += 3

            self.mlp = nn.Sequential()
            for i in range(len(mlp_spec) - 1):
                self.mlp.add_module(
                    f'layer{i}',
                    ConvModulev2(
                        mlp_spec[i],
                        mlp_spec[i + 1],
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        conv_cfg=dict(type='Conv2d'),
                        norm_cfg=norm_cfg,
                        bias=bias))
        if self.use_pre_conv:
            self.pre_roi_conv = ROIConv(**pre_roi_conv_cfg)
        self.roi_pooler = RoIPool3d(**roi_pool_cfg)

    def forward(self, points_xyz, features, rois):
        if self.use_agg:
            # (B, C, num_point, nsample)
            new_features = self.grouper(points_xyz, points_xyz, features)
            # (B, mlp[-1], num_point, nsample)
            new_features = self.mlp(new_features)
            # (B, mlp[-1], num_point, 1)
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
            features = new_features.squeeze(-1)  # (B, C, num_point)

        if self.use_pre_conv:
            features = self.pre_roi_conv(features)
        roi_feats = self.roi_pooler(points_xyz, features, rois)
        return features, roi_feats
