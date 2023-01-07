# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.layers import ROIAlign
from maskrcnn_benchmark.modeling.make_layers import make_conv3x3

from .utils import cat


class LevelMapper(object):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """

    def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-6):
        """
        Arguments:
            k_min (int)
            k_max (int)
            canonical_scale (int)
            canonical_level (int)
            eps (float)
        """
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists, pro):
        """
        Arguments:
            boxlists (list[BoxList])
        """
        # Compute level ids
        if pro == 0:
            s = torch.sqrt(cat([boxlist.area() for boxlist in boxlists]))
        else:
            s = torch.sqrt(cat([self.area_box(boxlist) for boxlist in boxlists]))

        # Eqn.(1) in FPN paper
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0 + self.eps))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return target_lvls.to(torch.int64) - self.k_min

    def area_box(self, box):
        TO_REMOVE = 1
        wight = box[:, 2] - box[:, 0] + TO_REMOVE
        height = box[:, 3] - box[:, 1] + TO_REMOVE
        area = wight * height
        # area = (box[:, 2] - box[:, 0] + TO_REMOVE).long() * (box[:, 3] - box[:, 1] + TO_REMOVE).long()

        return area


class Pooler(nn.Module):
    """
    Pooler for Detection with or without FPN.
    It currently hard-code ROIAlign in the implementation,
    but that can be made more generic later on.
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BoxList.
    """

    # NOTE: cat_all_levels is added for relationship detection. We want to concatenate
    # all levels, since detector is fixed in relation detection. Without concatenation
    # if there is any difference among levels, it can not be finetuned anymore.
    def __init__(self, output_size, scales, sampling_ratio, in_channels=512, cat_all_levels=False):
        """
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[float]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super(Pooler, self).__init__()
        poolers = []
        for scale in scales:
            poolers.append(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio
                )
            )
        self.poolers = nn.ModuleList(poolers)
        self.output_size = output_size
        self.cat_all_levels = cat_all_levels
        # get the levels in the feature map by leveraging the fact that the network always
        # downsamples by a factor of 2 at each level.
        lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        self.map_levels = LevelMapper(lvl_min, lvl_max)
        # reduce the channels
        if self.cat_all_levels:
            self.reduce_channel = make_conv3x3(in_channels * len(self.poolers), in_channels, dilation=1, stride=1,
                                               use_relu=True)

    def convert_to_roi_format(self, boxes, head_boxes, tail_boxes, cross):
        if cross != True:
            concat_boxes = cat([b.bbox for b in boxes], dim=0)
            pro_rois = self.rois_feat(boxes, concat_boxes)
            return pro_rois

        else:

            concat_boxes = cat([b.bbox for b in boxes], dim=0)
            pro_rois = self.rois_feat(boxes, concat_boxes)
            concat_up_boxes = cat([b for b in head_boxes], dim=0)
            concat_down_boxes = cat([b for b in tail_boxes], dim=0)
            up_rois = self.rois_feat(head_boxes, concat_up_boxes)
            down_rois = self.rois_feat(tail_boxes, concat_down_boxes)
            return pro_rois, up_rois, down_rois

    def rois_feat(self, boxes, concat_boxes):
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )

        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def forward(self, x, boxes, head_boxes, tail_boxes, cross):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[BoxList]): boxes to be used to perform the pooling operation.
        Returns:
            result (Tensor)
        """
        if cross != True:
            pro_num = 0
            pro_rois = self.convert_to_roi_format(boxes, head_boxes, tail_boxes, cross)
            pro_rois = self.feature(x, boxes, pro_rois, pro_num)
            return pro_rois


        else:

            pro_rois, up_rois, down_rois = self.convert_to_roi_format(boxes, head_boxes, tail_boxes, cross)
            pro_num = 0
            notpro_num = 1
            pro_result = self.feature(x, boxes, pro_rois, pro_num)
            up_result = self.feature(x, head_boxes, up_rois, notpro_num)
            down_result = self.feature(x, tail_boxes, down_rois, notpro_num)
            return pro_result, up_result, down_result

    def feature(self, x, boxes, rois, pro):
        num_levels = len(self.poolers)
        # assert rois.size(0) > 0
        if num_levels == 1:
            return self.poolers[0](x[0], rois)

        levels = self.map_levels(boxes, pro)

        num_rois = len(rois)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        final_channels = num_channels * num_levels if self.cat_all_levels else num_channels
        result = torch.zeros(
            (num_rois, final_channels, output_size, output_size),
            dtype=dtype,
            device=device,
        )
        for level, (per_level_feature, pooler) in enumerate(zip(x, self.poolers)):
            if self.cat_all_levels:
                result[:, level * num_channels:(level + 1) * num_channels, :, :] = pooler(per_level_feature, rois).to(
                    dtype)
            else:
                idx_in_level = torch.nonzero(levels == level).squeeze(1)
                rois_per_level = rois[idx_in_level]
                result[idx_in_level] = pooler(per_level_feature, rois_per_level).to(dtype)
        if self.cat_all_levels:
            result = self.reduce_channel(result)
        return result


def make_pooler(cfg, head_name):
    resolution = cfg.MODEL[head_name].POOLER_RESOLUTION
    scales = cfg.MODEL[head_name].POOLER_SCALES
    sampling_ratio = cfg.MODEL[head_name].POOLER_SAMPLING_RATIO
    pooler = Pooler(
        output_size=(resolution, resolution),
        scales=scales,
        sampling_ratio=sampling_ratio,
    )
    return pooler
