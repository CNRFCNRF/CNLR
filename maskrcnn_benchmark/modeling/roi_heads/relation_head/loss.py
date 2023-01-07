# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import numpy.random as npr

from maskrcnn_benchmark.layers import smooth_l1_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat


class RelationLossComputation(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
            self,
            attri_on,
            num_attri_cat,
            max_num_attri,
            attribute_sampling,
            attribute_bgfg_ratio,
            use_label_smoothing,
            predicate_proportion,
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        self.attri_on = attri_on
        self.num_attri_cat = num_attri_cat
        self.max_num_attri = max_num_attri
        self.attribute_sampling = attribute_sampling
        self.attribute_bgfg_ratio = attribute_bgfg_ratio
        self.use_label_smoothing = use_label_smoothing
        self.pred_weight = (1.0 / torch.FloatTensor([0.5, ] + predicate_proportion)).cuda()

        if self.use_label_smoothing:
            self.criterion_loss = Label_Smoothing_Regression(e=0.01)
        else:
            self.criterion_loss = nn.CrossEntropyLoss()

    def __call__(self, proposals, rel_labels, relation_logits, refine_logits, cross_id_matrix, original_ctx, cross_preds):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.

        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])

        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """
        if self.attri_on:
            if isinstance(refine_logits[0], (list, tuple)):
                refine_obj_logits, refine_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attri_on = False
                refine_obj_logits = refine_logits
        else:
            refine_obj_logits = refine_logits

        relation_logits = cat(relation_logits, dim=0)
        refine_obj_logits = cat(refine_obj_logits, dim=0)

        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        rel_labels1 = cat(rel_labels, dim=0)

        max_id = torch.argmax(original_ctx, dim=-1)
        hundred_min = -100 * torch.ones(original_ctx.shape[0], 1).to(torch.device("cuda"))
        original_ctx_dists_50 = original_ctx[:, 1:]
        original_ctx_dists_51 = torch.cat((hundred_min, original_ctx_dists_50), dim=-1)
        prediction_score, prediction_label_id = torch.sort(original_ctx_dists_51, dim=-1, descending=True)
        label_a = prediction_label_id[:, 0]
        label_b = prediction_label_id[:, 1]
        correlation_score = prediction_score[:, 1] / prediction_score[:, 0]

        cross_gt_label = []
        for label, id in zip(rel_labels, cross_id_matrix):
            label = label[id]
            cross_gt_label.append(label)
        rank_id = torch.tensor([0, 8, 44, 34, 28, 30, 17, 16, 10, 32, 31, 22, 40, 26, 23, 45, 20, 36,
                                49, 18, 2, 9, 3, 15, 27, 35, 48, 39, 41, 6, 4, 1, 47, 19, 33, 37,
                                43, 25, 21, 42, 12, 14, 38, 11, 46, 50, 24, 29, 5, 13, 7]).to(torch.device("cuda"))
        cross_gt_label = torch.cat(cross_gt_label, dim=0)
        label_a = torch.unsqueeze(label_a, dim=1)
        label_b = torch.unsqueeze(label_b, dim=1)
        max_id = torch.unsqueeze(max_id, dim=1)
        source = torch.unsqueeze(correlation_score, dim=1)

        id1_rank = rank_id[label_a]
        id2_rank = rank_id[label_b]
        rank_source = id1_rank - id2_rank
        cross_gt_label1 = torch.unsqueeze(cross_gt_label, dim=1)
        cross_label = torch.where(rank_source >= 0, label_a, label_b)
        cross_label = torch.where(source - 0.5 >= 0, cross_label, label_a)
        cross_label = torch.where(max_id == 0, cross_gt_label1, cross_label)

        cross_label = torch.squeeze(cross_label, dim=1)
        global_loss_relation = self.criterion_loss(original_ctx, cross_gt_label.long())
        cross_loss_relation = self.criterion_loss(cross_preds, cross_label.long())
        loss_relation = self.criterion_loss(relation_logits, rel_labels1.long())

        loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())

        # The following code is used to calcaulate sampled attribute loss
        if self.attri_on:
            refine_att_logits = cat(refine_att_logits, dim=0)
            fg_attributes = cat([proposal.get_field("attributes") for proposal in proposals], dim=0)

            attribute_targets, fg_attri_idx = self.generate_attributes_target(fg_attributes)
            if float(fg_attri_idx.sum()) > 0:
                # have at least one bbox got fg attributes
                refine_att_logits = refine_att_logits[fg_attri_idx > 0]
                attribute_targets = attribute_targets[fg_attri_idx > 0]
            else:
                refine_att_logits = refine_att_logits[0].view(1, -1)
                attribute_targets = attribute_targets[0].view(1, -1)

            loss_refine_att = self.attribute_loss(refine_att_logits, attribute_targets,
                                                  fg_bg_sample=self.attribute_sampling,
                                                  bg_fg_ratio=self.attribute_bgfg_ratio)
            return loss_relation, (loss_refine_obj, loss_refine_att)
        else:
            return loss_relation, loss_refine_obj, global_loss_relation, cross_loss_relation

    def generate_attributes_target(self, attributes):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        assert self.max_num_attri == attributes.shape[1]
        device = attributes.device
        num_obj = attributes.shape[0]

        fg_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch.zeros((num_obj, self.num_attri_cat), device=device).float()

        for idx in torch.nonzero(fg_attri_idx).squeeze(1).tolist():
            for k in range(self.max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, fg_attri_idx

    def attribute_loss(self, logits, labels, fg_bg_sample=True, bg_fg_ratio=3):
        if fg_bg_sample:
            loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').view(-1)
            fg_loss = loss_matrix[labels.view(-1) > 0]
            bg_loss = loss_matrix[labels.view(-1) <= 0]

            num_fg = fg_loss.shape[0]
            # if there is no fg, add at least one bg
            num_bg = max(int(num_fg * bg_fg_ratio), 1)
            perm = torch.randperm(bg_loss.shape[0], device=bg_loss.device)[:num_bg]
            bg_loss = bg_loss[perm]

            return torch.cat([fg_loss, bg_loss], dim=0).mean()
        else:
            attri_loss = F.binary_cross_entropy_with_logits(logits, labels)
            attri_loss = attri_loss * self.num_attri_cat / 20.0
            return attri_loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        print(target)
        target = target.view(-1)

        logpt = F.log_softmax(input)
        logpt = logpt.index_select(-1, target).diag()
        logpt = logpt.view(-1)
        pt = logpt.exp()

        logpt = logpt * self.alpha * (target > 0).float() + logpt * (1 - self.alpha) * (target <= 0).float()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def make_roi_relation_loss_evaluator(cfg):
    loss_evaluator = RelationLossComputation(
        cfg.MODEL.ATTRIBUTE_ON,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
        cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS,
        cfg.MODEL.ROI_RELATION_HEAD.REL_PROP,
    )

    return loss_evaluator
