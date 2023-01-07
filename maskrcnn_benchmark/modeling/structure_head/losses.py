# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT
import torch
from maskrcnn_benchmark.modeling import registry

@registry.ENERGY_LOSS.register("ContrastiveDivergence")
def ContrastiveDivergence(im_node_states , sg_node_states):

    similarity = torch.cosine_similarity(im_node_states, sg_node_states)
    loss_cos = 1 - similarity
    return {'COS Loss (cd)': loss_cos}

# class CosineEmbeddingLoss(_Loss):
#     __constants__ = ['margin', 'reduction']
#     def __init__(self, margin=0., size_average=None, reduce=None, reduction='mean'):
#         super(CosineEmbeddingLoss, self).__init__(size_average, reduce, reduction)
#         self.margin = margin
#     def forward(self, input1, input2, target):
#         return F.cosine_embedding_loss(input1, input2, target, margin=self.margin, reduction=self.reduction)

@registry.ENERGY_LOSS.register("SoftPlus")
def SoftPlus(cfg, positive_energy, negative_energy):

    loss_ml = torch.nn.Softplus()(cfg.ENERGY_MODEL.TEMP*(positive_energy- negative_energy))
    return {'ML Loss (sp)': loss_ml}

@registry.ENERGY_LOSS.register("LogSumExp")
def LogSumExp(cfg, positive_energy, negative_energy):

    negative_energy_reduced = (negative_energy - torch.min(negative_energy))

    coeff = torch.exp(-cfg.ENERGY_MODEL.TEMP*negative_energy_reduced)
    norm_const = torch.sum(coeff) + 1e-4

    pos_term = cfg.ENERGY_MODEL.TEMP* positive_energy
    pos_loss = torch.mean(pos_term)

    neg_loss = coeff * (-cfg.ENERGY_MODEL.TEMP*negative_energy_reduced) / norm_const

    loss_ml = pos_loss + torch.sum(neg_loss)

    return {'ML Loss (lse)': loss_ml}

def build_loss_function(cfg):
    loss_func = registry.ENERGY_LOSS[cfg.ENERGY_MODEL.LOSS]

    return loss_func