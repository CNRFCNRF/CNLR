# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import make_fc
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.modeling.roi_heads.relation_head.inference import make_roi_relation_post_processor
from .layers import EGNNLayer, GNNLayer
from .pooling import GatedPooling, EdgeGatedPooling, Compute_Source


class GraphEnergyModel(nn.Module):
    '''
    Energy Model class
    This class takes as input two graphs
    The first graph corresponds to the features extracted from the images
    The second graph corresponds to an intialization of the scene graph
    The goal is the leans an energy functin such that the enegry is low for scene graph that match the image
    Parameters:
    -----------
        config: Experiment configuration
        obj_classes: The object classes in the dataset
        rel_classes: All the realtion classes in the dataset
        obj_feature_dim: Dimension to which the object features will be embedded
        rel_feature_dim: Dimension to which the relation features will be embedded
        in_channels: Dimension for features extracted form the detector
        能量模型类
这个类接受两个图形作为输入
第一个图对应于从图像中提取的特征
所述第二图对应于所述场景图的初始化
目标是倾斜一个能量函数，使匹配图像的场景图的能量低
参数:
-----------
配置:实验配置
obj_classes:数据集中的对象类
rel_classes:数据集中的所有关系类
obj_feature_dim:对象特性将嵌入到的维度
rel_feature_dim:关系特性将嵌入到的维度
in_channels:从检测器中提取的特征的维
    '''

    def __init__(self, config, in_channels):

        super(GraphEnergyModel, self).__init__()
        self.config = config

        self.num_obj_classes = config.DATASETS.NUM_OBJ_CLASSES # -1 无
        self.num_rel_classes = config.DATASETS.NUM_REL_CLASSES # -1

        self.obj_embed_dim = self.config.MODEL.ROI_RELATION_HEAD.EBM.OBJ_EMBED_DIM # 512
        self.rel_embed_dim = self.config.MODEL.ROI_RELATION_HEAD.EBM.REL_EMBED_DIM # 512

        self.obj_label_embed_dim = self.config.MODEL.ROI_RELATION_HEAD.EBM.OBJ_EMBED_DIM # 512
        self.rel_label_embed_dim = self.config.MODEL.ROI_RELATION_HEAD.EBM.REL_EMBED_DIM # 512

        self.pooling_dim = self.config.MODEL.ROI_RELATION_HEAD.EBM.POOLING_DIM # 512
        # Obtain the generatin model based on availabel indformation
        if self.config.MODEL.ROI_RELATION_HEAD.USE_GT_BOX: # T
            if self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL: # F
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        ##########################################################################################
        #Embedding layers
        #These layers are required we want to incorporate the bounding box information 
        #Size of the embedding vector
        self.obj_dim = in_channels
        self.rel_dim = in_channels

        #Embedding for the features extracted from detector
        self.obj_emdedding = nn.Linear(self.obj_dim, self.obj_embed_dim)
        self.rel_embeding = nn.Linear(self.rel_dim, self.rel_embed_dim)

        #Embedding for the scene graph representaions
        self.obj_label_embedding = nn.Linear(self.num_obj_classes + 128, self.obj_label_embed_dim) #128 is for the positional encoding which will be appended to the object lables
        self.rel_label_embedding = nn.Linear(self.num_rel_classes, self.rel_label_embed_dim)

        # position embedding
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum= 0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])
        ##########################################################################################

        self.sg_layer = EGNNLayer(self.obj_label_embed_dim, self.rel_label_embed_dim)
        self.im_layer = GNNLayer(self.obj_embed_dim)

        self.sg_pooler = EdgeGatedPooling(self.obj_label_embed_dim, self.rel_label_embed_dim, self.pooling_dim)
        self.im_pooler = GatedPooling(self.obj_embed_dim, self.pooling_dim)

        self.compute_source = Compute_Source(source_input,source_output)
        self.energy = nn.Sequential(
            nn.Linear(self.pooling_dim*2, self.pooling_dim),
            nn.ReLU(), 
            nn.Linear(self.pooling_dim, 1)
        )

        self.post_processor = make_roi_relation_post_processor(config)
    # def get_contiguous_rel_pair_idx(self, rel_pair_idxs, proposals):
    #     '''
    #     This function converts the list of rel_pair_idxs into a single tensor 
    #     For example: if we have a batch with two images and the first images has two objects and the second image has two object then
    #     an exampele rel_pair_idxs would be 
    #         [ [[0,1],[1, 0]] , 
    #           [[0,1],[1,0]]]
    #     This funciont will convert it such that the 
    #      [
    #          [0,1], [1,0], [2,3], [3,2]
    #      ]
    #      This will help us define a single matrix that consist of all the objects and relations from all the images in the batch.
    #      We can then make sure that information does not flow between nodes/edges belonging to different images by make the adjacency matrix accordingly
    #     '''
    #     offset = 0
    #     pair_list = []
    #     for i, proposal in enumerate(proposals): 
    #         pair_list.append(rel_pair_idxs[i] + offset)
    #         offset += len(proposal)
        
    #     return torch.cat(pair_list, dim=0)

    def forward(self, im_graph, scene_graph, bbox):
        
        #Embedding the bounding boxes
        pos_embed = self.pos_embed(bbox)
        
        im_node_states, _ = im_graph.get_states()
        im_adj_matrix = im_graph.pair2matrix()
        im_batch_list = im_graph.get_batch_list()

        #Obtain the states for the image graph
        im_node_states = self.obj_emdedding(im_node_states)

        #Extract sg states form the graph object
        sg_node_states, sg_edge_states = scene_graph.get_states()
        sg_adj_list = scene_graph.get_adj()
        sg_batch_list = scene_graph.get_batch_list()
        sg_edge_batch_list = scene_graph.get_edge_batch_list()

        #Obtain the states of the scene graph
        sg_node_states = self.obj_label_embedding(cat((sg_node_states, pos_embed), -1))
        sg_edge_states = self.rel_label_embedding(sg_edge_states)

        
        sg_edge_states = torch.sparse.FloatTensor(sg_adj_list.t(), sg_edge_states, torch.Size([sg_node_states.shape[0], sg_node_states.shape[0], sg_edge_states.shape[-1]])).to_dense()
        
        #Refine the states of the image graph
        im_node_states = self.im_layer(im_node_states, im_adj_matrix)
        #Refine the states of the scene graph
        
        sg_adj_matrix = scene_graph.pair2matrix()
        sg_node_states, sg_edge_states = self.sg_layer(sg_node_states, sg_edge_states, sg_adj_matrix)
        sg_edge_states = sg_edge_states[sg_adj_list[:,0], sg_adj_list[:,1]]
        
        
        #Pooling the states
        # im_pooled = self.im_pooler(im_node_states, im_batch_list)
        # sg_pooled = self.im_pooler(sg_node_states, sg_batch_list)

        # energy = self.energy(cat((im_pooled, sg_pooled), -1))

        return im_node_states, sg_node_states
