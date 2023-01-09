import torch
from torch import nn


class CrossNodeGenerate(nn.Module):
    def __init__(self, hidden_dim=4424):
        super(CrossNodeGenerate, self).__init__()
        self.hidden_dim = hidden_dim
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),

        ])

    def forward(self, proposals, cross_head_feature, cross_tail_feature, cross_head_boxes, cross_tail_boxes, obj_embed,
                rel_pair_idxs, cross_idx_matrix):
        head_obj_embed = []
        tail_obj_embed = []
        for i, pair_idx, idx in zip(obj_embed, rel_pair_idxs, cross_idx_matrix):
            head_obj_embed.append(i[pair_idx[:, 0]][idx])
            tail_obj_embed.append(i[pair_idx[:, 1]][idx])
        head_obj_embed = torch.cat(head_obj_embed, dim=0)
        tail_obj_embed = torch.cat(tail_obj_embed, dim=0)
        cross_head_boxes = encode_box(cross_head_boxes, proposals)
        cross_head_boxes_embed = self.pos_embed(cross_head_boxes)
        cross_tail_boxes = encode_box(cross_tail_boxes, proposals)
        cross_tail_boxes_embed = self.pos_embed(cross_tail_boxes)
        # cross sub_nodes
        cross_head_pre_rep = torch.cat((cross_head_feature, head_obj_embed, cross_head_boxes_embed), dim=-1)
        cross_tail_pre_rep = torch.cat((cross_tail_feature, tail_obj_embed, cross_tail_boxes_embed), dim=-1)
        return cross_head_pre_rep, cross_tail_pre_rep


def encode_box(box, proposals):
    """
    encode proposed box information (x1, y1, x2, y2) to
    (cx/wid, cy/hei, w/wid, h/hei, x1/wid, y1/hei, x2/wid, y2/hei, wh/wid*hei)
    """
    assert proposals[0].mode == 'xyxy'
    boxes_info = []
    for i, (proposal, boxes) in enumerate(zip(proposals, box)):
        img_size = proposal.size
        wid = img_size[0]
        hei = img_size[1]
        wh = boxes[:, 2:] - boxes[:, :2] + 1.0
        xy = boxes[:, :2] + 0.5 * wh
        w, h = wh.split([1, 1], dim=-1)
        x, y = xy.split([1, 1], dim=-1)
        x1, y1, x2, y2 = boxes.split([1, 1, 1, 1], dim=-1)
        assert wid * hei != 0
        info = torch.cat([w / wid, h / hei, x / wid, y / hei, x1 / wid, y1 / hei, x2 / wid, y2 / hei,
                          w * h / (wid * hei)], dim=-1).view(-1, 9)
        boxes_info.append(info)

    return torch.cat(boxes_info, dim=0)
