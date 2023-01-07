import torch

def encode_box_info(proposals, rel_pair_idxs):
    boxes_head = []
    boxes_tail = []
    num_rels = []
    for proposal, pair_idx in zip(proposals, rel_pair_idxs):
        box = proposal.bbox
        box_head = box[pair_idx[:, 0]]
        box_tail = box[pair_idx[:, 1]]
        boxes_head.append(box_head)
        boxes_tail.append(box_tail)
        num_rels.append(box_head.shape[0])
    boxes_head = torch.cat(boxes_head, dim=0)
    boxes_tail = torch.cat(boxes_tail, dim=0)
    union_boxes = torch.cat((boxes_head, boxes_tail), dim=-1)
    x1, y1, x2, y2, x3, y3, x4, y4 = union_boxes.split([1, 1, 1, 1, 1, 1, 1, 1], dim=-1)
    wh_head = union_boxes[:, 2:4] - union_boxes[:, :2] + 1.0
    wh_tail = union_boxes[:, 6:] - union_boxes[:, 4:6] + 1.0
    central_xy_head = union_boxes[:, :2] + 0.5 * wh_head
    central_xy_tail = union_boxes[:, 4:6] + 0.5 * wh_tail
    central_head_x, central_head_y = central_xy_head.split([1, 1], dim=-1)
    central_tail_x, central_tail_y = central_xy_tail.split([1, 1], dim=-1)
    w_head, h_head = wh_head.split([1, 1], dim=-1)
    w_tail, h_tail = wh_tail.split([1, 1], dim=-1)

    inter_x1 = torch.max(x1, x3)
    inter_y1 = torch.max(y1, y3)
    inter_x2 = torch.min(x2, x4)
    inter_y2 = torch.min(y2, y4)

    intersection_judgment_0 = (w_head + w_tail) / 2 - abs(central_head_x - central_tail_x)
    intersection_judgment_1 = (h_head + h_tail) / 2 - abs(central_head_y - central_tail_y)

    x13 = x1 - x3
    x24 = x2 - x4
    y13 = y1 - y3
    y24 = y2 - y4
    cross_judgment_0 = x13 * x24
    cross_judgment_1 = y13 * y24
    return intersection_judgment_0, intersection_judgment_1, \
           inter_x1, inter_y1, inter_x2, inter_y2, \
           x1, y1, x2, y2, x3, y3, x4, y4, \
           cross_judgment_0, cross_judgment_1, \
           central_head_x, central_head_y, central_tail_x, central_tail_y, num_rels
