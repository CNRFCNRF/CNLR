import torch
from maskrcnn_benchmark.modeling.cnlr_head.box_coordinates import encode_box_info

def generate_cross_box(proposals, rel_pair_idxs):
    intersection_judgment_0, intersection_judgment_1, \
    inter_x1, inter_y1, inter_x2, inter_y2, \
    x1, y1, x2, y2, x3, y3, x4, y4, \
    cross_judgment_0, cross_judgment_1, central_head_x, central_head_y, central_tail_x, central_tail_y, num_rels \
        = encode_box_info(proposals, rel_pair_idxs)
    y_head = (y2 - y1) / 2
    y_tail = (y4 - y3) / 2
    y_where = y_head - y_tail
    num_rel = intersection_judgment_0.shape[0]

    ones = torch.ones(num_rel, 1).to(torch.device('cuda'))
    zeros = torch.zeros(num_rel, 1).to(torch.device('cuda'))

    intersection_matrix = torch.where(intersection_judgment_0 <= 0, zeros, ones)
    intersection_matrix = torch.where(intersection_judgment_1 <= 0, zeros, intersection_matrix)
    # horizontal intersection
    horizonta_matrix = torch.where(cross_judgment_0 >= 0, intersection_matrix, zeros)
    horizonta_matrix = torch.where(cross_judgment_1 <= 0, horizonta_matrix, zeros)
    # longitudinal intersection
    longitudinal_matrix = torch.where(cross_judgment_1 >= 0, intersection_matrix, zeros)

    # contain
    contain_matrix = torch.where(cross_judgment_0 < 0, intersection_matrix, zeros)
    contain_matrix = torch.where(cross_judgment_1 < 0, contain_matrix, zeros)

    contain_matrix_0 = torch.where(y_where <= 0, contain_matrix, zeros)
    contain_matrix_1 = torch.where(y_where > 0, contain_matrix, zeros)

    longitudinal_boxes_head = longitudinal_matrix * torch.cat(
        (torch.min(inter_x1, x1), torch.min(inter_y1, central_head_y),
         torch.max(inter_x2, x2), torch.max(inter_y2, central_head_y)),
        dim=-1)
    longitudinal_boxes_tail = longitudinal_matrix * torch.cat(
        (torch.min(inter_x1, x3), torch.min(inter_y1, central_tail_y),
         torch.max(inter_x2, x4), torch.max(inter_y2, central_tail_y)),
        dim=-1)

    horizonta_boxes_head = horizonta_matrix * torch.cat(
        (torch.min(inter_x1, central_head_x), torch.min(inter_y1, y1),
         torch.max(inter_x2, central_head_x), torch.max(inter_y2, y2)),
        dim=-1)
    horizonta_boxes_tail = horizonta_matrix * torch.cat(
        (torch.min(inter_x1, central_tail_x), torch.min(inter_y1, y3),
         torch.max(inter_x2, central_tail_x), torch.max(inter_y2, y4)),
        dim=-1)

    contain_boxes_head_0 = contain_matrix_0 * torch.cat(
        (torch.min(inter_x1, x1), torch.min(inter_y1, central_head_y),
         torch.max(inter_x2, x2), torch.max(inter_y2, y2)), dim=-1)
    contain_boxes_tail_0 = contain_matrix_0 * torch.cat((torch.min(inter_x1, x3), torch.min(inter_y1, y3),
                                                         torch.max(inter_x2, x4),
                                                         torch.max(inter_y2, central_tail_y)), dim=-1)

    contain_boxes_head_1 = contain_matrix_1 * torch.cat((torch.min(inter_x1, x1), torch.min(inter_y1, y1),
                                                         torch.max(inter_x2, x2),
                                                         torch.max(inter_y2, central_head_y)), dim=-1)
    contain_boxes_tail_1 = contain_matrix_1 * torch.cat(
        (torch.min(inter_x1, x3), torch.min(inter_y1, central_tail_y),
         torch.max(inter_x2, x4), torch.max(inter_y2, y4)), dim=-1)

    cross_head_boxes = longitudinal_boxes_head + horizonta_boxes_head + contain_boxes_head_0 + contain_boxes_head_1
    cross_tail_boxes = longitudinal_boxes_tail + horizonta_boxes_tail + contain_boxes_tail_0 + contain_boxes_tail_1
    cross_matrix = longitudinal_matrix + horizonta_matrix + contain_matrix_0 + contain_matrix_1
    cross_head_boxes = cross_head_boxes.split(num_rels, dim=0)
    cross_tail_boxes = cross_tail_boxes.split(num_rels, dim=0)
    cross_matrix = cross_matrix.split(num_rels, dim=0)
    cross_id_matrix = []
    cross_boxes_head = []
    cross_boxes_tail = []
    full_id_matrix = []
    for matrix, head_box, tail_box in zip(cross_matrix, cross_head_boxes, cross_tail_boxes):
        _, idx = torch.sort(matrix[:, 0], descending=True, dim=0)
        iddd = torch.nonzero(matrix == 1)
        num = iddd.shape[0]
        cross_id = idx[:num]
        head_box = head_box[cross_id]
        tail_box = tail_box[cross_id]
        # the object id with the intersection
        cross_id_matrix.append(cross_id)
        full_id_matrix.append(idx)
        # cross sub_boxes
        cross_boxes_head.append(head_box)
        cross_boxes_tail.append(tail_box)
    return cross_boxes_head, cross_boxes_tail, cross_matrix, cross_id_matrix, full_id_matrix
