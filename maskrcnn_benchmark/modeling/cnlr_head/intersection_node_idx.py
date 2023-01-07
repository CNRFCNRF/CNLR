import torch

def head_tail_box_info(head_boxes, tail_boxes):
    head_idx_matrix = []
    tail_idx_matrix = []
    for head_box, tail_box in zip(head_boxes, tail_boxes):
        union_boxes = torch.cat((head_box, tail_box), dim=-1)
        rel_num = union_boxes.shape[0]
        wh_head = union_boxes[:, 2:4] - union_boxes[:, :2] + 1.0
        wh_tail = union_boxes[:, 6:] - union_boxes[:, 4:6] + 1.0
        original_xy_head = union_boxes[:, :2] + 0.5 * wh_head
        original_xy_tail = union_boxes[:, 4:6] + 0.5 * wh_tail
        original_head_x, original_head_y = original_xy_head.split([1, 1], dim=-1)
        original_tail_x, original_tail_y = original_xy_tail.split([1, 1], dim=-1)
        w_head, h_head = wh_head.split([1, 1], dim=-1)
        w_tail, h_tail = wh_tail.split([1, 1], dim=-1)

        original_head_x0 = original_head_x.view(rel_num, 1).expand(rel_num, rel_num)
        original_head_x1 = original_head_x.view(1, rel_num).expand(rel_num, rel_num)
        original_head_y0 = original_head_y.view(rel_num, 1).expand(rel_num, rel_num)
        original_head_y1 = original_head_y.view(1, rel_num).expand(rel_num, rel_num)
        w_head0 = w_head.view(rel_num, 1).expand(rel_num, rel_num)
        w_head1 = w_head.view(1, rel_num).expand(rel_num, rel_num)
        h_head0 = h_head.view(rel_num, 1).expand(rel_num, rel_num)
        h_head1 = h_head.view(1, rel_num).expand(rel_num, rel_num)

        original_tail_x0 = original_tail_x.view(rel_num, 1).expand(rel_num, rel_num)
        original_tail_x1 = original_tail_x.view(1, rel_num).expand(rel_num, rel_num)
        original_tail_y0 = original_tail_y.view(rel_num, 1).expand(rel_num, rel_num)
        original_tail_y1 = original_tail_y.view(1, rel_num).expand(rel_num, rel_num)
        w_tail0 = w_tail.view(rel_num, 1).expand(rel_num, rel_num)
        w_tail1 = w_tail.view(1, rel_num).expand(rel_num, rel_num)
        h_tail0 = h_tail.view(rel_num, 1).expand(rel_num, rel_num)
        h_tail1 = h_tail.view(1, rel_num).expand(rel_num, rel_num)

        head_intersection_0 = (w_head0 + w_head1) / 2 - abs(original_head_x0 - original_head_x1)
        head_intersection_1 = (h_head0 + h_head1) / 2 - abs(original_head_y0 - original_head_y1)

        tail_intersection_0 = (w_tail0 + w_tail1) / 2 - abs(original_tail_x0 - original_tail_x1)
        tail_intersection_1 = (h_tail0 + h_tail1) / 2 - abs(original_tail_y0 - original_tail_y1)

        zeros_idx = torch.zeros(rel_num, rel_num).to(torch.device("cuda"))
        ones_idx = torch.ones(rel_num, rel_num).to(torch.device("cuda"))

        head_matrix_idx1 = torch.where(head_intersection_0 > 0, ones_idx, zeros_idx)
        head_matrix_idx2 = torch.where(head_intersection_1 > 0, ones_idx, zeros_idx)
        head_matrix_idx = head_matrix_idx1 * head_matrix_idx2
        tail_matrix_idx1 = torch.where(tail_intersection_0 > 0, ones_idx, zeros_idx)
        tail_matrix_idx2 = torch.where(tail_intersection_1 > 0, ones_idx, zeros_idx)
        tail_matrix_idx = tail_matrix_idx1 * tail_matrix_idx2
        head_idx_matrix.append(head_matrix_idx)
        tail_idx_matrix.append(tail_matrix_idx)
    return head_idx_matrix, tail_idx_matrix
