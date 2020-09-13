import torch
import torch.nn as nn
import torch.nn.functional as F

class TextLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def ohem(self, predict, target, train_mask, negative_ratio=3.):
        pos = (target * train_mask).byte()
        neg = ((1 - target) * train_mask).byte()

        n_pos = pos.float().sum()

        if n_pos.item() > 0:
            loss_pos = F.cross_entropy(predict[pos], target[pos], reduction='sum')
            loss_neg = F.cross_entropy(predict[neg], target[neg], reduction='none')
            n_neg = min(int(neg.float().sum().item()), int(negative_ratio * n_pos.float()))
        else:
            loss_pos = 0.
            loss_neg = F.cross_entropy(predict[neg], target[neg], reduction='none')
            n_neg = 100
        loss_neg, _ = torch.topk(loss_neg, n_neg)

        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).float()

    def forward(self, input, tr_mask, tcl_mask, sin_map, cos_map, radii_map, train_mask):
        """
        calculate textsnake loss
        :param input: (Variable), network predict, (BS, 7, H, W)
        :param tr_mask: (Variable), TR target, (BS, H, W)
        :param tcl_mask: (Variable), TCL target, (BS, H, W)
        :param sin_map: (Variable), sin target, (BS, H, W)
        :param cos_map: (Variable), cos target, (BS, H, W)
        :param radii_map: (Variable), radius target, (BS, H, W)
        :param train_mask: (Variable), training mask, (BS, H, W)
        :return: loss_tr, loss_tcl, loss_radii, loss_sin, loss_cos
        """

        tr_pred = input[:, :2].permute(0, 2, 3, 1).contiguous().view(-1, 2)  # (BSxHxW, 2)
        tcl_pred = input[:, 2:4].permute(0, 2, 3, 1).contiguous().view(-1, 2)  # (BSxHxW, 2)
        sin_pred = input[:, 4].contiguous().view(-1)  # (BSxHxW,)
        cos_pred = input[:, 5].contiguous().view(-1)  # (BSxHxW,)

        # regularize sin and cos: sum to 1
        scale = torch.sqrt(1.0 / (sin_pred ** 2 + cos_pred ** 2))
        sin_pred = sin_pred * scale
        cos_pred = cos_pred * scale

        radii_pred = input[:, 6].contiguous().view(-1)  # (BSxHxW,)
        train_mask = train_mask.view(-1)  # (BSxHxW,)

        tr_mask = tr_mask.contiguous().view(-1)
        tcl_mask = tcl_mask.contiguous().view(-1)
        radii_map = radii_map.contiguous().view(-1)
        sin_map = sin_map.contiguous().view(-1)
        cos_map = cos_map.contiguous().view(-1)

        # loss_tr = F.cross_entropy(tr_pred[train_mask], tr_mask[train_mask].long())
        loss_tr = self.ohem(tr_pred, tr_mask.long(), train_mask.long())

        loss_tcl = 0.
        tr_train_mask = train_mask * tr_mask
        if tr_train_mask.sum().item() > 0:
            loss_tcl = F.cross_entropy(tcl_pred[tr_train_mask], tcl_mask[tr_train_mask].long())

        # geometry losses
        loss_radii, loss_sin, loss_cos = 0., 0., 0.
        tcl_train_mask = train_mask * tcl_mask
        if tcl_train_mask.sum().item() > 0:
            ones = radii_map.new(radii_pred[tcl_mask].size()).fill_(1.).float()
            loss_radii = F.smooth_l1_loss(radii_pred[tcl_mask] / radii_map[tcl_mask], ones)
            loss_sin = F.smooth_l1_loss(sin_pred[tcl_mask], sin_map[tcl_mask])
            loss_cos = F.smooth_l1_loss(cos_pred[tcl_mask], cos_map[tcl_mask])

        return loss_tr, loss_tcl, loss_radii, loss_sin, loss_cos