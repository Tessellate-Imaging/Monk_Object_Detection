import torch
import numpy as np
import cv2
import os
from util.config import config as cfg


def visualize_network_output(output, tr_mask, tcl_mask, mode='train'):

    vis_dir = os.path.join(cfg.vis_dir, cfg.exp_name + '_' + mode)
    if not os.path.exists(vis_dir):
        os.mkdir(vis_dir)

    tr_pred = output[:, :2]
    tr_score, tr_predict = tr_pred.max(dim=1)

    tcl_pred = output[:, 2:4]
    tcl_score, tcl_predict = tcl_pred.max(dim=1)

    tr_predict = tr_predict.cpu().numpy()
    tcl_predict = tcl_predict.cpu().numpy()

    tr_target = tr_mask.cpu().numpy()
    tcl_target = tcl_mask.cpu().numpy()

    for i in range(len(tr_pred)):
        tr_pred = (tr_predict[i] * 255).astype(np.uint8)
        tr_targ = (tr_target[i] * 255).astype(np.uint8)

        tcl_pred = (tcl_predict[i] * 255).astype(np.uint8)
        tcl_targ = (tcl_target[i] * 255).astype(np.uint8)

        tr_show = np.concatenate([tr_pred, tr_targ], axis=1)
        tcl_show = np.concatenate([tcl_pred, tcl_targ], axis=1)
        show = np.concatenate([tr_show, tcl_show], axis=0)
        show = cv2.resize(show, (512, 512))

        path = os.path.join(vis_dir, '{}.png'.format(i))
        cv2.imwrite(path, show)


def visualize_detection(image, contours, tr=None, tcl=None):
    image_show = image.copy()
    image_show = np.ascontiguousarray(image_show[:, :, ::-1])
    image_show = cv2.polylines(image_show, contours, True, (0, 0, 255), 3)

    if (tr is not None) and (tcl is not None):
        tr = (tr > cfg.tr_thresh).astype(np.uint8)
        tcl = (tcl > cfg.tcl_thresh).astype(np.uint8)
        tr = cv2.cvtColor(tr * 255, cv2.COLOR_GRAY2BGR)
        tcl = cv2.cvtColor(tcl * 255, cv2.COLOR_GRAY2BGR)
        image_show = np.concatenate([image_show, tr, tcl], axis=1)
        return image_show
    else:
        return image_show
