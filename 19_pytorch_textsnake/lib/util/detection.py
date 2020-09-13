import numpy as np
import cv2
import torch
from util.config import config as cfg
from util.misc import fill_hole, regularize_sin_cos
from util.misc import norm2, vector_cos, vector_sin
from util.misc import disjoint_merge, merge_polygons


class TextDetector(object):

    def __init__(self, model, tr_thresh=0.4, tcl_thresh=0.6):
        self.model = model
        self.tr_thresh = tr_thresh
        self.tcl_thresh = tcl_thresh

        # evaluation mode
        model.eval()

    def find_innerpoint(self, cont):
        """
        generate an inner point of input polygon using mean of x coordinate by:
        1. calculate mean of x coordinate(xmean)
        2. calculate maximum and minimum of y coordinate(ymax, ymin)
        3. iterate for each y in range (ymin, ymax), find first segment in the polygon
        4. calculate means of segment
        :param cont: input polygon
        :return:
        """

        xmean = cont[:, 0, 0].mean()
        ymin, ymax = cont[:, 0, 1].min(), cont[:, 0, 1].max()
        found = False
        found_y = []
        #
        for i in np.arange(ymin - 1, ymax + 1, 0.5):
            # if in_poly > 0, (xmean, i) is in `cont`
            in_poly = cv2.pointPolygonTest(cont, (xmean, i), False)
            if in_poly > 0:
                found = True
                found_y.append(i)
            # first segment found
            if in_poly < 0 and found:
                break

        if len(found_y) > 0:
            return (xmean, np.array(found_y).mean())

        # if cannot find using above method, try each point's neighbor
        else:
            for p in range(len(cont)):
                point = cont[p, 0]
                for i in range(-1, 2, 1):
                    for j in range(-1, 2, 1):
                        test_pt = point + [i, j]
                        if cv2.pointPolygonTest(cont, (test_pt[0], test_pt[1]), False) > 0:
                            return test_pt

    def in_contour(self, cont, point):
        """
        utility function for judging whether `point` is in the `contour`
        :param cont: cv2.findCountour result
        :param point: 2d coordinate (x, y)
        :return:
        """
        x, y = point
        return cv2.pointPolygonTest(cont, (x, y), False) > 0

    def centerlize(self, x, y, H, W, tangent_cos, tangent_sin, tcl_contour, stride=1.):
        """
        centralizing (x, y) using tangent line and normal line.
        :return: coordinate after centralizing
        """

        # calculate normal sin and cos
        normal_cos = -tangent_sin
        normal_sin = tangent_cos

        # find upward
        _x, _y = x, y
        while self.in_contour(tcl_contour, (_x, _y)):
            _x = _x + normal_cos * stride
            _y = _y + normal_sin * stride
            if int(_x) >= W or int(_x) < 0 or int(_y) >= H or int(_y) < 0:
                break
        end1 = np.array([_x, _y])

        # find downward
        _x, _y = x, y
        while self.in_contour(tcl_contour, (_x, _y)):
            _x = _x - normal_cos * stride
            _y = _y - normal_sin * stride
            if int(_x) >= W or int(_x) < 0 or int(_y) >= H or int(_y) < 0:
                break
        end2 = np.array([_x, _y])

        # centralizing
        center = (end1 + end2) / 2

        return center

    def mask_to_tcl(self, pred_sin, pred_cos, pred_radii, tcl_contour, init_xy, direct=1):
        """
        Iteratively find center line in tcl mask using initial point (x, y)
        :param pred_sin: predict sin map
        :param pred_cos: predict cos map
        :param tcl_contour: predict tcl contour
        :param init_xy: initial (x, y)
        :param direct: direction [-1|1]
        :return:
        """

        H, W = pred_sin.shape
        x_shift, y_shift = init_xy

        result = []
        max_attempt = 200
        attempt = 0

        while self.in_contour(tcl_contour, (x_shift, y_shift)):

            attempt += 1

            sin = pred_sin[int(y_shift), int(x_shift)]
            cos = pred_cos[int(y_shift), int(x_shift)]
            x_c, y_c = self.centerlize(x_shift, y_shift, H, W, cos, sin, tcl_contour)

            sin_c = pred_sin[int(y_c), int(x_c)]
            cos_c = pred_cos[int(y_c), int(x_c)]
            radii_c = pred_radii[int(y_c), int(x_c)]

            result.append(np.array([x_c, y_c, radii_c]))

            # shift stride
            for shrink in [1/2., 1/4., 1/8., 1/16., 1/32.]:
                t = shrink * radii_c   # stride = +/- 0.5 * [sin|cos](theta), if new point is outside, shrink it until shrink < 1/32., hit ends
                x_shift_pos = x_c + cos_c * t * direct  # positive direction
                y_shift_pos = y_c + sin_c * t * direct  # positive direction
                x_shift_neg = x_c - cos_c * t * direct  # negative direction
                y_shift_neg = y_c - sin_c * t * direct  # negative direction

                # if first point, select positive direction shift
                if len(result) == 1:
                    x_shift, y_shift = x_shift_pos, y_shift_pos
                else:
                    # else select point further with second last point
                    dist_pos = norm2(result[-2][:2] - (x_shift_pos, y_shift_pos))
                    dist_neg = norm2(result[-2][:2] - (x_shift_neg, y_shift_neg))
                    if dist_pos > dist_neg:
                        x_shift, y_shift = x_shift_pos, y_shift_pos
                    else:
                        x_shift, y_shift = x_shift_neg, y_shift_neg
                # if out of bounds, skip
                if int(x_shift) >= W or int(x_shift) < 0 or int(y_shift) >= H or int(y_shift) < 0:
                    continue
                # found an inside point
                if self.in_contour(tcl_contour, (x_shift, y_shift)):
                    break
            # if out of bounds, break
            if int(x_shift) >= W or int(x_shift) < 0 or int(y_shift) >= H or int(y_shift) < 0:
                break
            if attempt > max_attempt:
                break
        return np.array(result)

    def build_tcl(self, tcl_pred, sin_pred, cos_pred, radii_pred):
        """
        Find TCL's center points and radii of each point
        :param tcl_pred: output tcl mask, (512, 512)
        :param sin_pred: output sin map, (512, 512)
        :param cos_pred: output cos map, (512, 512)
        :param radii_pred: output radii map, (512, 512)
        :return: (list), tcl array: (n, 3), 3 denotes (x, y, radii)
        """
        all_tcls = []

        # find disjoint regions
        tcl_mask = fill_hole(tcl_pred)
        tcl_contours, _ = cv2.findContours(tcl_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cont in tcl_contours:

            # find an inner point of polygon
            init = self.find_innerpoint(cont)

            if init is None:
                continue

            x_init, y_init = init

            # find left/right tcl
            tcl_left = self.mask_to_tcl(sin_pred, cos_pred, radii_pred, cont, (x_init, y_init), direct=1)
            tcl_right = self.mask_to_tcl(sin_pred, cos_pred, radii_pred, cont, (x_init, y_init), direct=-1)
            # concat
            tcl = np.concatenate([tcl_left[::-1][:-1], tcl_right])
            all_tcls.append(tcl)

        return all_tcls

    def detect_contours(self, image, tr_pred, tcl_pred, sin_pred, cos_pred, radii_pred):
        """
        Input: FCN output, Output: text detection after post-processing

        :param image: (np.array) input image (3, H, W)
        :param tr_pred: (np.array), text region prediction, (2, H, W)
        :param tcl_pred: (np.array), text center line prediction, (2, H, W)
        :param sin_pred: (np.array), sin prediction, (H, W)
        :param cos_pred: (np.array), cos line prediction, (H, W)
        :param radii_pred: (np.array), radii prediction, (H, W)

        :return:
            (list), tcl array: (n, 3), 3 denotes (x, y, radii)
        """

        # thresholding
        tr_pred_mask = tr_pred[1] > self.tr_thresh
        tcl_pred_mask = tcl_pred[1] > self.tcl_thresh

        # multiply TR and TCL
        tcl_mask = tcl_pred_mask * tr_pred_mask

        # regularize
        sin_pred, cos_pred = regularize_sin_cos(sin_pred, cos_pred)

        # find tcl in each predicted mask
        detect_result = self.build_tcl(tcl_mask, sin_pred, cos_pred, radii_pred)

        return self.postprocessing(image, detect_result, tr_pred_mask)

    def detect(self, image):
        """

        :param image:
        :return:
        """
        # get model output
        output = self.model(image)
        image = image[0].data.cpu().numpy()
        tr_pred = output[0, 0:2].softmax(dim=0).data.cpu().numpy()
        tcl_pred = output[0, 2:4].softmax(dim=0).data.cpu().numpy()
        sin_pred = output[0, 4].data.cpu().numpy()
        cos_pred = output[0, 5].data.cpu().numpy()
        radii_pred = output[0, 6].data.cpu().numpy()

        # find text contours
        contours = self.detect_contours(image, tr_pred, tcl_pred, sin_pred, cos_pred, radii_pred)  # (n_tcl, 3)

        output = {
            'image': image,
            'tr': tr_pred,
            'tcl': tcl_pred,
            'sin': sin_pred,
            'cos': cos_pred,
            'radii': radii_pred
        }
        return contours, output

    def merge_contours(self, all_contours):
        """ Merge overlapped instances to one instance with disjoint find / merge algorithm
        :param all_contours: (list(np.array)), each with (n_points, 2)
        :return: (list(np.array)), each with (n_points, 2)
        """

        def stride(disks, other_contour, left, step=0.3):
            if len(disks) < 2:
                return False
            if left:
                last_point, before_point = disks[:2]
            else:
                before_point, last_point = disks[-2:]
            radius = last_point[2]
            cos = vector_cos(last_point[:2] - before_point[:2])
            sin = vector_sin(last_point[:2] - before_point[:2])
            new_point = last_point[:2] + radius * step * np.array([cos, sin])
            return self.in_contour(other_contour, new_point)

        def can_merge(disks, other_contour):
            return stride(disks, other_contour, left=True) or stride(disks, other_contour, left=False)

        F = list(range(len(all_contours)))
        for i in range(len(all_contours)):
            cont_i, disk_i = all_contours[i]
            for j in range(i + 1, len(all_contours)):
                cont_j, disk_j = all_contours[j]
                if can_merge(disk_i, cont_j):
                    disjoint_merge(i, j, F)

        merged_polygons = merge_polygons([cont for cont, disks in all_contours], F)
        return merged_polygons

    def postprocessing(self, image, detect_result, tr_pred_mask):
        """ convert geometric info(center_x, center_y, radii) into contours
        :param image: (np.array), input image
        :param result: (list), each with (n, 3), 3 denotes (x, y, radii)
        :param tr_pred_mask: (np.array), predicted text area mask, each with shape (H, W)
        :return: (np.ndarray list), polygon format contours
        """

        all_conts = []
        for disk in detect_result:
            reconstruct_mask = np.zeros(image.shape[1:], dtype=np.uint8)
            for x, y, r in disk:
                # expand radius for higher recall
                if cfg.post_process_expand > 0.0:
                    r *= (1. + cfg.post_process_expand)
                cv2.circle(reconstruct_mask, (int(x), int(y)), max(1, int(r)), 1, -1)

            # according to the paper, at least half of pixels in the reconstructed text area should be classiﬁed as TR
            if (reconstruct_mask * tr_pred_mask).sum() < reconstruct_mask.sum() * 0.5:
                continue

            # filter out too small objects
            conts, _ = cv2.findContours(reconstruct_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(conts) > 1:
                conts.sort(key=lambda x: cv2.contourArea(x), reverse=True)
            elif not conts:
                continue
            all_conts.append((conts[0][:, 0, :], disk))

        # merge joined instances
        if cfg.post_process_merge:
            all_conts = self.merge_contours(all_conts)
        else:
            all_conts = [cont[0] for cont in all_conts]

        return all_conts