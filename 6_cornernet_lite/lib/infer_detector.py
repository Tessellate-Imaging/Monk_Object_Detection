import cv2
from core.detectors import *
from core.vis_utils import draw_bboxes


class Infer():
    def __init__(self, verbose=1):
        self.system_dict = {};
        self.system_dict["verbose"] = verbose;
        self.system_dict["local"] = {};

    def Model(self, class_list, base="CornerNet_Saccade", model_path="./cache/nnet/CornerNet_Saccade/CornerNet_Saccade_final.pkl"):
        if(base == "CornerNet_Saccade"):
            self.system_dict["local"]["detector"] = CornerNet_Saccade(test=True, class_list=class_list, model_path=model_path)
        elif(base == "CornerNet_Squeeze"):
            self.system_dict["local"]["detector"] = CornerNet_Squeeze(test=True, class_list=class_list, model_path=model_path)


    def Predict(self, img_path, vis_thresh=0.3, output_img="output.jpg"):
        image    = cv2.imread(img_path)
        bboxes = self.system_dict["local"]["detector"](image)
        image  = draw_bboxes(image, bboxes, thresh=vis_thresh)
        cv2.imwrite(output_img, image)

        return bboxes;
