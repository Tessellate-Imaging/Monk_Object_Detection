import cv2
from core.detectors import *
from core.vis_utils import draw_bboxes



class Infer():
    '''
    Class for main inference

    Args:
        verbose (int): Set verbosity levels
                        0 - Print Nothing
                        1 - Print desired details
    '''
    def __init__(self, verbose=1):
        self.system_dict = {};
        self.system_dict["verbose"] = verbose;
        self.system_dict["local"] = {};

    def Model(self, class_list, base="CornerNet_Saccade", model_path="./cache/nnet/CornerNet_Saccade/CornerNet_Saccade_final.pkl"):
        '''
        User function: Selet trained model params

        Args:
            class_list (list):  List containing all class names in the order same as training
            base (str): Select appropriate model
            model_path (str): Relative path to the trained model 

        Returns:
            None
        '''
        if(base == "CornerNet_Saccade"):
            self.system_dict["local"]["detector"] = CornerNet_Saccade(test=True, class_list=class_list, model_path=model_path)
        elif(base == "CornerNet_Squeeze"):
            self.system_dict["local"]["detector"] = CornerNet_Squeeze(test=True, class_list=class_list, model_path=model_path)


    def Predict(self, img_path, vis_thresh=0.3, output_img="output.jpg"):
        '''
        User function: Run inference on multiple images and visualize them

        Args:
            img_path (str): Relative path to the image file
            vis_thresh (float): Threshold for predicted scores. Scores for objects detected below this score will not be displayed 
            output_folder (str): Path to folder where output images will be saved

        Returns:
            None 
        '''
        image    = cv2.imread(img_path)
        bboxes = self.system_dict["local"]["detector"](image)
        image  = draw_bboxes(image, bboxes, thresh=vis_thresh)
        cv2.imwrite(output_img, image)

        return bboxes;
