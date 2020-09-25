from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv
import time

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
        
    def Model_Params(self, config_file, checkpoint_file, use_gpu=True):
        self.system_dict["local"]["config_file"] = config_file;
        self.system_dict["local"]["checkpoint_file"] = checkpoint_file;
        
        if(use_gpu):
            self.system_dict["local"]["model"] = init_detector(config_file, checkpoint_file, device='cuda:0');
        else:
            self.system_dict["local"]["model"] = init_detector(config_file, checkpoint_file, device='cpu');
        
    def Predict(self, 
                img_path=None,
                out_img_path=None,
                thresh=0.3):
        
               
        start = time.time();
        result = inference_detector(self.system_dict["local"]["model"], img_path)
        end = time.time();
        print("Inference Time: {} sec".format(end-start));
        
        
        # or save the visualization results to image files
        start = time.time();
        show_result_ins(img_path, result,
                        self.system_dict["local"]["model"].CLASSES, out_file=out_img_path,
                        score_thr=thresh)
        end = time.time();
        print("Saving Time: {} sec".format(end-start));
        
        return result;
