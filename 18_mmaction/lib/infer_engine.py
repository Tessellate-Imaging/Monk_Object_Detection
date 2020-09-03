from mmaction.apis import init_recognizer, inference_recognizer

class Infer_Videos():
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
        
    def Dataset_Params(self, class_list_file):
        self.system_dict["local"]["class_list_file"] = class_list_file;
        
    def Model_Params(self, config_file, checkpoint_file, use_gpu=True):
        self.system_dict["local"]["config_file"] = config_file;
        self.system_dict["local"]["checkpoint_file"] = checkpoint_file;
        
        if(use_gpu):
            self.system_dict["local"]["model"] = init_recognizer(config_file, checkpoint_file, device='cuda')
        else:
            self.system_dict["local"]["model"] = init_recognizer(config_file, checkpoint_file, device='cpu')
            
    def Predict(self, 
                video_path = None):
        results = inference_recognizer(self.system_dict["local"]["model"], 
                                       video_path, 
                                       self.system_dict["local"]["class_list_file"])

        # show the results
        classes = [];
        scores = [];
        for result in results:
            classes.append(result[0]);
            scores.append(result[1]/100);
            print(f'{result[0]}: ', result[1]/100)
            
        return classes, scores