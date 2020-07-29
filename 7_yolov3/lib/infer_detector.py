from sys import platform

from models import * 
from utils.datasets import *
from utils.utils import *


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


        self.system_dict["params"] = {};
        self.system_dict["params"]["output"] = "output";
        

        self.system_dict["params"]["fourcc"] = "mp4v";
        self.system_dict["params"]["half"] = False;
        self.system_dict["params"]["device"] = "0";
        self.system_dict["params"]["agnostic_nms"] = False;
        self.system_dict["params"]["classes"] = "";
        self.system_dict["params"]["device"] = "0";
        
        self.system_dict["params"]["view_img"] = False;
        self.system_dict["params"]["save_txt"] = True;
        self.system_dict["params"]["save_img"] = True;

        self.system_dict["params"]["cfg"] = "custom_data/ship/yolov3.cfg";
        self.system_dict["params"]["names"] = "custom_data/ship/classes_list.txt";
        self.system_dict["params"]["weights"] = "weights/last.pt";

        self.system_dict["params"]["img_size"] = 416;
        self.system_dict["params"]["conf_thres"] = 0.3;
        self.system_dict["params"]["iou_thres"] = 0.5;


    def Model(self, model_name, class_list, weight, use_gpu=True, input_size=416, half_precision=False):
        '''
        User function: Set Model parameters

            Available Models
                yolov3
                yolov3s
                yolov3-spp
                yolov3-spp3
                yolov3-tiny
                yolov3-spp-matrix
                csresnext50-panet-spp


        Args:
            model_name (str): Select model from available models
            class_list (list): List of classes as per given in training session
            weight (srt): Path to file storing model weights 
            use_gpu (bool): If True, model is loaded onto GPU device, else on CPU
            half_precision (bool): If True, uses only 16 bit floating point operations for faster inferencing

        Returns:
            None
        '''
        self.system_dict["params"]["cfg"] = model_name + ".cfg"; 
        self.system_dict["params"]["half"] = half_precision;
        self.system_dict["params"]["names"] = class_list;
        self.system_dict["params"]["weights"] = weight;
        self.system_dict["params"]["use_gpu"] = use_gpu;
        self.system_dict["params"]["img_size"] = input_size;

        self.system_dict["local"]["device"] = torch_utils.select_device(device=self.system_dict["params"]["device"] if use_gpu else 'cpu');

        self.system_dict["local"]["model"] = Darknet(self.system_dict["params"]["cfg"], 
                                                    self.system_dict["params"]["img_size"]);

        # Load weights
        attempt_download(self.system_dict["params"]["weights"])
        if self.system_dict["params"]["weights"].endswith('.pt'):  # pytorch format
            self.system_dict["local"]["model"].load_state_dict(torch.load(self.system_dict["params"]["weights"], 
                                                                            map_location=self.system_dict["local"]["device"])['model'])
        else:  # darknet format
            load_darknet_weights(self.system_dict["local"]["model"], 
                                    self.system_dict["params"]["weights"])

        # Second-stage classifier
        self.system_dict["params"]["classify"] = False
        if self.system_dict["params"]["classify"]:
            modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
            modelc.to(device).eval()

        self.system_dict["local"]["model"].to(self.system_dict["local"]["device"]).eval()

        # Half precision
        self.system_dict["params"]["half"] = self.system_dict["params"]["half"] and self.system_dict["local"]["device"].type != 'cpu'  # half precision only supported on CUDA
        if self.system_dict["params"]["half"]:
            self.system_dict["local"]["model"].half()



    def Predict(self, img_path, conf_thres=0.3, iou_thres=0.5):
        '''
        User function: Run inference on image and visualize it

        Args:
            img_path (str): Relative path to the image file
            conf_thres (float): Threshold for predicted scores. Scores for objects detected below this score will not be displayed 
            iou_thres (float): Threshold for bounding boxes nms merging

        Returns:
            None.
        '''
        self.system_dict["params"]["conf_thres"] = conf_thres;
        self.system_dict["params"]["iou_thres"] = iou_thres;
        view_img = self.system_dict["params"]["view_img"];
        save_txt = self.system_dict["params"]["save_txt"];
        save_img = self.system_dict["params"]["save_img"];
        out = self.system_dict["params"]["output"];
        source = "tmp";

        if(not os.path.isdir(source)):
            os.mkdir(source);
        else:
            os.system("rm -r " + source);
            os.mkdir(source);

        if(not os.path.isdir(out)):
            os.mkdir(out);
        else:
            os.system("rm -r " + out);
            os.mkdir(out);

        os.system("cp " + img_path + " " + source + "/");

        self.system_dict["local"]["dataset"] = LoadImages(source, 
                                                            img_size=self.system_dict["params"]["img_size"], 
                                                            half=self.system_dict["params"]["half"]);

        # Get names and colors
        names = self.system_dict["params"]["names"]
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


        # Run inference
        t0 = time.time()
        for path, img, im0s, vid_cap in self.system_dict["local"]["dataset"]:
            t = time.time()

            # Get detections
            img = torch.from_numpy(img).to(self.system_dict["local"]["device"])
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = self.system_dict["local"]["model"](img)[0]
            

            if self.system_dict["params"]["half"]:
                pred = pred.float()

            # Apply NMS
            pred = non_max_suppression(pred, self.system_dict["params"]["conf_thres"], 
                                       self.system_dict["params"]["conf_thres"], 
                                       classes=self.system_dict["params"]["classes"], 
                                       agnostic=self.system_dict["params"]["agnostic_nms"])
            

            # Apply Classifier
            if self.system_dict["params"]["classify"]:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name)
                s += '%gx%g ' % img.shape[2:]  # print string
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].detach().unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in det:
                        if save_txt:  # Write to file
                            with open(save_path + '.txt', 'a') as file:
                                file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, time.time() - t))

                # Stream results
                if view_img:
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if save_img:
                    if self.system_dict["local"]["dataset"].mode == 'images':
                        cv2.imwrite(save_path, im0)
                        cv2.imwrite("output.jpg", im0);


        if save_txt or save_img:
            print('Results saved to %s' % os.getcwd() + os.sep + out)
            if platform == 'darwin':  # MacOS
                os.system('open ' + out + ' ' + save_path)

        print('Done. (%.3fs)' % (time.time() - t0))
