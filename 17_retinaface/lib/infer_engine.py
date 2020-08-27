from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data.config import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
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
        
    def check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True


    def remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}


    def load_model(self, model, pretrained_path, load_to_cpu):
        print('Loading pretrained model from {}'.format(pretrained_path))
        if load_to_cpu:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model
    
    def Model_Params(self, 
                     model_type="mobilenet", 
                     model_path="weights_trained/mobilenet0.25_Final.pth", 
                     use_gpu=True):
        if(model_type == "mobilenet"):
            self.system_dict["local"]["network"] = "mobile0.25";
        elif(model_type == "resnet"):
            self.system_dict["local"]["network"] = "resnet50";
        else:
            print("Model type not found");
            print("Available models - mobilenet, resnet");
            
        self.system_dict["local"]["cpu"] = not use_gpu;
        self.system_dict["local"]["trained_model"] = model_path;
        
        torch.set_grad_enabled(False)
        self.system_dict["local"]["cfg"] = None
        if self.system_dict["local"]["network"] == "mobile0.25":
            self.system_dict["local"]["cfg"] = cfg_mnet
        elif self.system_dict["local"]["network"] == "resnet50":
            self.system_dict["local"]["cfg"] = cfg_re50
        # net and model
        self.system_dict["local"]["net"] = RetinaFace(cfg=self.system_dict["local"]["cfg"], phase = 'test')
        self.system_dict["local"]["net"] = self.load_model(self.system_dict["local"]["net"], 
                                                      self.system_dict["local"]["trained_model"], 
                                                      self.system_dict["local"]["cpu"])
        self.system_dict["local"]["net"].eval()
        print('Finished loading model!')
        cudnn.benchmark = True
        self.system_dict["local"]["device"] = torch.device("cpu" if self.system_dict["local"]["cpu"] else "cuda")
        self.system_dict["local"]["net"] = self.system_dict["local"]["net"].to(self.system_dict["local"]["device"])
        
        
    def Predict(self, img_path="test.jpg", thresh=0.5, out_img_path="result.jpg"):
        image_path = img_path;
        confidence_threshold = thresh
        vis_thres = thresh
        nms_threshold = 0.4;
        top_k = 1000;
        keep_top_k = 750;
        save_image = True;
        name = out_img_path;
        
        device = self.system_dict["local"]["device"];
        net = self.system_dict["local"]["net"];
        cfg = self.system_dict["local"]["cfg"];
        
        
        resize = 1
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)
        
        
        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        # show image
        tmp = {};
        tmp["scores"] = [];
        tmp["bboxes"] = [];
        tmp["labels"] = [];
        
        for b in dets:
            if b[4] < vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            tmp["scores"].append(b[4]);
            tmp["bboxes"].append([b[0], b[1], b[2], b[3]]);
            tmp["labels"].append(text);

            # landms
            #cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            #cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            #cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            #cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            #cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            # save image

        if save_image:    
            cv2.imwrite(name, img_raw)
        
        return tmp;
    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    