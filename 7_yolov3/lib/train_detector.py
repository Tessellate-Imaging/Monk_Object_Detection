import os
import sys
import numpy as np

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from validate import validate # import test.py to get mAP after each epoch

from models import *
from utils.datasets import *
from utils.utils import *

from torch.utils.tensorboard import SummaryWriter

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False
if(isnotebook()):
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm as tqdm

from update_cfg import update
from apex import amp


class Detector():
    '''
    Class to train a detector

    Args:
        verbose (int): Set verbosity levels
                        0 - Print Nothing
                        1 - Print desired details
    '''
    def __init__(self, verbose=1):
        self.system_dict = {};
        self.system_dict["verbose"] = verbose;
        self.system_dict["local"] = {};
        self.system_dict["fixed_params"] = {};
        self.system_dict["dataset"] = {};
        self.system_dict["dataset"]["train"] = {};
        self.system_dict["dataset"]["val"] = {};
        self.system_dict["dataset"]["val"]["status"] = False;
        self.system_dict["params"] = {};

        self.system_dict["params"]["mixed_precision"] = True
        try:  # Mixed precision training https://github.com/NVIDIA/apex
            from apex import amp
        except:
            self.system_dict["params"]["mixed_precision"] = False  # not installed


        self.set_fixed_params();


    def set_fixed_params(self):
        '''
        Internal function: Set fixed parameters

        Args:
            None

        Returns:
            None
        '''
        self.system_dict["fixed_params"]["wdir"] = 'weights' + os.sep  
        self.system_dict["fixed_params"]["last"] = self.system_dict["fixed_params"]["wdir"] + 'last.pt'
        self.system_dict["fixed_params"]["best"] = self.system_dict["fixed_params"]["wdir"] + 'best.pt'
        self.system_dict["fixed_params"]["results_file"] = 'results.txt'

        self.system_dict["fixed_params"]["hyp"] = {'giou': 3.54,  # giou loss gain
                                                   'cls': 37.4,  # cls loss gain
                                                   'cls_pw': 1.0,  # cls BCELoss positive_weight
                                                   'obj': 49.5,  # obj loss gain (*=img_size/320 if img_size != 320)
                                                   'obj_pw': 1.0,  # obj BCELoss positive_weight
                                                   'iou_t': 0.225,  # iou training threshold
                                                   'lr0': 0.00579,  # initial learning rate (SGD=5E-3, Adam=5E-4)
                                                   'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
                                                   'momentum': 0.937,  # SGD momentum
                                                   'weight_decay': 0.000484,  # optimizer weight decay
                                                   'fl_gamma': 0.5,  # focal loss gamma
                                                   'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
                                                   'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
                                                   'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
                                                   'degrees': 1.98,  # image rotation (+/- deg)
                                                   'translate': 0.05,  # image translation (+/- fraction)
                                                   'scale': 0.05,  # image scale (+/- gain)
                                                   'shear': 0.641}  # image shear (+/- deg)

        # Overwrite hyp with hyp*.txt (optional)
        f = glob.glob('hyp*.txt')
        if f:
            print('Using %s' % f[0])
            for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
                hyp[k] = v


    def set_train_dataset(self, img_dir, label_dir, class_list, batch_size=2, img_size=416, cache_images=False):
        '''
        User function: Set training dataset parameters

        Dataset Directory Structure

                    root_dir
                      |
                      |-----------images (img_dir)
                      |              |
                      |              |------------------img1.jpg
                      |              |------------------img2.jpg
                      |              |------------------.........(and so on)
                      |
                      |-----------labels (label_dir)
                      |              |
                      |              |------------------img1.txt
                      |              |------------------img2.txt
                      |              |------------------.........(and so on)
                      |
                      |------------classes.txt 
                      

            Classes file
             
                 List of classes in every new line.
                 The order corresponds to the IDs in annotation files
                 
                 Eg.
                      class1               (------------------------------> if will be 0)
                      class2               (------------------------------> if will be 1)
                      class3               (------------------------------> if will be 2)
                      class4               (------------------------------> if will be 3)
                      

            Annotation file format

                CLASS_ID BOX_X_CENTER BOX_Y_CENTER WIDTH BOX_WIDTH BOX_HEIGHT
                
                (All the coordinates should be normalized)
                (X coordinates divided by width of image, Y coordinates divided by height of image)
                
                Ex. (One line per bounding box of object in image)
                    class_id x1 y1 w h
                    class_id x1 y1 w h
                    ..... (and so on)
        

        Args:
            img_dir (str): Relative path to folder containing all training images
            label_dir (str): Relative path to folder containing all training labels text files
            class_list (list): List of all classes in dataset
            batch_size (int): Mini batch sampling size for training epochs
            cache_images (bool): If True, images are cached for faster loading

        Returns:
            None
        '''
        self.system_dict["dataset"]["train"]["img_dir"] = img_dir;
        self.system_dict["dataset"]["train"]["label_dir"] = label_dir;
        self.system_dict["dataset"]["train"]["class_list"] = class_list;
        self.system_dict["params"]["batch_size"] = batch_size;
        self.system_dict["params"]["accumulate"] = batch_size;
        self.system_dict["params"]["img_size"] = [img_size];
        self.system_dict["params"]["img_size_selected"] = [img_size];
        self.system_dict["params"]["cache_images"] = cache_images;


    def set_val_dataset(self, img_dir, label_dir):
        '''
        User function: Set training dataset parameters

        Dataset Directory Structure

                    root_dir
                      |
                      |-----------images (img_dir)
                      |              |
                      |              |------------------img1.jpg
                      |              |------------------img2.jpg
                      |              |------------------.........(and so on)
                      |
                      |-----------labels (label_dir)
                      |              |
                      |              |------------------img1.txt
                      |              |------------------img2.txt
                      |              |------------------.........(and so on)
                      |
                      |------------classes.txt 
                      

            Classes file
             
                 List of classes in every new line.
                 The order corresponds to the IDs in annotation files
                 
                 Eg.
                      class1               (------------------------------> if will be 0)
                      class2               (------------------------------> if will be 1)
                      class3               (------------------------------> if will be 2)
                      class4               (------------------------------> if will be 3)
                      

            Annotation file format

                CLASS_ID BOX_X_CENTER BOX_Y_CENTER WIDTH BOX_WIDTH BOX_HEIGHT
                
                (All the coordinates should be normalized)
                (X coordinates divided by width of image, Y coordinates divided by height of image)
                
                Ex. (One line per bounding box of object in image)
                    class_id x1 y1 w h
                    class_id x1 y1 w h
                    ..... (and so on)
        

        Args:
            img_dir (str): Relative path to folder containing all validation images
            label_dir (str): Relative path to folder containing all validation labels text files

        Returns:
            None
        '''
        self.system_dict["dataset"]["val"]["img_dir"] = img_dir;
        self.system_dict["dataset"]["val"]["label_dir"] = label_dir;
        self.system_dict["dataset"]["val"]["status"] = True;



    def set_model(self, model_name="yolov3"):
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
            gpu_devices (list): List of GPU Device IDs to be used in training

        Returns:
            None
        '''
        tmp_cfg = os.path.dirname(os.path.realpath(__file__)) + "/cfg/" + model_name + ".cfg";
        cmd = "cp " + tmp_cfg + " " + os.getcwd() + "/" + model_name + ".cfg";
        os.system(cmd);
        self.system_dict["params"]["cfg"] = model_name + ".cfg";



    def set_hyperparams(self, optimizer="sgd", lr=0.00579, multi_scale=False, evolve=False, num_generations=2, 
                        mixed_precision=True, gpu_devices="0"):
        '''
        User function: Set hyper parameters
            Available optimizers
                sgd
                adam

        Args:
            optimizer (str): Select the right optimizer
            lr (float): Initial learning rate for training
            multi_scale (bool): If True, run multi-scale training.
            evolve (bool): If True, runs multiple epochs in every generation and updates hyper-params accordingly
            mixed_precision (bool): If True, uses both 16-bit and 32-bit floating-point types in a model during training to make it run faster and use less memory.
            gpu_devices (str): List of all GPU device IDs separated by a comma in a string

        Returns:
            None
        '''
        self.system_dict["params"]["multi_scale"] = multi_scale;
        if(optimizer == "sgd"):
            self.system_dict["params"]["adam"] = False;
        self.system_dict["fixed_params"]["hyp"]["lr0"] = lr;
        self.system_dict["params"]["rect"] = False;
        self.system_dict["params"]["resume"] = False;
        self.system_dict["params"]["nosave"] = False;
        self.system_dict["params"]["notest"] = False;
        self.system_dict["params"]["evolve"] = evolve;
        self.system_dict["params"]["num_generations"] = num_generations;
        self.system_dict["params"]["bucket"] = "";
        self.system_dict["params"]["weights"] = "";
        self.system_dict["params"]["arc"] = "default";
        self.system_dict["params"]["name"] = "";
        self.system_dict["params"]["device"] = gpu_devices;
        self.system_dict["params"]["mixed_precision"] = mixed_precision;


    def setup(self):
        '''
        Internal function: Setup all the dataset, model and data params 

        Args:
            None

        Returns:
            None
        '''
        if(not os.path.isdir("weights")):
            os.mkdir("weights");


        #Device Setup
        self.system_dict["params"]["weights"] = last if self.system_dict["params"]["resume"] else self.system_dict["params"]["weights"]
        self.system_dict["local"]["device"] = torch_utils.select_device(self.system_dict["params"]["device"], 
                                                                        apex=self.system_dict["params"]["mixed_precision"], 
                                                                        batch_size=self.system_dict["params"]["batch_size"])
        if self.system_dict["local"]["device"].type == 'cpu':
            self.system_dict["params"]["mixed_precision"] = False

        self.system_dict["local"]["tb_writer"] = None
        self.system_dict["local"]["tb_writer"] = SummaryWriter()


        #Data Setup
        img_size, img_size_test = self.system_dict["params"]["img_size"] if len(self.system_dict["params"]["img_size"]) == 2 else self.system_dict["params"]["img_size"] * 2 

        init_seeds()
        if self.system_dict["params"]["multi_scale"]:
            img_sz_min = round(img_size / 32 / 1.5)
            img_sz_max = round(img_size / 32 * 1.5)
            img_size = img_sz_max * 32  # initiate with maximum multi_scale size
            print('Using multi-scale %g - %g' % (img_sz_min * 32, img_size))

            self.system_dict["params"]["img_sz_min"] = img_sz_min;
            self.system_dict["params"]["img_sz_max"] = img_sz_max;

        self.system_dict["params"]["img_size"] = img_size;
        self.system_dict["params"]["img_size_test"] = img_size_test;

        f = open(self.system_dict["dataset"]["train"]["class_list"], 'r');
        lines = f.readlines();
        f.close();

        self.system_dict["local"]["classes"] = [];
        for i in range(len(lines)):
            if(lines[i] != "" and lines[i] != "\n" ):
                self.system_dict["local"]["classes"].append(lines[i]);
        self.system_dict["local"]["num_classes"] = int(len(self.system_dict["local"]["classes"]));
        if(self.system_dict["local"]["num_classes"] == 1):
            self.system_dict["params"]["single_cls"] = True;
        else:
            self.system_dict["params"]["single_cls"] = False;


        self.system_dict["local"]["nc"] = 1 if self.system_dict["params"]["single_cls"] else self.system_dict["local"]["num_classes"]

        # Remove previous results
        for f in glob.glob('*_batch*.png') + glob.glob(self.system_dict["fixed_params"]["results_file"]):
            os.remove(f)

        if 'pw' not in self.system_dict["params"]["arc"]:  # remove BCELoss positive weights
            self.system_dict["fixed_params"]["hyp"]['cls_pw'] = 1.
            self.system_dict["fixed_params"]["hyp"]['obj_pw'] = 1.
        


        #Update Config file
        update(self.system_dict["params"]["cfg"], self.system_dict["local"]["num_classes"]);

        #Model
        self.system_dict["local"]["model"] = Darknet(self.system_dict["params"]["cfg"], 
                                            arc=self.system_dict["params"]["arc"]).to(self.system_dict["local"]["device"])

        attempt_download(self.system_dict["params"]["weights"])

        # Optimizer
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in dict(self.system_dict["local"]["model"].named_parameters()).items():
            if '.bias' in k:
                pg2 += [v]  # biases
            elif 'Conv2d.weight' in k:
                pg1 += [v]  # apply weight_decay
            else:
                pg0 += [v]  # all else

        if self.system_dict["params"]["adam"]:
            # hyp['lr0'] *= 0.1  # reduce lr (i.e. SGD=5E-3, Adam=5E-4)
            self.system_dict["local"]["optimizer"] = optim.Adam(pg0, lr=self.system_dict["fixed_params"]["hyp"]['lr0'])
            # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
        else:
            self.system_dict["local"]["optimizer"] = optim.SGD(pg0, lr=self.system_dict["fixed_params"]["hyp"]['lr0'], 
                                                                    momentum=self.system_dict["fixed_params"]["hyp"]['momentum'], nesterov=True)

        self.system_dict["local"]["optimizer"].add_param_group({'params': pg1, 'weight_decay': self.system_dict["fixed_params"]["hyp"]['weight_decay']})  # add pg1 with weight_decay
        self.system_dict["local"]["optimizer"].add_param_group({'params': pg2})  # add pg2 (biases)
        del pg0, pg1, pg2


        self.system_dict["local"]["start_epoch"] = 0
        self.system_dict["local"]["best_fitness"] = float('inf')

        if self.system_dict["params"]["weights"].endswith('.pt'):  
            chkpt = torch.load(self.system_dict["params"]["weights"], map_location=self.system_dict["local"]["device"])

            # load model
            try:
                chkpt['model'] = {k: v for k, v in chkpt['model'].items() if self.system_dict["local"]["model"].state_dict()[k].numel() == v.numel()}
                self.system_dict["local"]["model"].load_state_dict(chkpt['model'], strict=False)
            except KeyError as e:
                s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                    "See https://github.com/ultralytics/yolov3/issues/657" % (self.system_dict["params"]["weights"], self.system_dict["params"]["cfg"], 
                                                                              self.system_dict["params"]["weights"])
                raise KeyError(s) from e

            # load optimizer
            if chkpt['optimizer'] is not None:
                self.system_dict["local"]["optimizer"].load_state_dict(chkpt['optimizer'])
                self.system_dict["local"]["best_fitness"] = chkpt['best_fitness']

            # load results
            if chkpt.get('training_results') is not None:
                with open(self.system_dict["fixed_params"]["results_file"], 'w') as file:
                    file.write(chkpt['training_results'])

            self.system_dict["local"]["start_epoch"] = chkpt['epoch'] + 1
            del chkpt

        elif len(self.system_dict["params"]["weights"]) > 0:  # darknet format
            # possible weights are '*.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
            load_darknet_weights(self.system_dict["local"]["model"], self.system_dict["params"]["weights"])


        #Scheduler
        self.system_dict["local"]["scheduler"] = lr_scheduler.MultiStepLR(self.system_dict["local"]["optimizer"], 
                                                    milestones=[round(self.system_dict["params"]["epochs"] * x) for x in [0.8, 0.9]], gamma=0.1)
        self.system_dict["local"]["scheduler"].last_epoch = self.system_dict["local"]["start_epoch"] - 1


        if self.system_dict["params"]["mixed_precision"]:
            self.system_dict["local"]["model"], self.system_dict["local"]["optimizer"] = amp.initialize(self.system_dict["local"]["model"], 
                                                                                                        self.system_dict["local"]["optimizer"], 
                                                                                                        opt_level='O1', verbosity=0)
            
            
        # Initialize distributed training
        if self.system_dict["local"]["device"].type != 'cpu' and torch.cuda.device_count() > 1:
            dist.init_process_group(backend='nccl', 
                                    init_method='tcp://127.0.0.1:9999',  
                                    world_size=1,  
                                    rank=0)  
            self.system_dict["local"]["model"] = torch.nn.parallel.DistributedDataParallel(self.system_dict["local"]["model"], 
                                                                                            find_unused_parameters=True)
            self.system_dict["local"]["model"].yolo_layers = self.system_dict["local"]["model"].module.yolo_layers 



        # Dataset
        self.system_dict["local"]["dataset"] = LoadImagesAndLabels(self.system_dict["dataset"]["train"]["img_dir"], 
                                        self.system_dict["dataset"]["train"]["label_dir"],
                                        self.system_dict["params"]["img_size"], 
                                        self.system_dict["params"]["batch_size"],
                                        augment=True,
                                        hyp=self.system_dict["fixed_params"]["hyp"], 
                                        rect=self.system_dict["params"]["rect"],  
                                        cache_labels=True,
                                        cache_images=self.system_dict["params"]["cache_images"],
                                        single_cls=self.system_dict["params"]["single_cls"])
        
        # Dataloader
        self.system_dict["params"]["batch_size"] = min(self.system_dict["params"]["batch_size"], len(self.system_dict["local"]["dataset"]))
        self.system_dict["local"]["nw"] = min([os.cpu_count(), self.system_dict["params"]["batch_size"] if self.system_dict["params"]["batch_size"] > 1 else 0, 8])  
        

        self.system_dict["local"]["dataloader"] = torch.utils.data.DataLoader(self.system_dict["local"]["dataset"],
                                                 batch_size=self.system_dict["params"]["batch_size"],
                                                 num_workers=self.system_dict["local"]["nw"],
                                                 shuffle=not self.system_dict["params"]["rect"], 
                                                 pin_memory=True,
                                                 collate_fn=self.system_dict["local"]["dataset"].collate_fn)



        # Testloader
        if(self.system_dict["dataset"]["val"]["status"]):
            self.system_dict["local"]["testloader"] = torch.utils.data.DataLoader(LoadImagesAndLabels(self.system_dict["dataset"]["val"]["img_dir"], 
                                                                                                        self.system_dict["dataset"]["val"]["label_dir"],
                                                                                                        self.system_dict["params"]["img_size"], 
                                                                                                        self.system_dict["params"]["batch_size"] * 2,
                                                                                                        hyp=self.system_dict["fixed_params"]["hyp"],
                                                                                                        rect=False,
                                                                                                        cache_labels=True,
                                                                                                        cache_images=self.system_dict["params"]["cache_images"],
                                                                                                        single_cls=self.system_dict["params"]["single_cls"]),
                                                                                     batch_size=self.system_dict["params"]["batch_size"] * 2,
                                                                                     num_workers=self.system_dict["local"]["nw"],
                                                                                     pin_memory=True,
                                                                                     collate_fn=self.system_dict["local"]["dataset"].collate_fn)


    def Train(self, num_epochs=2):
        '''
        User function: Set training params and train

        Args:
            num_epochs (int): Number of epochs in training

        Returns:
            None
        '''
        self.system_dict["params"]["epochs"] = num_epochs;

        if not self.system_dict["params"]["evolve"]:
            self.setup();
            self.start_training();
        else:
            if(not self.system_dict["dataset"]["val"]["status"]):
                print("Validation data required for evolving hyper-parameters");
            else:
                for _ in range(self.system_dict["params"]["num_generations"]):  # generations to evolve
                    if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                        # Select parent(s)
                        x = np.loadtxt('evolve.txt', ndmin=2)
                        parent = 'single'  # parent selection method: 'single' or 'weighted'
                        if parent == 'single' or len(x) == 1:
                            x = x[fitness(x).argmax()]
                        elif parent == 'weighted':  # weighted combination
                            n = min(10, len(x))  # number to merge
                            x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                            w = fitness(x) - fitness(x).min()  # weights
                            x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # new parent

                        # Mutate
                        method = 3
                        s = 0.3  # 20% sigma
                        np.random.seed(int(time.time()))
                        g = np.array([1, 1, 1, 1, 1, 1, 1, 0, .1, 1, 0, 1, 1, 1, 1, 1, 1, 1])  # gains
                        ng = len(g)
                        if method == 1:
                            v = (np.random.randn(ng) * np.random.random() * g * s + 1) ** 2.0
                        elif method == 2:
                            v = (np.random.randn(ng) * np.random.random(ng) * g * s + 1) ** 2.0
                        elif method == 3:
                            v = np.ones(ng)
                            while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                                r = (np.random.random(ng) < 0.1) * np.random.randn(ng)  # 10% mutation probability
                                v = (g * s * r + 1) ** 2.0
                        for i, k in enumerate(self.system_dict["fixed_params"]["hyp"].keys()):  # plt.hist(v.ravel(), 300)
                            self.system_dict["fixed_params"]["hyp"][k] = x[i + 7] * v[i]  # mutate

                    # Clip to limits
                    keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
                    limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9), (0, 3)]
                    for k, v in zip(keys, limits):
                        self.system_dict["fixed_params"]["hyp"][k] = np.clip(self.system_dict["fixed_params"]["hyp"][k], v[0], v[1])

                    # Train mutation
                    self.setup();
                    results = self.start_training();
                    self.system_dict["params"]["img_size"] = self.system_dict["params"]["img_size_selected"];

                    # Write mutation results
                    print_mutation(self.system_dict["fixed_params"]["hyp"], results, self.system_dict["params"]["bucket"])

                    # Plot results
                    plot_evolution_results(self.system_dict["fixed_params"]["hyp"])







    def start_training(self):
        '''
        Internal function: Start training post setting up all params

        Args:
            None

        Returns:
            str: Training and validation epoch results
        '''
        self.system_dict["local"]["nb"] = len(self.system_dict["local"]["dataloader"])
        prebias = self.system_dict["local"]["start_epoch"] == 0
        self.system_dict["local"]["model"].nc = self.system_dict["local"]["nc"]  # attach number of classes to model
        self.system_dict["local"]["model"].arc = self.system_dict["params"]["arc"]  # attach yolo architecture
        self.system_dict["local"]["model"].hyp = self.system_dict["fixed_params"]["hyp"]  # attach hyperparameters to model
        self.system_dict["local"]["model"].class_weights = labels_to_class_weights(self.system_dict["local"]["dataset"].labels, 
                                                                self.system_dict["local"]["nc"]).to(self.system_dict["local"]["device"])  # attach class weights
        maps = np.zeros(self.system_dict["local"]["nc"])  # mAP per class
        # torch.autograd.set_detect_anomaly(True)
        results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
        t0 = time.time()
        torch_utils.model_info(self.system_dict["local"]["model"], report='summary')  # 'full' or 'summary'
        print('Using %g dataloader workers' % self.system_dict["local"]["nw"])
        print('Starting training for %g epochs...' % self.system_dict["params"]["epochs"])


        for epoch in range(self.system_dict["local"]["start_epoch"], self.system_dict["params"]["epochs"]):  # epoch ------------------------------
            self.system_dict["local"]["model"].train()

            # Prebias
            if prebias:
                if epoch < 3:  # prebias
                    ps = 0.1, 0.9  # prebias settings (lr=0.1, momentum=0.9)
                else:  # normal training
                    ps = self.system_dict["fixed_params"]["hyp"]['lr0'], self.system_dict["fixed_params"]["hyp"]['momentum']  # normal training settings
                    print_model_biases(self.system_dict["local"]["model"])
                    prebias = False

                # Bias optimizer settings
                self.system_dict["local"]["optimizer"].param_groups[2]['lr'] = ps[0]
                if self.system_dict["local"]["optimizer"].param_groups[2].get('momentum') is not None:  # for SGD but not Adam
                    self.system_dict["local"]["optimizer"].param_groups[2]['momentum'] = ps[1]

            # Update image weights (optional)
            if self.system_dict["local"]["dataset"].image_weights:
                print("in here")
                w = self.system_dict["local"]["model"].class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
                image_weights = labels_to_image_weights(self.system_dict["local"]["dataset"].labels, 
                                                        nc=self.system_dict["local"]["nc"], 
                                                        class_weights=w)
                self.system_dict["local"]["dataset"].indices = random.choices(range(self.system_dict["local"]["dataset"].n), 
                                                    weights=image_weights, k=self.system_dict["local"]["dataset"].n)  # rand weighted idx

            mloss = torch.zeros(4).to(self.system_dict["local"]["device"])  # mean losses
            print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
            pbar = tqdm(enumerate(self.system_dict["local"]["dataloader"]), total=self.system_dict["local"]["nb"])  # progress bar

            for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------

                
                ni = i + self.system_dict["local"]["nb"] * epoch  # number integrated batches (since train start)
                imgs = imgs.to(self.system_dict["local"]["device"]).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
                targets = targets.to(self.system_dict["local"]["device"])

                # Multi-Scale training
                if self.system_dict["params"]["multi_scale"]:
                    if ni / self.system_dict["params"]["accumulate"] % 10 == 0:  #  adjust (67% - 150%) every 10 batches
                        self.system_dict["params"]["img_size"] = random.randrange(self.system_dict["params"]["img_sz_min"], self.system_dict["params"]["img_sz_max"] + 1) * 32
                    sf = self.system_dict["params"]["img_size"] / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / 32.) * 32 for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                        imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Plot images with bounding boxes
                if ni == 0:
                    fname = 'train_batch%g.png' % i
                    plot_images(imgs=imgs, targets=targets, paths=paths, fname=fname)
                    if self.system_dict["local"]["tb_writer"]:
                        self.system_dict["local"]["tb_writer"].add_image(fname, cv2.imread(fname)[:, :, ::-1], dataformats='HWC')

                # Run model
                pred = self.system_dict["local"]["model"](imgs)

                # Compute loss
                loss, loss_items = compute_loss(pred, targets, self.system_dict["local"]["model"], not prebias)
                if not torch.isfinite(loss):
                    print('WARNING: non-finite loss, ending training ', loss_items)
                    return results

                # Scale loss by nominal batch_size of 64
                loss *= self.system_dict["params"]["batch_size"] / 64

                # Compute gradient
                if self.system_dict["params"]["mixed_precision"]:
                    with amp.scale_loss(loss, self.system_dict["local"]["optimizer"]) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # Accumulate gradient for x batches before optimizing
                if ni % self.system_dict["params"]["accumulate"] == 0:
                    self.system_dict["local"]["optimizer"].step()
                    self.system_dict["local"]["optimizer"].zero_grad()

                # Print batch results
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)
                s = ('%10s' * 2 + '%10.3g' * 6) % (
                    '%g/%g' % (epoch, self.system_dict["params"]["epochs"] - 1), '%.3gG' % mem, *mloss, len(targets), self.system_dict["params"]["img_size"])
                pbar.set_description(s)

                # end batch ------------------------------------------------------------------------------------------------

            # Process epoch results
            final_epoch = epoch + 1 == self.system_dict["params"]["epochs"]


            if(self.system_dict["dataset"]["val"]["status"]):
                if not self.system_dict["params"]["notest"] or final_epoch:  # Calculate mAP
                    is_coco = False
                    results, maps = validate(self.system_dict["params"]["cfg"],
                                                self.system_dict["dataset"]["val"]["img_dir"],
                                                self.system_dict["dataset"]["val"]["label_dir"],
                                                self.system_dict["local"]["classes"],
                                                batch_size=self.system_dict["params"]["batch_size"] * 2,
                                                img_size=self.system_dict["params"]["img_size_test"],
                                                model=self.system_dict["local"]["model"],
                                                conf_thres=0.001 if final_epoch and is_coco else 0.1,  # 0.1 for speed
                                                iou_thres=0.6,
                                                save_json=final_epoch and is_coco,
                                                single_cls=self.system_dict["params"]["single_cls"],
                                                dataloader=self.system_dict["local"]["testloader"])

            # Update scheduler
            self.system_dict["local"]["scheduler"].step()


            if(self.system_dict["dataset"]["val"]["status"]):
                # Write epoch results
                with open(self.system_dict["fixed_params"]["results_file"], 'a') as f:
                    f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
                if len(self.system_dict["params"]["name"]) and self.system_dict["params"]["bucket"]:
                    os.system('gsutil cp results.txt gs://%s/results%s.txt' % (self.system_dict["params"]["bucket"], 
                                                                                self.system_dict["params"]["name"]))

                # Write Tensorboard results
                if self.system_dict["local"]["tb_writer"]:
                    x = list(mloss) + list(results)
                    titles = ['GIoU', 'Objectness', 'Classification', 'Train loss',
                              'Precision', 'Recall', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification']
                    for xi, title in zip(x, titles):
                        self.system_dict["local"]["tb_writer"].add_scalar(title, xi, epoch)

                # Update best mAP
                fitness = sum(results[4:])  # total loss
                if fitness < self.system_dict["local"]["best_fitness"]:
                    self.system_dict["local"]["best_fitness"] = fitness


                # Save training results
                save = (not self.system_dict["params"]["nosave"]) or (final_epoch and not self.system_dict["params"]["evolve"])
                if save:
                    with open(self.system_dict["fixed_params"]["results_file"], 'r') as f:
                        # Create checkpoint
                        chkpt = {'epoch': epoch,
                                 'best_fitness': self.system_dict["local"]["best_fitness"],
                                 'training_results': f.read(),
                                 'model': self.system_dict["local"]["model"].module.state_dict() if type(
                                      self.system_dict["local"]["model"]) is nn.parallel.DistributedDataParallel else  self.system_dict["local"]["model"].state_dict(),
                                 'optimizer': None if final_epoch else  self.system_dict["local"]["optimizer"].state_dict()}

                    # Save last checkpoint
                    torch.save(chkpt, self.system_dict["fixed_params"]["last"])

                    # Save best checkpoint
                    if self.system_dict["local"]["best_fitness"] == fitness:
                        torch.save(chkpt, self.system_dict["fixed_params"]["best"])

                    # Save backup every 1 epochs (optional)
                    if epoch > 0 and epoch % 1 == 0:
                        torch.save(chkpt, self.system_dict["fixed_params"]["wdir"] + 'backup%g.pt' % epoch)

                    # Delete checkpoint
                    del chkpt

            else:
                chkpt = {'epoch': epoch,
                                 'best_fitness': 0.0,
                                 'training_results': "",
                                 'model': self.system_dict["local"]["model"].module.state_dict() if type(
                                      self.system_dict["local"]["model"]) is nn.parallel.DistributedDataParallel else  self.system_dict["local"]["model"].state_dict(),
                                 'optimizer': None if final_epoch else  self.system_dict["local"]["optimizer"].state_dict()}

                # Save last checkpoint
                torch.save(chkpt, self.system_dict["fixed_params"]["last"])


                # Save backup every 1 epochs (optional)
                if epoch > 0 and epoch % 1 == 0:
                    torch.save(chkpt, self.system_dict["fixed_params"]["wdir"] + 'backup%g.pt' % epoch)

                # Delete checkpoint
                del chkpt


        # end training
        if(self.system_dict["dataset"]["val"]["status"]):
            n = self.system_dict["params"]["name"]
            if len(n):
                n = '_' + n if not n.isnumeric() else n
                fresults, flast, fbest = 'results%s.txt' % n, 'last%s.pt' % n, 'best%s.pt' % n
                os.rename('results.txt', fresults)
                os.rename(self.system_dict["fixed_params"]["wdir"] + 'last.pt', 
                    self.system_dict["fixed_params"]["wdir"] + flast) if os.path.exists(self.system_dict["fixed_params"]["wdir"] + 'last.pt') else None
                os.rename(self.system_dict["fixed_params"]["wdir"] + 'best.pt', 
                    self.system_dict["fixed_params"]["wdir"] + fbest) if os.path.exists(self.system_dict["fixed_params"]["wdir"] + 'best.pt') else None

                # save to cloud
                if self.system_dict["params"]["bucket"]:
                    os.system('gsutil cp %s %s gs://%s' % (fresults, self.system_dict["fixed_params"]["wdir"] + flast, 
                        self.system_dict["params"]["bucket"]))

            if not self.system_dict["params"]["evolve"]:
                plot_results()  # save as results.png
            print('%g epochs completed in %.3f hours.\n' % (epoch - self.system_dict["local"]["start_epoch"] + 1, (time.time() - t0) / 3600))
            dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
            torch.cuda.empty_cache()

            return results

                
