import argparse
import time

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import test  # import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    mixed_precision = False  # not installed

from tqdm import tqdm
from update_cfg import update
from torch.utils.tensorboard import SummaryWriter
from validate import validate
from prune import *

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

    def updateBN(self, scale, model):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.grad.data.add_(scale*torch.sign(m.weight.data))

    def set_fixed_params(self):

        self.system_dict["fixed_params"]["wdir"] = 'weights' + os.sep  
        self.system_dict["fixed_params"]["last"] = self.system_dict["fixed_params"]["wdir"] + 'last.pt'
        self.system_dict["fixed_params"]["best"] = self.system_dict["fixed_params"]["wdir"] + 'best.pt'
        self.system_dict["fixed_params"]["results_file"] = 'results.txt'

        self.system_dict["fixed_params"]["hyp"] = {'giou': 1.582,  # giou loss gain
                                                   'cls': 27.76,  # cls loss gain  (CE=~1.0, uCE=~20)
                                                   'cls_pw': 1.446,  # cls BCELoss positive_weight
                                                   'obj': 21.35,  # obj loss gain (*=80 for uBCE with 80 classes)
                                                   'obj_pw': 3.941,  # obj BCELoss positive_weight
                                                   'iou_t': 0.2635,  # iou training threshold
                                                   'lr0': 0.0002324,  # initial learning rate
                                                   'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
                                                   'momentum': 0.97,  # SGD momentum
                                                   'weight_decay': 0.0004569,  # optimizer weight decay
                                                   'hsv_s': 0.5703,  # image HSV-Saturation augmentation (fraction)
                                                   'hsv_v': 0.3174,  # image HSV-Value augmentation (fraction)
                                                   'degrees': 1.113,  # image rotation (+/- deg)
                                                   'translate': 0.06797,  # image translation (+/- fraction)
                                                   'scale': 0.1059,  # image scale (+/- gain)
                                                   'shear': 0.5768}  # image shear (+/- deg)


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
        self.system_dict["params"]["img_size"] = img_size;
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


    def set_hyperparams(self, optimizer="sgd", lr=0.0002324, multi_scale=False, evolve=False, num_generations=2, 
                        mixed_precision=True, gpu_devices="0", sparsity=0):
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
        self.system_dict["params"]["sparsity"] = sparsity;


    def setup_and_train(self, finetune=False):
        if(not finetune):
            model_name = "yolov3-spp3"
            tmp_cfg = os.path.dirname(os.path.realpath(__file__)) + "/cfg/" + model_name + ".cfg";
            cmd = "cp " + tmp_cfg + " " + os.getcwd() + "/" + model_name + ".cfg";
            os.system(cmd);
            self.system_dict["params"]["cfg"] = model_name + ".cfg";

            #update(self.system_dict["params"]["cfg"], self.system_dict["local"]["num_classes"]);
            #attempt_download('darknet53.conv.74')


        if(not os.path.isdir("weights")):
            os.mkdir("weights");

        cfg = self.system_dict["params"]["cfg"]
        img_size = self.system_dict["params"]["img_size"]
        epochs = self.system_dict["params"]["epochs"]  
        batch_size = self.system_dict["params"]["batch_size"]
        accumulate = self.system_dict["params"]["accumulate"] 
        weights = self.system_dict["params"]["weights"]
        sparsity = self.system_dict["params"]["sparsity"]
        arc = self.system_dict["params"]["arc"];
        hyp = self.system_dict["fixed_params"]["hyp"]
        rect = self.system_dict["params"]["rect"]
        cache_images = self.system_dict["params"]["cache_images"]
        batch_size = self.system_dict["params"]["batch_size"]
        train_img_dir = self.system_dict["dataset"]["train"]["img_dir"]
        train_label_dir = self.system_dict["dataset"]["train"]["label_dir"]
        val_img_dir = self.system_dict["dataset"]["val"]["img_dir"]
        val_label_dir = self.system_dict["dataset"]["val"]["label_dir"]


        # Initialize
        init_seeds()
        wdir = self.system_dict["fixed_params"]["wdir"]
        last = self.system_dict["fixed_params"]["last"]
        best = self.system_dict["fixed_params"]["best"]
        mixed_precision = self.system_dict["params"]["mixed_precision"]
        device = torch_utils.select_device(apex=mixed_precision)
        multi_scale = self.system_dict["params"]["multi_scale"]

        if multi_scale:
            img_sz_min = round(img_size / 32 / 1.5) + 1
            img_sz_max = round(img_size / 32 * 1.5) - 1
            img_size = img_sz_max * 32  # initiate with maximum multi_scale size
            print('Using multi-scale %g - %g' % (img_sz_min * 32, img_size))

        f = open(self.system_dict["dataset"]["train"]["class_list"], 'r');
        lines = f.readlines();
        f.close();
        self.system_dict["local"]["classes"] = [];
        for i in range(len(lines)):
            if(lines[i] != "" and lines[i] != "\n" ):
                self.system_dict["local"]["classes"].append(lines[i]);
        self.system_dict["local"]["num_classes"] = int(len(self.system_dict["local"]["classes"]));

        if(not finetune):
            update(self.system_dict["params"]["cfg"], self.system_dict["local"]["num_classes"]);
            attempt_download('darknet53.conv.74');
            weights="darknet53.conv.74";

        classes = self.system_dict["local"]["classes"];
        nc = self.system_dict["local"]["num_classes"];

        # Initialize model
        model = Darknet(cfg, arc=arc).to(device)

        if self.system_dict["params"]["adam"]:
            optimizer = optim.Adam(model.parameters(), lr=hyp['lr0'], weight_decay=hyp['weight_decay'])
        else:
            optimizer = optim.SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay'],
                          nesterov=True)

        cutoff = -1  # backbone reaches to cutoff layer
        start_epoch = 0
        best_fitness = 0.
        if weights.endswith('.pt'):  # pytorch format
            # possible weights are 'last.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc
            chkpt = torch.load(weights, map_location=device)

            # load model
            if not finetune:
                chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
                model.load_state_dict(chkpt['model'], strict=False)
            else:
                model.load_state_dict(chkpt['model'])

            # load optimizer
            if chkpt['optimizer'] is not None:
                optimizer.load_state_dict(chkpt['optimizer'])
                best_fitness = chkpt['best_fitness']

            del chkpt
        elif len(weights) > 0:  # darknet format
            # possible weights are 'yolov3.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
            cutoff = load_darknet_weights(model, weights)


        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(epochs * x) for x in [0.8, 0.9]], gamma=0.1)
        scheduler.last_epoch = start_epoch - 1

        if mixed_precision:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

        if torch.cuda.device_count() > 1:
            dist.init_process_group(backend='nccl',  # 'distributed backend'
                                    init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                    world_size=1,  # number of nodes for distributed training
                                    rank=0)  # distributed training node rank
            model = torch.nn.parallel.DistributedDataParallel(model)
            model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

        dataset = LoadImagesAndLabels(train_img_dir, 
                                        train_label_dir,
                                        img_size, 
                                        batch_size,
                                        augment=True,
                                        hyp=hyp, 
                                        rect=rect,
                                        cache_images=cache_images);

        dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=min(os.cpu_count(), batch_size),
                                             shuffle=not rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

        # Remove previous results
        for f in glob.glob('*_batch*.jpg') + glob.glob('results.txt'):
            os.remove(f)

        tb_writer = SummaryWriter()

        # Start training
        model.nc = nc  # attach number of classes to model
        model.arc = arc  # attach yolo architecture
        model.hyp = hyp  # attach hyperparameters to model
        model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
        model_info(model, report='summary')  # 'full' or 'summary'
        nb = len(dataloader)
        maps = np.zeros(nc)  # mAP per class
        results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
        t0 = time.time()
        for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
            model.train()
            print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))

            # Update scheduler
            if epoch > 0:
                scheduler.step()

            # Freeze backbone at epoch 0, unfreeze at epoch 1 (optional)
            freeze_backbone = False
            if freeze_backbone and epoch < 2:
                for name, p in model.named_parameters():
                    if int(name.split('.')[1]) < cutoff:  # if layer < 75
                        p.requires_grad = False if epoch == 0 else True

            # Update image weights (optional)
            if dataset.image_weights:
                w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
                image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
                dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx

            mloss = torch.zeros(4).to(device)  # mean losses
            pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
            for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(device)
                targets = targets.to(device)

                # Multi-Scale training
                if multi_scale:
                    if ni / accumulate % 10 == 0:  #  adjust (67% - 150%) every 10 batches
                        img_size = random.randrange(img_sz_min, img_sz_max + 1) * 32
                    sf = img_size / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / 32.) * 32 for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                        imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Plot images with bounding boxes
                if ni == 0:
                    fname = 'train_batch%g.jpg' % i
                    plot_images(imgs=imgs, targets=targets, paths=paths, fname=fname)
                    if tb_writer:
                        tb_writer.add_image(fname, cv2.imread(fname)[:, :, ::-1], dataformats='HWC')

                # Hyperparameter burn-in
                # n_burn = nb - 1  # min(nb // 5 + 1, 1000)  # number of burn-in batches
                # if ni <= n_burn:
                #     for m in model.named_modules():
                #         if m[0].endswith('BatchNorm2d'):
                #             m[1].momentum = 1 - i / n_burn * 0.99  # BatchNorm2d momentum falls from 1 - 0.01
                #     g = (i / n_burn) ** 4  # gain rises from 0 - 1
                #     for x in optimizer.param_groups:
                #         x['lr'] = hyp['lr0'] * g
                #         x['weight_decay'] = hyp['weight_decay'] * g

                # Run model
                pred = model(imgs)

                # Compute loss
                loss, loss_items = compute_loss(pred, targets, model)
                if torch.isnan(loss):
                    print('WARNING: nan loss detected, ending training')
                    return results

                # Divide by accumulation count
                if accumulate > 1:
                    loss /= accumulate

                # Compute gradient
                if mixed_precision:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # Accumulate gradient for x batches before optimizing
                if ni % accumulate == 0:
                    if sparsity != 0:
                        self.updateBN(sparsity, model)
                    optimizer.step()
                    optimizer.zero_grad()

                # Print batch results
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)
                s = ('%10s' * 2 + '%10.3g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, *mloss, len(targets), img_size)
                pbar.set_description(s)  # end batch -----------------------------------------------------------------------

            final_epoch = epoch + 1 == epochs
            
            with torch.no_grad():
                results, maps = validate(cfg,
                                            val_img_dir,
                                            val_label_dir,
                                            classes,
                                            batch_size=batch_size,
                                            img_size=img_size,
                                            model=model,
                                            conf_thres=0.001 if final_epoch and epoch > 0 else 0.1,  # 0.1 for speed
                                            save_json=False)

            # Write epoch results
            with open('results.txt', 'a') as file:
                file.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

            # Write Tensorboard results
            if tb_writer:
                x = list(mloss) + list(results)
                titles = ['GIoU', 'Objectness', 'Classification', 'Train loss',
                          'Precision', 'Recall', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification']
                for xi, title in zip(x, titles):
                    tb_writer.add_scalar(title, xi, epoch)

            # Update best mAP
            fitness = results[2]  # mAP
            if fitness > best_fitness:
                best_fitness = fitness

            # Save training results
            save = True
            if save:
                with open('results.txt', 'r') as file:
                    # Create checkpoint
                    chkpt = {'epoch': epoch,
                             'best_fitness': best_fitness,
                             'training_results': file.read(),
                             'model': model.module.state_dict() if type(
                                 model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                             'optimizer': None if final_epoch else optimizer.state_dict()}

                # Save last checkpoint
                torch.save(chkpt, last)
                
                # Save best checkpoint
                if best_fitness == fitness:
                    torch.save(chkpt, best)

                # Delete checkpoint
                del chkpt  # end epoch -------------------------------------------------------------------------------------

        # Report time
        plot_results()  # save as results.png
        print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
        torch.cuda.empty_cache()
        return results




    def Train(self, num_epochs=2, finetune=False):
        '''
        User function: Set training params and train

        Args:
            num_epochs (int): Number of epochs in training

        Returns:
            None
        '''

        self.system_dict["params"]["epochs"] = num_epochs;

        if not self.system_dict["params"]["evolve"]:
            result = self.setup_and_train(finetune=finetune);
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
                    result = self.setup_and_train(finetune=finetune);
                    self.system_dict["params"]["img_size"] = self.system_dict["params"]["img_size_selected"];

                    # Write mutation results
                    print_mutation(self.system_dict["fixed_params"]["hyp"], results, self.system_dict["params"]["bucket"])

                    # Plot results
                    plot_evolution_results(self.system_dict["fixed_params"]["hyp"])


    def prune_weights(self, input_cfg, input_weights, output_cfg, output_weights, img_size=406, overall_ratio=0.5, per_layer_ratio=0.1):
        with torch.no_grad():
            prune_weights(input_cfg,
                            input_weights,
                            output_cfg,
                            output_weights,
                            img_size=img_size,
                            save="weights",
                            overall_ratio=overall_ratio,
                            perlayer_ratio=per_layer_ratio
                        );


    def set_finetune_params(self, 
                            input_cfg, 
                            input_weights):

        self.system_dict["params"]["cfg"] = input_cfg;
        self.system_dict["params"]["weights"] = input_weights;