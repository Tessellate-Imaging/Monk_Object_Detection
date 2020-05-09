import os
import json
import torch
import numpy as np
import queue
import pprint
import random 
import argparse
import importlib
import threading
import traceback
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from torch.multiprocessing import Process, Queue, Pool

from core.dbs import datasets
from core.utils import stdout_to_tqdm
from core.config import SystemConfig
from core.sample import data_sampling_func
from core.nnet.py_factory import NetworkFactory


def prefetch_data(system_config, db, queue, sample_data, data_aug):
    ind = 0
    print("start prefetching data...")
    np.random.seed(os.getpid())
    while True:
        try:
            data, ind = sample_data(system_config, db, ind, data_aug=data_aug)
            queue.put(data)
        except Exception as e:
            traceback.print_exc()
            raise e
            
def _pin_memory(ts):
    if type(ts) is list:
        return [t.pin_memory() for t in ts]
    return ts.pin_memory()


def pin_memory(data_queue, pinned_data_queue, sema):
    while True:
        data = data_queue.get()

        data["xs"] = [_pin_memory(x) for x in data["xs"]]
        data["ys"] = [_pin_memory(y) for y in data["ys"]]

        pinned_data_queue.put(data)

        if sema.acquire(blocking=False):
            return

        
def init_parallel_jobs(system_config, dbs, queue, fn, data_aug):
    tasks = [Process(target=prefetch_data, args=(system_config, db, queue, fn, data_aug)) for db in dbs]
    for task in tasks:
        task.daemon = True
        task.start()
    return tasks

def terminate_tasks(tasks):
    for task in tasks:
        task.terminate()


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
        self.system_dict["dataset"] = {};
        self.system_dict["dataset"]["train"] = {};
        self.system_dict["dataset"]["val"] = {};
        self.system_dict["dataset"]["val"]["status"] = False;
        self.system_dict["dataset"]["params"] = {};
        self.system_dict["dataset"]["params"]["workers"] = 4;


        self.system_dict["model"] = {};
        self.system_dict["model"]["params"] = {};
        self.system_dict["model"]["params"]["cfg_file"] = "CornerNet_Saccade";
        self.system_dict["model"]["params"]["initialize"] = False;
        self.system_dict["model"]["params"]["distributed"] = False;
        self.system_dict["model"]["params"]["world_size"] = 0;
        self.system_dict["model"]["params"]["rank"] = 0;
        self.system_dict["model"]["params"]["dist_url"] = None;
        self.system_dict["model"]["params"]["dist_backend"] = "nccl";
        self.system_dict["model"]["params"]["use_gpu"] = True;

        self.system_dict["training"] = {};
        self.system_dict["training"]["params"] = {};
        self.system_dict["training"]["params"]["start_iter"] = 0;
        self.system_dict["training"]["params"]["gpu"] = None;
        


    def Train_Dataset(self, root_dir, coco_dir, img_dir, set_dir, batch_size=4, use_gpu=True, num_workers=4):
        '''
        User function: Set training dataset parameters

        Dataset Directory Structure

                   root_dir
                      |
                      |------coco_dir 
                      |         |
                      |         |----img_dir
                      |                |
                      |                |------<set_dir_train> (set_dir) (Train)
                      |                         |
                      |                         |---------img1.jpg
                      |                         |---------img2.jpg
                      |                         |---------..........(and so on)  
                      |
                      |
                      |         |---annotations 
                      |         |----|
                      |              |--------------------instances_Train.json  (instances_<set_dir_train>.json)
                      |              |--------------------classes.txt
                      
                      
             - instances_Train.json -> In proper COCO format
             - classes.txt          -> A list of classes in alphabetical order
             

            For TrainSet
             - root_dir = "../sample_dataset";
             - coco_dir = "kangaroo";
             - img_dir = "images";
             - set_dir = "Train";
            
             
            Note: Annotation file name too coincides against the set_dir

        Args:
            root_dir (str): Path to root directory containing coco_dir
            coco_dir (str): Name of coco_dir containing image folder and annotation folder
            img_dir (str): Name of folder containing all training and validation folders
            set_dir (str): Name of folder containing all training images
            batch_size (int): Mini batch sampling size for training epochs
            use_gpu (bool): If True use GPU else run on CPU
            num_workers (int): Number of parallel processors for data loader 

        Returns:
            None
        '''
        self.system_dict["dataset"]["train"]["root_dir"] = root_dir;
        self.system_dict["dataset"]["train"]["coco_dir"] = coco_dir;
        self.system_dict["dataset"]["train"]["img_dir"] = img_dir;
        self.system_dict["dataset"]["train"]["set_dir"] = set_dir;

        self.system_dict["dataset"]["params"]["batch_size"] = batch_size;
        self.system_dict["dataset"]["params"]["workers"] = num_workers;

        self.system_dict["model"]["params"]["use_gpu"] = use_gpu;



    def Val_Dataset(self, root_dir, coco_dir, img_dir, set_dir):
        '''
        User function: Set training dataset parameters

        Dataset Directory Structure

                   root_dir
                      |
                      |------coco_dir 
                      |         |
                      |         |----img_dir
                      |                |
                      |                |------<set_dir_val> (set_dir) (Validation)
                      |                         |
                      |                         |---------img1.jpg
                      |                         |---------img2.jpg
                      |                         |---------..........(and so on)  
                      |
                      |
                      |         |---annotations 
                      |         |----|
                      |              |--------------------instances_Val.json  (instances_<set_dir_val>.json)
                      |              |--------------------classes.txt
                      
                      
             - instances_Train.json -> In proper COCO format
             - classes.txt          -> A list of classes in alphabetical order

             
            For ValSet
             - root_dir = "..sample_dataset";
             - coco_dir = "kangaroo";
             - img_dir = "images";
             - set_dir = "Val";
             
             Note: Annotation file name too coincides against the set_dir

        Args:
            root_dir (str): Path to root directory containing coco_dir
            coco_dir (str): Name of coco_dir containing image folder and annotation folder
            img_dir (str): Name of folder containing all training and validation folders
            set_dir (str): Name of folder containing all validation images

        Returns:
            None
        '''
        self.system_dict["dataset"]["val"]["status"] = True;
        self.system_dict["dataset"]["val"]["root_dir"] = root_dir;
        self.system_dict["dataset"]["val"]["coco_dir"] = coco_dir;
        self.system_dict["dataset"]["val"]["img_dir"] = img_dir;
        self.system_dict["dataset"]["val"]["set_dir"] = set_dir;



    def Model(self, model_name="CornerNet_Saccade", use_distributed=False):
        '''
        User function: Set Model parameters

            Available Models
                CornerNet_Saccade
                CornerNet_Squeeze

        Args:
            model_name (str): Select appropriate model
            use_distributed (bool): If true, use distributed training

        Returns:
            None
        '''
        self.system_dict["model"]["params"]["cfg_file"] = model_name;
        self.system_dict["model"]["params"]["distributed"] = use_distributed;

        if(self.system_dict["model"]["params"]["distributed"]):
            print("Distributed training not enabled yet");


    def Hyper_Params(self, lr=0.00025, total_iterations=1000, val_interval=500):
        '''
        User function: Set hyper parameters

        Args:
            lr (float): Initial learning rate for training
            total_iterations (float): Total mini batch iterations for training
            val_interval (int): Post specified number of training epochs, a validation epoch will be carried out

        Returns:
            None
        '''
        self.system_dict["training"]["params"]["lr"] = lr;
        self.system_dict["training"]["params"]["total_iterations"] = total_iterations;
        self.system_dict["training"]["params"]["val_interval"] = val_interval;



    def Setup(self):
        '''
        User function: Setup dataset, model and hyper-params 

        Args:
            None

        Returns:
            None
        '''
        distributed = self.system_dict["model"]["params"]["distributed"]
        world_size  = self.system_dict["model"]["params"]["world_size"]

        ngpus_per_node  = torch.cuda.device_count()

        current_dir = os.path.dirname(os.path.realpath(__file__));

        cfg_file = os.path.join(current_dir, "configs", self.system_dict["model"]["params"]["cfg_file"] + ".json")
        with open(cfg_file, "r") as f:
            self.system_dict["local"]["config"] = json.load(f)

        self.system_dict["local"]["config"]["db"]["root_dir"] = self.system_dict["dataset"]["train"]["root_dir"];
        self.system_dict["local"]["config"]["db"]["coco_dir"] = self.system_dict["dataset"]["train"]["coco_dir"];
        self.system_dict["local"]["config"]["db"]["img_dir"] = self.system_dict["dataset"]["train"]["img_dir"];
        self.system_dict["local"]["config"]["db"]["set_dir"] = self.system_dict["dataset"]["train"]["set_dir"];

        f = open(self.system_dict["dataset"]["train"]["root_dir"] + "/" + self.system_dict["dataset"]["train"]["coco_dir"] + "/annotations/classes.txt");
        lines = f.readlines();
        f.close();

        self.system_dict["local"]["config"]["db"]["categories"] = len(lines);

        self.system_dict["local"]["config"]["system"]["batch_size"] = self.system_dict["dataset"]["params"]["batch_size"];
        self.system_dict["local"]["config"]["system"]["chunk_sizes"] = [self.system_dict["dataset"]["params"]["batch_size"]];
        self.system_dict["local"]["config"]["system"]["max_iter"] = self.system_dict["training"]["params"]["total_iterations"];

        self.system_dict["local"]["config"]["system"]["snapshot_name"] = self.system_dict["model"]["params"]["cfg_file"]
        self.system_dict["local"]["system_config"] = SystemConfig().update_config(self.system_dict["local"]["config"]["system"])

        self.system_dict["local"]["training_dbs"] = [datasets[self.system_dict["local"]["system_config"].dataset](self.system_dict["local"]["config"]["db"], 
                                                            sys_config=self.system_dict["local"]["system_config"]) for _ in range(self.system_dict["dataset"]["params"]["workers"])]

        if(self.system_dict["dataset"]["val"]["status"]):
            self.system_dict["local"]["config"]["db"]["root_dir"] = self.system_dict["dataset"]["val"]["root_dir"];
            self.system_dict["local"]["config"]["db"]["coco_dir"] = self.system_dict["dataset"]["val"]["coco_dir"];
            self.system_dict["local"]["config"]["db"]["img_dir"] = self.system_dict["dataset"]["val"]["img_dir"];
            self.system_dict["local"]["config"]["db"]["set_dir"] = self.system_dict["dataset"]["val"]["set_dir"];

            self.system_dict["local"]["validation_db"] = datasets[self.system_dict["local"]["system_config"].dataset](self.system_dict["local"]["config"]["db"], 
                                                                sys_config=self.system_dict["local"]["system_config"])


        if(not os.path.isdir("cache/")):
            os.mkdir("cache");
        if(not os.path.isdir("cache/nnet")):
            os.mkdir("cache/nnet/");
        if(not os.path.isdir("cache/nnet/" + self.system_dict["model"]["params"]["cfg_file"])):
            os.mkdir("cache/nnet/" + self.system_dict["model"]["params"]["cfg_file"]);

        model_file  = "core.models.{}".format(self.system_dict["model"]["params"]["cfg_file"])
        print("Loading Model - {}".format(model_file))
        model_file  = importlib.import_module(model_file)
        self.system_dict["local"]["model"] = model_file.model(self.system_dict["local"]["config"]["db"]["categories"])
        print("Model Loaded");


    def Train(self, display_interval=100):
        '''
        User function: Start training

        Args:
            display_interval (int): Post every specified iteration the training losses and accuracies will be printed

        Returns:
            None
        '''
                # reading arguments from command
        start_iter  = self.system_dict["training"]["params"]["start_iter"]
        distributed = self.system_dict["model"]["params"]["distributed"]
        world_size  = self.system_dict["model"]["params"]["world_size"]
        initialize  = self.system_dict["model"]["params"]["initialize"]
        gpu         = None
        rank        = self.system_dict["model"]["params"]["rank"]

        # reading arguments from json file
        batch_size       = self.system_dict["dataset"]["params"]["batch_size"]
        learning_rate    = self.system_dict["training"]["params"]["lr"]
        max_iteration    = self.system_dict["training"]["params"]["total_iterations"]
        pretrained_model = None;

        stepsize         = int(self.system_dict["training"]["params"]["total_iterations"]*0.8)
        snapshot         = int(self.system_dict["training"]["params"]["total_iterations"]*0.5)
        val_iter         = self.system_dict["training"]["params"]["val_interval"]
        display          = display_interval
        decay_rate       = self.system_dict["local"]["system_config"].decay_rate

        print("start_iter       = {}".format(start_iter));
        print("distributed      = {}".format(distributed));
        print("world_size       = {}".format(world_size));
        print("initialize       = {}".format(initialize));
        print("batch_size       = {}".format(batch_size));
        print("learning_rate    = {}".format(learning_rate));
        print("max_iteration    = {}".format(max_iteration));
        print("stepsize         = {}".format(stepsize));
        print("snapshot         = {}".format(snapshot));
        print("val_iter         = {}".format(val_iter));
        print("display          = {}".format(display));
        print("decay_rate       = {}".format(decay_rate));



        print("Process {}: building model...".format(rank))
        self.system_dict["local"]["nnet"] = NetworkFactory(self.system_dict["local"]["system_config"], 
                                self.system_dict["local"]["model"], distributed=distributed, gpu=gpu)


        # queues storing data for training
        training_queue   = Queue(self.system_dict["local"]["system_config"].prefetch_size)
        validation_queue = Queue(5)

        # queues storing pinned data for training
        pinned_training_queue   = queue.Queue(self.system_dict["local"]["system_config"].prefetch_size)
        pinned_validation_queue = queue.Queue(5)


        # allocating resources for parallel reading
        training_tasks = init_parallel_jobs(self.system_dict["local"]["system_config"], 
                                            self.system_dict["local"]["training_dbs"], 
                                            training_queue, data_sampling_func, True)


        
        if self.system_dict["dataset"]["val"]["status"]:
            validation_tasks = init_parallel_jobs(self.system_dict["local"]["system_config"], 
                                                    [self.system_dict["local"]["validation_db"]], 
                                                    validation_queue, data_sampling_func, False)


        training_pin_semaphore   = threading.Semaphore()
        validation_pin_semaphore = threading.Semaphore()
        training_pin_semaphore.acquire()
        validation_pin_semaphore.acquire()

        training_pin_args   = (training_queue, pinned_training_queue, training_pin_semaphore)
        training_pin_thread = threading.Thread(target=pin_memory, args=training_pin_args)
        training_pin_thread.daemon = True
        training_pin_thread.start()

        validation_pin_args   = (validation_queue, pinned_validation_queue, validation_pin_semaphore)
        validation_pin_thread = threading.Thread(target=pin_memory, args=validation_pin_args)
        validation_pin_thread.daemon = True
        validation_pin_thread.start()
        
        if pretrained_model is not None:
            if not os.path.exists(pretrained_model):
                raise ValueError("pretrained model does not exist")
            print("Process {}: loading from pretrained model".format(rank))
            self.system_dict["local"]["nnet"].load_pretrained_params(pretrained_model)

        if start_iter:
            self.system_dict["local"]["nnet"].load_params(start_iter)
            learning_rate /= (decay_rate ** (start_iter // stepsize))
            self.system_dict["local"]["nnet"].set_lr(learning_rate)
            print("Process {}: training starts from iteration {} with learning_rate {}".format(rank, start_iter + 1, learning_rate))
        else:
            self.system_dict["local"]["nnet"].set_lr(learning_rate)


        if rank == 0:
            print("training start...")

        self.system_dict["local"]["nnet"].cuda()
        self.system_dict["local"]["nnet"].train_mode()   

        if(self.system_dict["dataset"]["val"]["status"]):
            old_val_loss = 100000.0;
            with stdout_to_tqdm() as save_stdout:
                for iteration in tqdm(range(start_iter + 1, max_iteration + 1), file=save_stdout, ncols=80):
                    training = pinned_training_queue.get(block=True)
                    training_loss = self.system_dict["local"]["nnet"].train(**training)

                    if display and iteration % display == 0:
                        print("Process {}: training loss at iteration {}: {}".format(rank, iteration, training_loss.item()))
                    del training_loss

                    if val_iter and self.system_dict["local"]["validation_db"].db_inds.size and iteration % val_iter == 0:
                        self.system_dict["local"]["nnet"].eval_mode()
                        validation = pinned_validation_queue.get(block=True)
                        validation_loss = self.system_dict["local"]["nnet"].validate(**validation)
                        print("Process {}: validation loss at iteration {}: {}".format(rank, iteration, validation_loss.item()))
                        if(validation_loss < old_val_loss):
                            print("Loss Reduced from {} to {}".format(old_val_loss, validation_loss))
                            self.system_dict["local"]["nnet"].save_params("best");
                            old_val_loss = validation_loss;
                        else:
                            print("validation loss did not go below {}, current loss - {}".format(old_val_loss, validation_loss))

                        self.system_dict["local"]["nnet"].train_mode()
                        

                    if iteration % stepsize == 0:
                        learning_rate /= decay_rate
                        self.system_dict["local"]["nnet"].set_lr(learning_rate)

            self.system_dict["local"]["nnet"].save_params("final");

            # sending signal to kill the thread
            training_pin_semaphore.release()
            validation_pin_semaphore.release()

            # terminating data fetching processes
            terminate_tasks(training_tasks)
            terminate_tasks(validation_tasks)


        else:
            with stdout_to_tqdm() as save_stdout:
                for iteration in tqdm(range(start_iter + 1, max_iteration + 1), file=save_stdout, ncols=80):
                    training = pinned_training_queue.get(block=True)
                    training_loss = self.system_dict["local"]["nnet"].train(**training)

                    if display and iteration % display == 0:
                        print("Process {}: training loss at iteration {}: {}".format(rank, iteration, training_loss.item()))
                    del training_loss


                    if(iteration % val_iter == 0):
                        self.system_dict["local"]["nnet"].save_params("intermediate");                     

                    if iteration % stepsize == 0:
                        learning_rate /= decay_rate
                        self.system_dict["local"]["nnet"].set_lr(learning_rate)

            self.system_dict["local"]["nnet"].save_params("final");

            # sending signal to kill the thread
            training_pin_semaphore.release()

            # terminating data fetching processes
            terminate_tasks(training_tasks)
    


