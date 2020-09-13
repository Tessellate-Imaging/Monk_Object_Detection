import argparse
import torch
import os
import torch.backends.cudnn as cudnn

from datetime import datetime

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def arg2str(args):
    args_dict = vars(args)
    option_str = datetime.now().strftime('%b%d_%H-%M-%S') + '\n'

    for k, v in sorted(args_dict.items()):
        option_str += ('{}: {}\n'.format(str(k), str(v)))

    return option_str

class BaseOptions(object):

    def __init__(self):

        self.parser = argparse.ArgumentParser()

        # basic opts
        self.parser.add_argument('exp_name', type=str, help='Experiment name')
        self.parser.add_argument('--net', default='vgg', type=str, choices=['vgg', 'resnet'], help='Network architecture')
        self.parser.add_argument('--dataset', default='total-text', type=str, choices=['synth-text', 'total-text'], help='Dataset name')
        self.parser.add_argument('--resume', default=None, type=str, help='Path to target resume checkpoint')
        self.parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
        self.parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
        self.parser.add_argument('--mgpu', action='store_true', help='Use multi-gpu to train model')
        self.parser.add_argument('--save_dir', default='./save/', help='Path to save checkpoint models')
        self.parser.add_argument('--vis_dir', default='./vis/', help='Path to save visualization images')
        self.parser.add_argument('--log_dir', default='./logs/', help='Path to tensorboard log')
        self.parser.add_argument('--loss', default='CrossEntropyLoss', type=str, help='Training Loss')
        self.parser.add_argument('--input_channel', default=1, type=int, help='number of input channels' )
        self.parser.add_argument('--pretrain', default=False, type=str2bool, help='Pretrained AutoEncoder model')
        self.parser.add_argument('--verbose', '-v', default=True, type=str2bool, help='Whether to output debug info')
        self.parser.add_argument('--viz', action='store_true', help='Whether to output debug info')

        # train opts
        self.parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
        self.parser.add_argument('--max_epoch', default=200, type=int, help='Max epochs')
        self.parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
        self.parser.add_argument('--lr_adjust', default='fix', choices=['fix', 'poly'], type=str, help='Learning Rate Adjust Strategy')
        self.parser.add_argument('--stepvalues', default=[], nargs='+', type=int, help='# of iter to change lr')
        self.parser.add_argument('--weight_decay', '--wd', default=0., type=float, help='Weight decay for SGD')
        self.parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD lr')
        self.parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
        self.parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
        self.parser.add_argument('--optim', default='SGD', type=str, choices=['SGD', 'Adam'], help='Optimizer')
        self.parser.add_argument('--display_freq', default=50, type=int, help='display training metrics every # iterations')
        self.parser.add_argument('--viz_freq', default=50, type=int, help='visualize training process every # iterations')
        self.parser.add_argument('--save_freq', default=10, type=int, help='save weights every # epoch')
        self.parser.add_argument('--log_freq', default=100, type=int, help='log to tensorboard every # iterations')
        self.parser.add_argument('--val_freq', default=100, type=int, help='do validation every # iterations')

        # data args
        self.parser.add_argument('--rescale', type=float, default=255.0, help='rescale factor')
        self.parser.add_argument('--means', type=int, default=(0.485, 0.456, 0.406), nargs='+', help='mean')
        self.parser.add_argument('--stds', type=int, default=(0.229, 0.224, 0.225), nargs='+', help='std')
        self.parser.add_argument('--input_size', default=512, type=int, help='model input size')

        # eval args
        self.parser.add_argument('--checkepoch', default=-1, type=int, help='Load checkpoint number')

        # demo args
        self.parser.add_argument('--img_root', default=None, type=str, help='Path to deploy images')

    def parse(self, fixed=None):

        if fixed is not None:
            args = self.parser.parse_args(fixed)
        else:
            args = self.parser.parse_args()

        return args

    def initialize(self, fixed=None):

        # Parse options
        self.args = self.parse(fixed)

        # Setting default torch Tensor type
        if self.args.cuda and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            cudnn.benchmark = True
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        # Create weights saving directory
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        # Create weights saving directory of target model
        model_save_path = os.path.join(self.args.save_dir, self.args.exp_name)

        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)

        return self.args

    def update(self, args, extra_options):

        for k, v in extra_options.items():
            setattr(args, k, v)
