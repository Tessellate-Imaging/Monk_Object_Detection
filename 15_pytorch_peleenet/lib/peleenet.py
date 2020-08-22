# -*- coding:utf-8 -*-
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as init
from collections import OrderedDict

import math
from utils.core import print_info


class conv_bn_relu(nn.Module):
    """docstring for conv_bn_relu"""

    def __init__(self, in_channels, out_channels, activation=True, **kwargs):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        out = self.norm(self.conv(x))
        if self.activation:
            out = F.relu(out, inplace=True)
        return out


class conv_relu(nn.Module):
    """docstring for conv_relu"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              bias=False, **kwargs)

    def forward(self, x):
        out = F.relu(self.conv(x), inplace=True)
        return out


class _DenseLayer(nn.Module):
    """docstring for _DenseLayer"""

    def __init__(self, num_input_features, growth_rate, bottleneck_width, drop_rate):
        super(_DenseLayer, self).__init__()
        growth_rate = growth_rate // 2
        inter_channel = int(growth_rate * bottleneck_width / 4) * 4

        if inter_channel > num_input_features / 2:
            inter_channel = int(num_input_features / 8) * 4
            print('adjust inter_channel to ', inter_channel)

        self.branch1a = conv_bn_relu(
            num_input_features, inter_channel, kernel_size=1)
        self.branch1b = conv_bn_relu(
            inter_channel, growth_rate, kernel_size=3, padding=1)

        self.branch2a = conv_bn_relu(
            num_input_features, inter_channel, kernel_size=1)
        self.branch2b = conv_bn_relu(
            inter_channel, growth_rate, kernel_size=3, padding=1)
        self.branch2c = conv_bn_relu(
            growth_rate, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out1 = self.branch1a(x)
        out1 = self.branch1b(out1)

        out2 = self.branch2a(x)
        out2 = self.branch2b(out2)
        out3 = self.branch2c(out2)

        out = torch.cat([x, out1, out2], dim=1)
        return out


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i *
                                growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _StemBlock(nn.Module):

    def __init__(self, num_input_channels, num_init_features):
        super(_StemBlock, self).__init__()

        num_stem_features = int(num_init_features / 2)

        self.stem1 = conv_bn_relu(
            num_input_channels, num_init_features, kernel_size=3, stride=2, padding=1)
        self.stem2a = conv_bn_relu(
            num_init_features, num_stem_features, kernel_size=1, stride=1, padding=0)
        self.stem2b = conv_bn_relu(
            num_stem_features, num_init_features, kernel_size=3, stride=2, padding=1)
        self.stem3 = conv_bn_relu(
            2 * num_init_features, num_init_features, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, x):
        out = self.stem1(x)

        branch2 = self.stem2a(out)
        branch2 = self.stem2b(branch2)
        branch1 = self.pool(out)

        out = torch.cat([branch1, branch2], dim=1)
        out = self.stem3(out)

        return out


class ResBlock(nn.Module):
    """docstring for ResBlock"""

    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.res1a = conv_relu(in_channels, 128, kernel_size=1)
        self.res1b = conv_relu(128, 128, kernel_size=3, padding=1)
        self.res1c = conv_relu(128, 256, kernel_size=1)

        self.res2a = conv_relu(in_channels, 256, kernel_size=1)

    def forward(self, x):
        out1 = self.res1a(x)
        out1 = self.res1b(out1)
        out1 = self.res1c(out1)

        out2 = self.res2a(x)
        out = out1 + out2
        return out


class PeleeNet(nn.Module):
    r"""PeleeNet model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf> and
     "Pelee: A Real-Time Object Detection System on Mobile Devices" <https://arxiv.org/pdf/1608.06993.pdf>` 
    Args:
        growth_rate (int or list of 4 ints) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bottleneck_width (int or list of 4 ints) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, phase, size, cfg=None):

        super(PeleeNet, self).__init__()

        self.phase = phase
        self.size = size
        self.cfg = cfg

        self.features = nn.Sequential(OrderedDict([
            ('stemblock', _StemBlock(3, cfg.num_init_features)),
        ]))

        if type(cfg.growth_rate) is list:
            growth_rates = cfg.growth_rate
            assert len(
                growth_rates) == 4, 'The growth rate must be the list and the size must be 4'
        else:
            growth_rates = [cfg.growth_rate] * 4

        if type(cfg.bottleneck_width) is list:
            bottleneck_widths = cfg.bottleneck_width
            assert len(
                bottleneck_widths) == 4, 'The bottleneck width must be the list and the size must be 4'
        else:
            bottleneck_widths = [cfg.bottleneck_width] * 4

        # Each denseblock
        num_features = cfg.num_init_features
        for i, num_layers in enumerate(cfg.block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bottleneck_widths[i], growth_rate=growth_rates[i], drop_rate=cfg.drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rates[i]

            self.features.add_module('transition%d' % (i + 1), conv_bn_relu(
                num_features, num_features, kernel_size=1, stride=1, padding=0))

            if i != len(cfg.block_config) - 1:
                self.features.add_module('transition%d_pool' % (
                    i + 1), nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True))
                num_features = num_features

        extras = add_extras(704, batch_norm=True)
        self.extras = nn.ModuleList(extras)

        nchannels = [512, 704, 256, 256, 256]

        resblock = add_resblock(nchannels)
        self.resblock = nn.ModuleList(resblock)

        self.loc = nn.ModuleList()
        self.conf = nn.ModuleList()

        for i, x in enumerate([256] * 5):
            n = cfg.anchor_config.anchor_nums[i]
            #print(n, cfg.num_classes);
            self.loc.append(nn.Conv2d(x, n * 4, kernel_size=1))
            self.conf.append(nn.Conv2d(x, n * cfg.num_classes, kernel_size=1))

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()
        for k, feat in enumerate(self.features):
            x = feat(x)
            if k == 8 or k == len(self.features) - 1:
                sources += [x]

        for k, v in enumerate(self.extras):
            x = v(x)
            if k % 2 == 1:
                sources += [x]

        for k, x in enumerate(sources):
            sources[k] = self.resblock[k](x)

        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == 'test':
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.cfg.num_classes))  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.cfg.num_classes)
            )
        return output

    def init_model(self, pretained_model, use_gpu):
        if(not use_gpu):
            base_state = torch.load(pretained_model, map_location='cpu');
        else:
            base_state = torch.load(pretained_model);
        self.features.load_state_dict(base_state)
        print_info('Loading base network...')

        def weights_init(m):
            '''
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        init.kaiming_normal_(
                            m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = 1
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0
            '''
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if 'bias' in m.state_dict().keys():
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        print_info(
            'Initializing weights for [extras, resblock,multibox]...')
        self.extras.apply(weights_init)
        self.resblock.apply(weights_init)
        self.loc.apply(weights_init)
        self.conf.apply(weights_init)

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print_info('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print_info('Finished!')
        else:
            print_info('Sorry only .pth and .pkl files supported.')


def add_extras(i, batch_norm=False):
    layers = []
    in_channels = i
    channels = [128, 256, 128, 256, 128, 256]
    stride = [1, 2, 1, 1, 1, 1]
    padding = [0, 1, 0, 0, 0, 0]

    for k, v in enumerate(channels):
        if k % 2 == 0:
            if batch_norm:
                layers += [conv_bn_relu(in_channels, v,
                                        kernel_size=1, padding=padding[k])]
            else:
                layers += [conv_relu(in_channels, v,
                                     kernel_size=1, padding=padding[k])]
        else:
            if batch_norm:
                layers += [conv_bn_relu(in_channels, v,
                                        kernel_size=3, stride=stride[k], padding=padding[k])]
            else:
                layers += [conv_relu(in_channels, v,
                                     kernel_size=3, stride=stride[k], padding=padding[k])]
        in_channels = v

    return layers


def add_resblock(nchannels):
    layers = []
    for k, v in enumerate(nchannels):
        layers += [ResBlock(v)]
    return layers


def build_net(phase, size, config=None):
    if not phase in ['test', 'train']:
        raise ValueError("Error: Phase not recognized")

    if size != 304:
        raise NotImplementedError(
            "Error: Sorry only Pelee300 are supported!")

    return PeleeNet(phase, size, config)

if __name__ == '__main__':
    net = PeleeNet()
    print(net)
    # net.features.load_state_dict(torch.load('./peleenet.pth'))
    state_dict = torch.load('./weights/peleenet.pth')
    # print(state_dict.keys())
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[9:]] = v

    torch.save(new_state_dict, './weights/peleenet_new.pth')
    net.features.load_state_dict(new_state_dict)
    inputs = torch.randn(2, 3, 304, 304)
    out = net(inputs)
    # print(out.size())
