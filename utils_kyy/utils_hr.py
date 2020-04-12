import os
import sys

sys.path.insert(0,'../')

import torch
import torch.nn as nn
import numpy as np

from utils_kyy import models

import torch
import torch.nn as nn

import math

from utils_kyy.utils_graph import load_graph, get_graph_info


class depthwise_separable_conv_3x3(nn.Module):
    def __init__(self, nin, nout, stride):
        # input node 일때, stride = 1; => size 유지
        # input node 아닐 대, stride = 2; =>  (x-1)/2 + 1
        super(depthwise_separable_conv_3x3, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, stride=stride, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout,
                                   kernel_size=1)  # default: stride=1, padding=0, dilation=1, groups=1, bias=True

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class conv2d_3x3(nn.Module):
    def __init__(self, nin, nout, stride):
        super(conv2d_3x3, self).__init__()
        self.conv = nn.Conv2d(nin, nout, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        out = self.conv(x)
        return out


class depthwise_conv_3x3(nn.Module):
    def __init__(self, nin, nout, stride):
        # input node 일때, stride = 1; => size 유지
        # input node 아닐 대, stride = 2; =>  (x-1)/2 + 1
        super(depthwise_conv_3x3, self).__init__()
        self.depthwise = nn.Conv2d(nin, nout, kernel_size=3, stride=stride, padding=1, groups=nin)

    def forward(self, x):
        out = self.depthwise(x)
        return out


class separable_conv_3x3(nn.Module):
    def __init__(self, nin, nout, stride):
        # input node 일때, stride = 1; => size 유지
        # input node 아닐 대, stride = 2; =>  (x-1)/2 + 1
        super(separable_conv_3x3, self).__init__()
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=3,
                                   stride=stride)  # default: stride=1, padding=0, dilation=1, groups=1, bias=True

    def forward(self, x):
        out = self.pointwise(x)
        return out


class maxpool2d_3x3(nn.Module):
    def __init__(self, nin, nout, stride):
        super(maxpool2d_3x3, self).__init__()
        self.conv = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        out = self.conv(x)
        return out


class avgpool2d_3x3(nn.Module):
    def __init__(self, nin, nout, stride):
        super(avgpool2d_3x3, self).__init__()
        self.conv = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        out = self.conv(x)
        return out


class identity(nn.Module):
    def __init__(self, nin, nout, stride):
        # input node 일때, stride = 1; => size 유지
        # input node 아닐 대, stride = 2; =>  (x-1)/2 + 1
        super(identity, self).__init__()

    def forward(self, x):
        out = x
        return out


class Triplet_unit(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1):
        super(Triplet_unit, self).__init__()
        self.relu = nn.ReLU()
        self.conv = depthwise_separable_conv_3x3(inplanes, outplanes, stride)
        self.bn = nn.BatchNorm2d(outplanes)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv(out)
        out = self.bn(out)
        return out


def operation_dictionary():
    temp_dict = {}
    temp_dict[0] = conv2d_3x3  # nin, nout, kernel_size=3, stride=stride, padding=
    temp_dict[1] = depthwise_conv_3x3  # nin, nout ,  stride
    temp_dict[2] = separable_conv_3x3
    temp_dict[3] = depthwise_separable_conv_3x3
    temp_dict[4] = maxpool2d_3x3  # parameter Kernel_size , stride, padding
    temp_dict[5] = avgpool2d_3x3
    temp_dict[6] = identity

    return temp_dict