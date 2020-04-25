from deap import base, creator
from deap import tools


import random
from itertools import repeat
from collections import Sequence

# For evaluate function --------------------------
import glob
from easydict import EasyDict

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn    # for hardware tunning (cudnn.benchmark = True)

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from thop import profile
from thop import clever_format

import logging

# Gray code package
from utils_kyy.utils_graycode_v2 import *

# custom package in utils_kyy
from utils_kyy.utils_graph import load_graph
from utils_kyy.utils_hr import *
from utils_kyy.models_hr import RWNN
from utils_kyy.train_validate import train, validate, train_AMP
from utils_kyy.lr_scheduler import LRScheduler
from torchsummary import summary
# -------------------------------------------------

#from apex import amp



import time

################ hr
## For MNIST
class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


def init_graph(nsize):
    # nsize : number of nodes

    individuals = []

    for i in range(3):  # For s stages
        grph_ex = make_random_graph_ex(nsize)  # Make random graph using WS

        ## G matrix 구하기
        nds, inds, onds = get_graph_info(grph_ex)
        g_mat = get_g_matrix(nds)

        individual = gmat2ind(g_mat)
        individuals.extend(individual)

    return individuals

def create_toolbox_for_NSGA_RWNN_hr(args_train, data_path=None, log_file_name=None):
    # => Min ( -val_accuracy(top_1),  flops )
    creator.create('FitnessMin', base.Fitness, weights=(-1.0, -1.0))  # name, base (class), attribute //
    creator.create('Individual', list, fitness=creator.FitnessMin)  # creator.FitnessMaxMin attribute로 가짐

    #####################################
    # Initialize the toolbox
    #####################################
    toolbox = base.Toolbox()

    if args_train.hr:
        ## Nsize -> individual size
        nsize = args_train.nsize
        BOUND_LOW = 0
        BOUND_UP = 7  # Operation 종류
        # toolbox.register('attr_int', random.randint, BOUND_LOW, BOUND_UP)

        toolbox.register('init_graph',init_graph, nsize)
        # input : nsize
        # output : nsize -> Graph -> Gmat -> indivdual로 연계되게

    else :
        print("change hr to True")

    ## HR 일 때는 WS로 initiat 하고 싶은데 이 부분 좀 생각이 필요할듯
    toolbox.register('individual', tools.initRepeat,
                     creator.Individual, toolbox.init_graph, n=1)

    toolbox.register('population', tools.initRepeat,
                     list, toolbox.individual)  # n은 생략함. toolbox.population 함수를 뒤에서 실행할 때 넣어줌.

    # crossover

    if args_train.hr:
        toolbox.register('mate', cxhr)
    else:
        toolbox.register('mate', tools.cxTwoPoint)  # crossover

    # mutation
    toolbox.register('mutate', mutUniformInt_custom, low=BOUND_LOW, up=BOUND_UP)

    # selection
    # => return A list of selected individuals.
    toolbox.register('select', tools.selNSGA2,
                     nd='standard')  # selection.  // k – The number of individuals to select. k는 함수 쓸 때 받아야함

    #########################
    # Seeding a population - train_log 읽어와서 해당 log의 마지막 population으로 init 후 이어서 train 시작
    #########################
    # [Reference] https://deap.readthedocs.io/en/master/tutorials/basic/part1.html
    def LoadIndividual(icls, content):
        return icls(content)

    def LoadPopulation(pcls, ind_init, last_population):  # list of [chromosome, [-val_accuracy, flops]]
        return pcls(ind_init(last_population[i][0]) for i in range(len(last_population)))

    toolbox.register("individual_load", LoadIndividual, creator.Individual)

    toolbox.register("population_load", LoadPopulation, list, toolbox.individual_load)

    return toolbox


###

# stage_pool_path_list ~ graphs 만들어주는 과정 -> indivudal에서 바로 생성하는 걸로!


############################
# Mutate
############################
# 기존 mutUniformInt에 xrange() 함수가 사용됐어서, range로 수정함.
# indpb: toolbox.mutate() 함수로 사용할 때, MUTPB로 넣어줌. individual의 각 원소에 mutation 적용될 확률.
# indpb – Independent probability for each attribute to be mutated.
def mutUniformInt_custom(individual, low, up, indpb):
    """Mutate an individual by replacing attributes, with probability *indpb*,
    by a integer uniformly drawn between *low* and *up* inclusively.
    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param low: The lower bound or a :term:`python:sequence` of
                of lower bounds of the range from wich to draw the new
                integer.
    :param up: The upper bound or a :term:`python:sequence` of
               of upper bounds of the range from wich to draw the new
               integer.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    """
    size = len(individual)
    if not isinstance(low, Sequence):
        low = repeat(low, size)
    elif len(low) < size:
        raise IndexError("low must be at least the size of individual: %d < %d" % (len(low), size))
    if not isinstance(up, Sequence):
        up = repeat(up, size)
    elif len(up) < size:
        raise IndexError("up must be at least the size of individual: %d < %d" % (len(up), size))

    for i, xl, xu in zip(range(size), low, up):
        if random.random() < indpb:
            individual[i] = random.randint(xl, xu)

    return individual,


############################
# Evaluate
############################
"""
# fitness function
    input: [0, 5, 10]   하나의 크로모좀.

    2) training (임시로 1 epoch. 실제 실험 시, RWNN과 같은 epoch 학습시키기)

    3) return flops, val_accuracy

"""


# stage_pool_path_list ~ graphs 만들어주는 과정 -> indivudal에서 바로 생성하는 걸로!


def evaluate_hr_full_train(individual, args_train, data_path, channels=109,
                           log_file_name=None):  # individual

    graphs = []
    gmats = []

    # Need to divdie ind in to 3
    one_len = len(individual)//3
    print("length of one individual",one_len)
    for i in range(3):
        ind = individual[i*one_len : (i+1)*one_len]
        gmat = ind2gmat(ind, args_train.nsize)
        gmats.append(gmat)
        graphs.append(gmat2graph(gmat))

    graphs = EasyDict({'stage_1': graphs[0],
                       'stage_2': graphs[1],
                       'stage_3': graphs[2]
                       })

    # 2) build RWNN
    channels = channels
    NN_model = RWNN(net_type='small', graphs=graphs, gmats=gmats, channels=channels, num_classes=args_train.num_classes,
                    input_channel=args_train.input_dim)
    NN_model.cuda()

    ###########################
    # Flops 계산 - [Debug] nn.DataParallele (for multi-gpu) 적용 전에 확인.
    ###########################
    input_flops = torch.randn(1, args_train.input_dim, 32, 32).cuda()
    flops, params = profile(NN_model, inputs=(input_flops,), verbose=False)

    ## Model summary
    # summary(NN_model, input_size=(1, 224, 224))

    # 3) Prepare for train### 일단 꺼보자!
    # NN_model = nn.DataParallel(NN_model)  # for multi-GPU
    # NN_model = nn.DataParallel(NN_model, device_ids=[0,1,2,3])
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(NN_model.parameters(), args_train.base_lr,
                                momentum=args_train.momentum,
                                weight_decay=args_train.weight_decay)

    start_epoch = 0
    best_prec1 = 0

    cudnn.benchmark = True  # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.

    ###########################
    # Dataset & Dataloader
    ###########################

    # 이미 다운 받아놨으니 download=False
    # 데이터가 없을 경우, 처음에는 download=True 로 설정해놓고 실행해주어야함

    if data_path is None:
        data_path = './data'

    if args_train.data == "CIFAR10":

        cutout_length = 16  # from nsga-net github

        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]  # from nsga-net github
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #  Cutout(cutout_length),  # from nsga-net github
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
            ])

        val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
            ])

        train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                                     download=True, transform=train_transform)

        val_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                                   download=True, transform=val_transform)

    else:
        raise Exception("Data Error, Only CIFAR10 allowed for the moment")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args_train.batch_size,
                                               shuffle=True, num_workers=args_train.workers)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args_train.batch_size,
                                             shuffle=False, num_workers=args_train.workers)

    ###########################
    # Train
    ###########################
    niters = len(train_loader)
    niters = 1

    lr_scheduler = LRScheduler(optimizer, niters,
                               args_train)  # (default) args.step = [30, 60, 90], args.decay_factor = 0.1, args.power = 2.0
    epoch_ = 0

    for epoch in range(start_epoch, args_train.epochs):
        # train for one epoch
        train(train_loader, NN_model, criterion, optimizer, lr_scheduler, epoch, args_train.print_freq, log_file_name)

        # evaluate on validation set
        prec1 = validate(val_loader, NN_model, criterion, epoch, log_file_name)

        # remember best prec@1 and save checkpoint
        #         is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        epoch_ = epoch

    return (-best_prec1, flops), epoch_  # Min (-val_accuracy, flops) 이므로 val_accuracy(top1)에 - 붙여서 return


# [Reference] https://github.com/ianwhale/nsga-net/blob/master/misc/utils.py
class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img