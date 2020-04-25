import os
import sys

sys.path.insert(0,'../')

import torch
import torch.nn as nn
import numpy as np

from utils_kyy import models

import torch
import torch.nn as nn
import networkx as nx
import math

from utils_kyy.utils_graph import load_graph, get_graph_info

import sys
import random

sys.path.insert(0,'../')
from utils_kyy.utils_graph import make_random_graph
from utils_kyy.utils_graph import *

import networkx as nx
import numpy as np
from easydict import EasyDict


def gmat2graph(g_mat):
    G = nx.Graph()
    nsize = g_mat.shape[0]
    G.add_nodes_from(range(nsize))
    for i_idx in range(nsize-1):
        for j_idx in range(i_idx + 1, nsize):
            if g_mat[i_idx][j_idx] != 7:
                G.add_edge(i_idx, j_idx)

    return G

def gmat2ind(g_mat):
    nsize = g_mat.shape[0]
    g_encode = []

    for idx, col_idx in enumerate(range(1, nsize)):
        row_idx = idx + 1
        g_encode.extend(g_mat[:row_idx, col_idx].tolist())

    return g_encode


def ind2gmat(g_encode, nsize):

    decode_mat = np.ones((nsize, nsize))

    # to initlize 7 (disconnected)
    decode_mat = decode_mat * 7

    row_idx = 0
    col_idx = 1
    for idx, v in enumerate(g_encode):
        decode_mat[row_idx][col_idx] = v
        if row_idx + 1 == col_idx:
            col_idx += 1
            row_idx = 0
        else:
            row_idx += 1

    return decode_mat

def make_random_graph_ex(num_graph):
    # 해당 stage_pool_path에 stage가 꽉 차있는지 확인

    #Nodes = random.randint(20, 40)  # => [Nodes 값 수정 시 주의] 아래 K값은 Nodes보다 작아야함.
    Nodes = num_graph
    graph_model = 'WS'
    K = random.randint(4, Nodes - 10)  # [min, max] // WS에서는 K nearest. 따라서, 4 ~ 30 random 선택 하도록
    P = round(random.uniform(0.25, 0.75), 2)  # 소수 둘째자리까지만 나오도록 반올림 => 결과 e.g. 0.75

    args = EasyDict({'graph_model': graph_model, 'K': K, 'P': P})

    P_str = str(P)[0] + str(P)[2:]  # 0.75 => 075

    graph = build_graph(Nodes, args)

    return graph


def get_graph_info(graph):
    input_nodes = []
    output_nodes = []
    Nodes = []
    for node in range(graph.number_of_nodes()):
        # node i 에 대해
        tmp = list(graph.neighbors(node))
        tmp.sort()  # 오름차순 정렬

        # node type 정의
        type = -1  # input node도, output node도 아닌. 그래프의 중간에 매개자처럼 있는 중간 node.
        if node < tmp[0]:
            input_nodes.append(node)
            type = 0  # id 가장 작은 노드보다 작으면, 이건 외부에서 input을 받는 노드. 즉 input node.
        if node > tmp[-1]:
            output_nodes.append(node)
            type = 1  # id 가장 큰 노드보다 크면, 이건 외부로 output 내보내는 노드. 즉 output node.

        # dag로 변환 (자신의 id보다 작은 노드들과의 연결만 남기기)
        # [type] 0: input node, 1: output node, -1: input도 output도 아닌, 그래프 중간에 매개자처럼 있는 중간 node
        Nodes.append(Node(node, [n for n in tmp if n < node], type))  # DAG(Directed Acyclic Graph)로 변환

    ## Checking Connection
    connection_flag = False
    here = len(Nodes) - 1
    visited = [False] * (len(Nodes) - 1)
    q = []
    neigbors = Nodes[here][1]

    q.append(neigbors[0])
    visited[neigbors[0]] = True

    while q:
        #      print("here",here)
        neigbors = Nodes[here][1]
        #       print("neigbors",neigbors)

        for n in neigbors:
            if visited[n] == False:
                visited[n] = True
                q.append(n)

        here = q[0]
        q = q[1:]

        if visited[0] == True:
            #            print("Connected")
            connection_flag = True
            break

    if connection_flag == False:
        ## 강제로 연결
        Nodes[-1][1].insert(0, 0)
        Nodes[-1][1].sort()

    return Nodes, input_nodes, output_nodes


def get_g_matrix(Nodes):
    nsize = len(Nodes)
    adj_mat = np.array([[7] * nsize for i in range(nsize)])
    random.seed(1234)

    for n in Nodes:
        node_id = n[0]
        for n_input in n[1]:
            operation = random.randint(0, 6)
            adj_mat[n_input][node_id] = operation

    return adj_mat


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
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, stride=stride, padding=0, groups=1)  # default: stride=1, padding=0, dilation=1, groups=1, bias=True

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


## For Crossover and mutation


def cxhr(ind1, ind2):

    ## individual 받고 -> 3등분 -> 3등분 안에서만 시행
    new_ind1 = []
    new_ind2 = []


    one_len = ind1/3

    for i in range(3):
        temp_ind1 = ind1[one_len*i : one_len* (i+1)]
        temp_ind2 = ind2[one_len*i : one_len* (i+1)]

        x1, x2 = cxOnePoint(temp_ind1,temp_ind2)

        new_ind1.extend(x1)
        new_ind2.extend(x2)

    return  new_ind1, new_ind2


def cxOnePoint(ind1, ind2):
    """Executes a one point crossover on the input :term:`sequence` individuals.
    The two individuals are modified in place. The resulting individuals will
    respectively have the length of the other.
    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.
    This function uses the :func:`~random.randint` function from the
    python base :mod:`random` module.
    """
    size = min(len(ind1), len(ind2))
    cxpoint = random.randint(1, size - 1)
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]

    return ind1, ind2


def cxgray(ind1, ind2, num_graph):
    gray_len = len(str(grayCode(num_graph)))

    ## gray_len : 한 개 gray code의 길이, 전체 //3
    new_ind1 = []
    new_ind2 = []

    for i in range(3):
        temp_ind1 = ind1[gray_len * i:gray_len * (i + 1)]
        temp_ind2 = ind2[gray_len * i:gray_len * (i + 1)]
        if random.random() < 0.7:
            x1, x2 = cxOnePoint(temp_ind1, temp_ind2)
            new_ind1.extend(x1)
            new_ind2.extend(x2)
        else:
            new_ind1.extend(temp_ind1)
            new_ind2.extend(temp_ind2)

    return new_ind1, new_ind2

