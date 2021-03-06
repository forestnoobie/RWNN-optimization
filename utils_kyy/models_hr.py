import torch
import torch.nn as nn

import math

from utils_kyy.utils_graph import load_graph, get_graph_info
from utils_kyy.utils_hr import operation_dictionary, get_g_matrix

operation_dict = operation_dictionary()

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


class double_unit(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1):
        super(double_unit, self).__init__()
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(outplanes)

    def forward(self, x):
        out = self.relu(x)
        out = self.bn(out)
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


class Node_OP(nn.Module):
    def __init__(self, Node, inplanes, outplanes, gmat):
        super(Node_OP, self).__init__()
        self.is_input_node = Node.type == 0
        self.input_nums = len(Node.inputs)  # 해당 Node에 input으로 연결된 노드의 개수

        ## For hierachical representation
        self.id = Node.id
        self.oplist = nn.ModuleList()
        self.gmat = gmat
        self.Node = Node

        if self.input_nums >= 1:
            for in_idx in Node.inputs:
                operation_idx = gmat[in_idx][self.id]
                op = operation_dict[operation_idx]
                self.oplist.append(op(nin=outplanes, nout=outplanes, stride=1))  ## input node가 아닐 떄는 channel size 유지!!

        # input 개수가 1보다 크면, 여러 input을 합쳐야함.
        if self.input_nums >= 1:
            self.mean_weight = nn.Parameter(torch.ones(self.input_nums))  # type: torch.nn.parameter.Parameter
            self.sigmoid = nn.Sigmoid()

        if self.is_input_node:
            self.conv = Triplet_unit(inplanes, outplanes, stride=2)
        else:
            self.conv = double_unit(outplanes, outplanes, stride=1)

    # [참고] nn.Sigmoid()(torch.ones(1)) = 0.7311
    # seoungwonpark source 에서는 torch.zeros()로 들어감. => 0.5
    def forward(self, *input):
        if self.input_nums > 1 and self.is_input_node == False:
            out = self.sigmoid(self.mean_weight[0]) * self.oplist[0](input[0])  ## input node가 아니라고 생각해도 됨!
            for i in range(1, self.input_nums):
                out = out + self.sigmoid(self.mean_weight[i]) * self.oplist[i](input[i])

        elif self.input_nums == 1 and self.is_input_node == False:
            out = self.sigmoid(self.mean_weight[0]) * self.oplist[0](input[0])  ## input node가 아니라고 생각해도 됨!
            # print("Components \n self.mean_weight : {} Operation : {} Sigmoid : {} input : {} ".format(self.mean_weight[0], self.oplist[0],self.sigmoid(self.mean_weight[0]),input[0].size()))

        else:
            out = input[0]
            # print("Node ID : {} is a input Node \n Output : {}".format(self.id,out))

        out = self.conv(out)


        return out


class StageBlock(nn.Module):
    def __init__(self, graph, inplanes, outplanes, gmat):
        super(StageBlock, self).__init__()
        # graph를 input으로 받아서, Node_OP class. 즉, neural network graph로 전환함.
        self.nodes, self.input_nodes, self.output_nodes = get_graph_info(graph)
        self.nodeop = nn.ModuleList()  # Holds submodules in a list.
        self.gmat = gmat

        for node in self.nodes:
            # 각각의 node들을 Node_OP class로 만들어준 뒤, nn.ModuleList()인 self.nodeop에 append 해주기
            self.nodeop.append(Node_OP(node, inplanes, outplanes, self.gmat))

    def forward(self, x):
        results = {}
        # input
        for id in self.input_nodes:
            results[id] = self.nodeop[id](x)  # input x를 먼저 graph's input node에 각각 넣어줌.
        # graph 중간 계산
        for id, node in enumerate(self.nodes):
            # 각각의 노드 id에 대해
            if id not in self.input_nodes:
                # graph's input node가 아니라면, 그래프 내에서 해당 노드의 인풋들인 node.inputs의 output인 results[_id]
                #    => 그 결과를 results[id]에 저장.
                # self.nodeop[id]는 해당 id의 Node_OP. 즉, input들을 받아서 forward(모아서, conv 태우기)하는 것.
                # 따라서, input으로 넣을 때 unpack 함.
                # id 작은 노드부터 result를 차근차근 계산하면서, id를 올라감.
                # print("input size if id {}".format(id))
                results[id] = self.nodeop[id](*[results[_id] for _id in node.inputs])



        result = results[self.output_nodes[0]]
        # output
        # graph's output_nodes의 output 들을 평균내기
        for idx, id in enumerate(self.output_nodes):
            if idx > 0:
                result = result + results[id]
        result = result / len(self.output_nodes)
        return result


# Node_OP -> StageBlock class 정의해놓고,
# conv2, conv3, conv4에 각각 random graph 생성해서 모듈로 추가함
# e.g.
#  graphs = EasyDict({'stage_1': stage_1_graph,
#                     'stage_2': stage_2_graph,
#                     'stage_3': stage_3_graph
#  })   # stage_1_graph = 해당 graph 파일의 path
# channels = 109

class RWNN(nn.Module):
    def __init__(self, net_type, graphs, gmats, channels, num_classes=1000, input_channel=3):
        super(RWNN, self).__init__()

        self.input_channel = input_channel
        self.gmats = gmats
        # 논문에서도 conv1 쪽은 예외적으로 Conv-BN 이라고 언급함. (나머지에서는 Conv-ReLU-BN 을 conv 로 표기)
        #         self.conv1 = depthwise_separable_conv_3x3(input_channel, channels // 2, 2)    # nin, nout, stride
        self.conv1 = depthwise_separable_conv_3x3(input_channel, channels // 2, 1)  # nin, nout, stride
        self.bn1 = nn.BatchNorm2d(channels // 2)

        # 채널수 변화도, 논문에서처럼 conv2: C, conv3: 2C, conv4: 4C, conv5: 8C
        if net_type == 'small':
            #             self.conv2 = Triplet_unit(channels // 2, channels, 2)    # inplanes, outplanes, stride=2
            self.conv2 = Triplet_unit(channels // 2, channels, 1)  # inplanes, outplanes, stride=1

            #             self.conv3 = StageBlock(graphs.stage_1, channels // 2, channels)
            self.conv3 = StageBlock(graphs.stage_1, channels, channels, self.gmats[0])

            self.conv4 = StageBlock(graphs.stage_2, channels, channels * 2, self.gmats[1])

            self.conv5 = StageBlock(graphs.stage_3, channels * 2, channels * 4, self.gmats[2])

            self.relu = nn.ReLU()

            self.bn2 = nn.BatchNorm2d(channels * 4)
            self.avgpool = nn.AvgPool2d(4, stride=1)  # 마지막은 global average pooling
            self.fc = nn.Linear(channels * 4, num_classes)

        #         if net_type == 'small':
        #             self.conv2 = Triplet_unit(channels // 2, channels, 2)    # inplanes, outplanes, stride=2

        #             self.conv3 = StageBlock(graphs.stage_1, channels, channels)

        #             self.conv4 = StageBlock(graphs.stage_2, channels, channels *2)

        #             self.conv5 = StageBlock(graphs.stage_3, channels * 2, channels * 4)

        #             self.relu = nn.ReLU()
        # #             self.conv = nn.Conv2d(channels * 4, 1280, kernel_size=1)   # 마지막에 1x1 conv, 1280-d
        # #             self.bn2 = nn.BatchNorm2d(1280)

        #######################################
        # 원 코드에서 regular 부분 지움
        #######################################
        #         self.avgpool = nn.AvgPool2d(7, stride=1)  # 마지막은 global average pooling
        #         self.fc = nn.Linear(1280, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        # print("==========Data Size", x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        # print("==========Conv1 Size", x.size())  # => (-, 54, 16, 16)

        x = self.conv2(x)
        # print("==========Conv2  Size", x.size())  # => (-, 109, 8, 8)
        x = self.conv3(x)
        # print("==========Conv3  Size", x.size())  # => (-, 109, 8, 8)

        x = self.conv4(x)
        #         print("==========Conv4  Size", x.size())  # => (-, 218, 4, 4)

        x = self.conv5(x)
        # print("==========Conv5  Size", x.size())  # => (-, 436, 2, 2)

        #         x = self.relu(x)  # => (-, 436, 1, 1)
        #         x = self.conv(x)
        #         print("==========Conv6  Size", x.size())  # => (-, 1280, 1, 1)

        x = self.bn2(x)
        x = self.relu(x)

        # print("==========before avgpool  Size", x.size())   ## [수정] CIFAR-10 에서는 여기까지오면 (-, 1280, 7, 7) 이 아니라 (-, ,1280, 1, 1)
        x = self.avgpool(x)
        #         print("==========after avgpool  Size", x.size())
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

###################################################
# #print("==========Data Size", x.size())
# x = self.conv1(x)
# x = self.bn1(x)

# x = self.conv2(x)
# x = self.conv3(x)
# x = self.conv4(x)
# x = self.conv5(x)

# x = self.relu(x)
# x = self.conv(x)

# x = self.bn2(x)
# x = self.relu(x)
# x = self.avgpool(x)

# x = x.view(x.size(0), -1)
# x = self.fc(x)
