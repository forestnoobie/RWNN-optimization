from __future__ import division
import random
import warnings

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

from itertools import repeat

def check_Upper(gy, num_graph):

    tmp = ''
    for i in gy:
        tmp += str(i)

    gray_len = len(str(grayCode(num_graph)))

    decimal = graydecode(int(tmp))
    if decimal > num_graph - 1:
        return num2gray(num_graph-1,gray_len)
    else :
        return gy

def grayCode(n):
    # Decimal to binary graycode
    # Right Shift the number
    # by 1 taking xor with
    # original number
    grayval = n ^ (n >> 1)

    return int(bin(grayval)[2:])


def graydecode(binary):
    # binary -> decimal

    binary1 = binary
    decimal, i, n = 0, 0, 0
    while (binary != 0):
        dec = binary % 10
        decimal = decimal + dec * pow(2, i)
        binary = binary // 10
        i += 1

    # Taking xor until
    # n becomes zero
    inv = 0
    while (decimal):
        inv = inv ^ decimal;
        decimal = decimal >> 1;

    return inv


def num2gray(n, gray_len):
    gy = str(grayCode(n))

    if len(gy) < gray_len:
        gy = '0' * (gray_len - len(gy)) + gy

    return gy

def cxOnePoint_gray(ind1, ind2):
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

############################
# Crossover - for grayencoding
############################
def cxgray(ind1, ind2, cxpb, num_graph):
    gray_len = len(str(grayCode(num_graph)))
#     gray_len = 7
    # 21개 -> cx point index = 7, 14, -1 (마지막)  => ind1[:7] + ind2[7:], ind2[:7] + ind2[7:] 이런 식으로
    cx_point_list = [gray_len, gray_len*2]
    
    if random.random() < cxpb:
        cx_point = random.choice(cx_point_list)
        ind1[cx_point:], ind2[cx_point:] = ind2[cx_point:], ind1[cx_point:]

    return ind1, ind2


############################
# Mutate - for grayencoding
############################
# 크로모좀 내에서 스테이지마다, mutpb 의 확률로 random flip 적용
def mutgray(individual, mutpb, num_graph):
    gray_len = len(str(grayCode(num_graph)))
#     gray_len = 7
    
    size = len(individual)  # 인덱스 0 ~ (size-1)
    
    for i in range(3):
        if random.random() < mutpb:
            flip_idx = random.randint(gray_len * i, gray_len * (i + 1) - 1)  # 0:6, 7:13, 14:20
            # bit flip
            if individual[flip_idx] == 0:
                individual[flip_idx] = 1
            elif individual[flip_idx] == 1:
                individual[flip_idx] = 0

    return individual,