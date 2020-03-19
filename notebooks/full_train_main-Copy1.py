import os
import sys
import os
import logging
from easydict import EasyDict
import numpy as np
import random
import time
import datetime
from deap import tools
from collections import OrderedDict
from pprint import pprint
import json
import torch

sys.path.insert(0, '../')
from utils_kyy.utils_graph import make_random_graph_v2
from utils_kyy.create_toolbox import create_toolbox_for_NSGA_RWNN, evaluate_v2_full_train

import argparse
import random


class full_train:
    def __init__(self, json_file):
        self.root = os.path.abspath(os.path.join(os.getcwd(), '..'))
        self.param_dir = os.path.join(self.root + '/parameters/', json_file)
        f = open(self.param_dir)
        params = json.load(f)
        pprint(params)
        self.name = params['NAME']

        ## toolbox params
        self.args_train = EasyDict(params['ARGS_TRAIN'])
        self.data_path = params['DATA_PATH']
        self.run_code = params['RUN_CODE']
        self.stage_pool_path = '../graph_pool' + '/' + self.run_code + '_' + 'experiment_1' + '/'
        self.stage_pool_path_list = []
        for i in range(1, 4):
            stage_pool_path_i = self.stage_pool_path + str(i) + '/'  # eg. [graph_pool/run_code_name/1/, ... ]
            self.stage_pool_path_list.append(stage_pool_path_i)

        self.log_path = '../logs/' + self.run_code + '_' + self.name + '/'
        # self.log_file_name : Initialize 부터 GA 진행상황 등 코드 전체에 대한 logging
        self.log_file_name = self.log_path + 'logging.log'
        # self.train_log_file_name : fitness (= flops, val_accuracy). 즉 GA history 를 저장 후, 나중에 사용하기 위한 logging.
        self.train_log_file_name = self.log_path + 'train_logging.log'

        if not os.path.exists(self.stage_pool_path):
            os.makedirs(self.stage_pool_path)
            for i in range(3):
                os.makedirs(self.stage_pool_path_list[i])

        if not os.path.isdir(self.log_path):
            os.makedirs(self.log_path)

        logging.basicConfig(filename=self.log_file_name, level=logging.INFO)
        logging.info('[Start] Rwns_train class is initialized.')
        logging.info('Start to write log.')

        self.num_graph = params['NUM_GRAPH']

        ## Temperary
        self.models = [[1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1],
                       [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1],
                       [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1],
                       [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
                       [1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                       [1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
                       [1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0]]

        self.random_models = [[0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1],
                              [1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0],
                              [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
                              [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0],
                              [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
                              [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0]]

        # self.random_models =

        ## logs
        self.log = OrderedDict()
        self.log['hp'] = self.args_train
        self.train_log = OrderedDict()

        ## 기존 train_log 불러와서 이어서 train하기
        self.TRAIN_FROM_LOGS = params['TRAIN_FROM_LOGS']
        self.REAL_TRAIN_LOG_PATH = params['REAL_TRAIN_LOG_PATH']

    def train(self, mode=0):

        # mode = 0 : GA, 1: random
        inds = self.models
        if mode == 1: inds = self.random_models

        ###################################
        # 1. Initialize the population.  (toolbox.population은 creator.Individual n개를 담은 list를 반환. (=> population)
        ###################################
        now = datetime.datetime.now()
        now_str = now.strftime('%Y-%m-%d %H:%M:%S')

        ## ind는 gray code로 되어 있다
        ### 모델 가져오기 --> evaluate 함수
        ##  모델 트레이닝

        ### inds 만들기
        # 떨궈진 파일 파싱하기?
        inds = self.models

        if self.TRAIN_FROM_LOGS == False:

            for idx, ind in enumerate(inds):
                logging.info("Start Model Training " + now_str + " model " + str(idx))
                init_start_time = time.time()
                model_dict = {}
                fitness, epoch = evaluate_v2_full_train(ind, args_train=self.args_train,
                                                        stage_pool_path_list=self.stage_pool_path_list,
                                                        data_path=self.data_path,
                                                        log_file_name=self.log_file_name)

                now_str = now.strftime('%Y-%m-%d %H:%M:%S')
                logging.info("Initialion is finished at " + now_str)

                end_time = time.time()
                model_dict['model_id'] = ind
                model_dict['fitness'] = fitness
                model_dict['time'] = end_time - init_start_time
                model_dict['epoch'] = epoch

                ## log 기록 - initialize (= 0th generation)
                self.train_log[str(idx)] = model_dict
                self.save_log()



        # train_log 읽어와서 중간부터 이어서 train 하는 경우
        # [Reference] Seeding a population => https://deap.readthedocs.io/en/master/tutorials/basic/part1.html
        elif self.TRAIN_FROM_LOGS == True:
            print("################# [KYY-check] Read train_log from the middle #################")
            logging.info("################# [KYY-check] Read train_log from the middle #################")

            # train_log 읽어오기
            with open(self.REAL_TRAIN_LOG_PATH) as train_log_json_file:
                data = json.load(train_log_json_file)  # hp(=hyperparameter), train_log 있음

            train_log_past = data['train_log']
            niter = len(train_log_past)  # 기록 상 총 init 횟수

            start_gen = niter  # niter = 11 이면, log 상에 0 ~ 10번까지 기록되어있는 것.

            # self.train_log 에 읽어온 로그 넣어놓기 (OrderedDict())
            for i in range(niter):
                self.train_log[str(i)] = train_log_past[str(i)]

            for idx in range(niter, len(inds)):
                ind = inds[idx]

                logging.info("Start Model Training " + now_str + " model " + idx)
                init_start_time = time.time()
                model_dict = {}
                fitness, epoch = evaluate_v2_full_train(ind, args_train=self.args_train,
                                                        stage_pool_path_list=self.stage_pool_path_list,
                                                        data_path=self.data_path,
                                                        log_file_name=self.log_file_name)

                now_str = now.strftime('%Y-%m-%d %H:%M:%S')
                logging.info("Initialion is finished at " + now_str)

                end_time = time.time()
                model_dict['model_id'] = ind
                model_dict['fitness'] = fitness
                model_dict['time'] = end_time - init_start_time
                model_dict['epoch'] = epoch

                ## log 기록 - initialize (= 0th generation)
                self.train_log[str(idx)] = model_dict
                self.save_log()

    ## Save Log
    def save_log(self):
        ## 필요한 log 추후 정리하여 추가
        self.log['train_log'] = self.train_log

        with open(self.train_log_file_name, 'w', encoding='utf-8') as make_file:
            json.dump(self.log, make_file, ensure_ascii=False, indent='\t')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, help='Parameter Json file')

    args = parser.parse_args()

    trainer = full_train(json_file=args.params)

    trainer.train()
    trainer.save_log()
