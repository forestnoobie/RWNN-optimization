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
from utils_kyy.create_toolbox import evaluate_v2_full_train

import argparse

# 추가 - full training
import pandas as pd
from utils_kyy.pareto_front import identify_pareto


class Full_train:
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
        self.stage_pool_path = '../graph_pool' + '/' + self.run_code + '_' + 'experiment_1' + '/'   #####
        self.stage_pool_path_list = []
        for i in range(1, 4):
            stage_pool_path_i = self.stage_pool_path + str(i) + '/'  # eg. [graph_pool/run_code_name/1/, ... ]
            self.stage_pool_path_list.append(stage_pool_path_i)
        
        self.log_path = '../logs/' + self.run_code + '_' + self.name + '/'
        # self.log_file_name : Initialize 부터 GA 진행상황 등 코드 전체에 대한 logging
        self.log_file_name = self.log_path + 'logging.log'
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
        
        
        ## logs
        self.log = OrderedDict()
        self.log['hp'] = self.args_train
        self.train_log = OrderedDict()
        
        
        # training log 불러올 디렉토리
        self.GA_data_path = params['GA_DATA_PATH']
        self.RS_data_path = params['RS_DATA_PATH']


    def train(self):
        ######################################################################
        # 1. Initialize
        ######################################################################
        now = datetime.datetime.now()
        now_str = now.strftime('%Y-%m-%d %H:%M:%S')
        print("Initialion starts ...")
        logging.info("Initialion starts at " + now_str)

        
        ######################################################################
        # 2. Full training - GA pareto frontier
        ######################################################################
        
        ###################################
        ##### 2.1. GA training.log 읽어오기
        ###################################
        with open(os.path.join(self.GA_data_path, "train_logging.log")) as json_file:
            data = json.load(json_file)

        train_log = data['train_log']
        niter = len(train_log)
        npop = len(train_log['0'])

        objs_fitness = []
        objs_chromo = []
        gen_num = []
        for i in range(niter):
            gen_num.extend([i for j in range(npop)])
            fitness_i = [train_log[str(i)][j][1] for j in range(npop)]  # [-val_acc, flops]
            chromo_i = [train_log[str(i)][j][0] for j in range(npop)]  # [-val_acc, flops]
            objs_fitness.append(fitness_i)
            objs_chromo.append(chromo_i)

        objs_fitness = np.array(objs_fitness)
        epoch = list(range(niter))

        objs_fitness[:,:,0]= -1*objs_fitness[:,:,0]  # -val_acc => +val_acc

        y1 = objs_fitness[:,:,0].reshape(-1).tolist()  # val_accuracy 는 - 붙어있는채로 사용 => minimize 하는 pareto frontier 찾는 함수 그대로 사용
        y2 = objs_fitness[:,:,1].reshape(-1).tolist()
        idxs = [i for i in range(len(y1))]
        pareto = [0 for i in range(len(y1))]
        
        df = pd.DataFrame({'gen':gen_num,'idx': idxs, 'acc':y1, 'flops': y2})
        
        ###################################
        ###### 2.2 pareto front 찾기
        ###################################
        data_30gen_score = df[['acc','flops']].values[:400, :]  ########################### ~ 20 gen ####################

        # 1) flops 에 - 붙이기 => score 로 만들기
        data_30gen_score[:, 1] = -data_30gen_score[:, 1]

        # 2) 파레토 프론티어 찾기
        pareto_30gen_idx = identify_pareto(data_30gen_score)
        pareto_front_30gen = data_30gen_score[pareto_30gen_idx]

        # 3) 파레토 프론티어에 있는 크로모좀 리스트 만들기
        pareto_chromos = []
        for idx in list(pareto_30gen_idx):
            i = int(idx / 20)   # e.g. 33 => 1 * 20 + 13 => 1 gen 의 14번째 => objs_chromo[1][13]  ## 각각 0번째 ~ 19번째 있음
            j = idx - i*20
            temp_chromo = objs_chromo[i][j]
            pareto_chromos.append( temp_chromo )
            
        ###################################
        ###### 2.3. 파레토 프론티어에 있는 크로모좀들 풀트레이닝 - train_logging.log 떨구기까지
        ###################################
        print("############## [GA] Number of Chromosomes on Pareto Frontier :", len(pareto_chromos)) 
        logging.info("############## [GA] Number of Chromosomes on Pareto Frontier : " + str(len(pareto_chromos)))     
        for idx, ind in enumerate(pareto_chromos):
            num = idx + 1
            print('\t', num, 'th Chromosome - evaluation...')
            logging.info(str(num) + 'th Chromosome - evaluation...')
            train_init_time = time.time()
            
            model_dict = {}
            fitness, epoch = evaluate_v2_full_train(ind, args_train=self.args_train,
                                                        stage_pool_path_list=self.stage_pool_path_list,
                                                        data_path=self.data_path,
                                                        channels=self.args_train.channels,
                                                        log_file_name=self.log_file_name)
            

            trained_end_time = time.time() - train_init_time
            
            model_dict['model_id'] = ind
            model_dict['fitness'] = fitness
            model_dict['time'] = trained_end_time
            model_dict['epoch'] = epoch

            ## log 기록 - initialize (= 0th generation)
            self.train_log[str(idx)] = model_dict
            self.save_log()
            print("\t trained_end_time: ", trained_end_time)
            logging.info('\t trained_end_time: %.3fs' % (trained_end_time))
            
        
        
        ######################################################################
        # 3. Full training - RS pareto frontier
        ######################################################################
        
        ###################################
        ##### 3.1. RS training.log 읽어오기
        ###################################
#         with open(os.path.join(self.RS_data_path, "train_logging.log")) as json_file:
#             data = json.load(json_file)

#         train_log = data['train_log']
#         niter = len(train_log)
#         npop = len(train_log['0'])

#         objs_fitness = []
#         objs_chromo = []
#         gen_num = []
#         for i in range(niter):
#             gen_num.extend([i for j in range(npop)])
#             fitness_i = [train_log[str(i)][j][1] for j in range(npop)]  # [-val_acc, flops]
#             chromo_i = [train_log[str(i)][j][0] for j in range(npop)]  # [-val_acc, flops]
#             objs_fitness.append(fitness_i)
#             objs_chromo.append(chromo_i)

#         objs_fitness = np.array(objs_fitness)
#         epoch = list(range(niter))

#         objs_fitness[:,:,0]= -1*objs_fitness[:,:,0]  # -val_acc => +val_acc

#         y1 = objs_fitness[:,:,0].reshape(-1).tolist()  # val_accuracy 는 - 붙어있는채로 사용 => minimize 하는 pareto frontier 찾는 함수 그대로 사용
#         y2 = objs_fitness[:,:,1].reshape(-1).tolist()
#         idxs = [i for i in range(len(y1))]
#         pareto = [0 for i in range(len(y1))]
        
#         df = pd.DataFrame({'gen':gen_num,'idx': idxs, 'acc':y1, 'flops': y2})
        
#         ###################################
#         ###### 3.2 pareto front 찾기
#         ###################################
#         data_30gen_score = df[['acc','flops']].values  # df 전부

#         # 1) flops 에 - 붙이기 => score 로 만들기
#         data_30gen_score[:, 1] = -data_30gen_score[:, 1]

#         # 2) 파레토 프론티어 찾기
#         pareto_30gen_idx = identify_pareto(data_30gen_score)
#         pareto_front_30gen = data_30gen_score[pareto_30gen_idx]

#         # 3) 파레토 프론티어에 있는 크로모좀 리스트 만들기
#         pareto_chromos = []
#         for idx in list(pareto_30gen_idx):
#             i = int(idx / 20)   # e.g. 33 => 1 * 20 + 13 => 1 gen 의 14번째 => objs_chromo[1][13]  ## 각각 0번째 ~ 19번째 있음
#             j = idx - i*20
#             temp_chromo = objs_chromo[i][j]
#             pareto_chromos.append( temp_chromo )
            
#         ###################################
#         ###### 3.3. 파레토 프론티어에 있는 크로모좀들 풀트레이닝 한 뒤에, 데이터 프레임 저장해놓기
#         ###################################
#         print("############## [RS] Number of Chromosomes on Pareto Frontier:", len(pareto_chromos))
#         logging.info("############## [RS] Number of Chromosomes on Pareto Frontier : " + str(len(pareto_chromos)))             
#         acc_full_pt = []
#         trained_epoch = []
#         trained_time = []
#         num = 1
#         for ind in pareto_chromos:
#             print('\t', num, 'th Chromosome - evaluation...')
#             logging.info(str(num) + 'th Chromosome - evaluation...')
#             train_init_time = time.time()

#             fitness, epoch = evaluate_v2_full_train_kyy(ind, args_train=self.args_train,
#                                                         stage_pool_path_list=self.stage_pool_path_list,
#                                                         epoch_weight = self.epoch_weight,
#                                                         data_path=self.data_path,
#                                                         log_file_name=self.log_file_name)
            
#             acc_full = -fitness[0]
#             acc_full_pt.append( acc_full )
#             trained_epoch.append( epoch )
            
#             trained_end_time = time.time() - train_init_time
#             trained_time.append( trained_end_time )
            
#             print("\t trained_end_time: ", trained_end_time)
#             logging.info('\t trained_end_time: %.3fs' % (trained_end_time))            
            
#             num += 1

#         # pareto frontier를 찾은 뒤, full training 까지 해서 완성 할 데이터 프레임
#         gen_pt = []
#         idx_pt = list(pareto_30gen_idx)
#         flops_pt = []
#         acc_5epoch_pt = []

#         for idx in list(pareto_30gen_idx):
#             df_idx = df.iloc[[idx]].values[0]  # acc, floops, gen, idx

#             gen_pt.append(df_idx[2])
#             idx_pt.append(df_idx[3])
#             flops_pt.append(df_idx[1])
#             acc_5epoch_pt.append(df_idx[0])

#         ## 데이터 프레임 생성
#         df_pareto_RS = pd.DataFrame({'gen': gen_pt, 'idx': idx_pt, 'flops': flops_pt, 'acc_5epoch':acc_5epoch_pt, 'acc_full':acc_full_pt,
#                                     'trained_epoch': trained_epoch, 'trained_time': trained_time})

#         ## 데이터 프레임 저장
#         df_pareto_RS.to_pickle(self.RS_save_path)  # [참고] temp = pd.read_pickle(load_path)        
        
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
    
    trainer = Full_train(json_file=args.params)

    trainer.train()
    
    print("Finished.")
