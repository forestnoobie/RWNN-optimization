########################################
# v2 보다 업데이트된 로직
########################################
# 1. 하나의 크로모좀에 대해 Multi-GPU 로 training 하지 않고, 크로모좀당 단일 gpu로 학습. (단, 4개 크로모좀 동시에 진행)
#   - Multi-GPU 처리하는 부분에서 에러가 나는건지.. 그냥 배치 사이즈가 커서 메모리 에러가 나는건지 불확실..
#      => 만약에 이렇게 처리했는데도 같은 에러 발생하면, '메모리 에러'일 확률이 훨씬 커지니 문제가 명확해짐. 배치를 줄이던지...
########################################
# 주의사항
# 1. pop_size = 20 고정해야함. => 크로모좀 4개씩 나눠서, 인덱스로 0 ~ 19 를 다룸.

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
from utils_kyy.create_toolbox_MultiProcessing import create_toolbox_for_NSGA_RWNN, evaluate_Multiprocess


import argparse


class rwns_train:
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
        self.stage_pool_path = '../graph_pool' + '/' + self.run_code + '_' + self.name + '/'
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
        
        self.toolbox = self.create_toolbox()

        
        ## GA params
        self.pop_size = params['POP_SIZE']
        self.ngen = params['NGEN']
        self.cxpb = params['CXPB']
        self.mutpb = params['MUTPB']

        ## logs
        self.log = OrderedDict()
        self.log['hp'] = self.args_train
        self.train_log = OrderedDict()
        
        ## 기존 train_log 불러와서 이어서 train하기
        self.TRAIN_FROM_LOGS = params['TRAIN_FROM_LOGS']
        self.REAL_TRAIN_LOG_PATH = params['REAL_TRAIN_LOG_PATH']
        
        self.RANDOM_SEARCH = params['RANDOM_SEARCH']
                        

    def create_toolbox(self):
        make_random_graph_v2(self.num_graph, self.stage_pool_path_list)

        return create_toolbox_for_NSGA_RWNN(self.num_graph, self.args_train, self.data_path, self.log_file_name)


    def train(self):
        ###################################
        # 1. Initialize the population.  (toolbox.population은 creator.Individual n개를 담은 list를 반환. (=> population)
        ###################################
        now = datetime.datetime.now()
        now_str = now.strftime('%Y-%m-%d %H:%M:%S')
        print("[GA] Initialion starts ...")
        logging.info("[GA] Initialion starts at " + now_str)
        init_start_time = time.time()

        # fitness values (= accuracy, flops) 모음
        GA_history_list = []    # (choromosome, accuracy, flops) 이렇게 담아놓기
                                # e.g.  [ [[1,3,4], 40, 1000], [[2,6,10], 30%, 2000], ...  ]
        start_gen = 1

        # train_log 읽어와서 중간부터 이어서 train 하지 않고, 처음부터 train 하는 경우
        if self.TRAIN_FROM_LOGS == False:
            pop = self.toolbox.population(n=self.pop_size)
            ###################################
            # 2. Evaluate the population (with an invalid fitness)
            ###################################
            invalid_ind = [ind for ind in pop]

            # pop size = 20 고정
            # 크로모좀 4개씩 gpu 하나씩 배정해서 training 하기
            for i in [0, 4, 8, 12, 16]:
                # i ~ i+1
                # i ~ (i+3) 크로모좀을 각각 training 하기 - 먼저 끝났으면 기다리도록.
                # 로그도 다르게 떨궈야하네.

                eval_time_for_4_chromo = time.time()

                # 각각 training
                # i ~ i+3
                fitness_dict = evaluate_Multiprocess(ind_list=[invalid_ind[i], invalid_ind[i+1], invalid_ind[i+2], invalid_ind[i+3] ],
                                                      args_train=self.args_train,
                                                     stage_pool_path_list=self.stage_pool_path_list,
                                                     data_path=self.data_path,
                                                     channels=self.args_train.channels,
                                                     log_file_name=self.log_file_name)

                # <= evaluate() returns  (-prec, flops), NN_model
                eval_time_for_4_chromo = time.time() - eval_time_for_4_chromo
                print('\t\t [eval_time_for_4_chromo: %.3fs]' % eval_time_for_4_chromo, i, '~', (i+3), 'chromo is evaluated.')
                logging.info('\t\t [eval_time_for_4_chromo: %.3fs] %03d ~ %03d th chromo is evaluated.' % (eval_time_for_4_chromo, i, i+3))

                for fit_idx, chromo_idx in zip([0, 1, 2, 3], [i, i+1, i+2, i+3]):
                    fitness_for_idx_chromo = fitness_dict[fit_idx]
                    invalid_ind[chromo_idx].fitness.values = fitness_for_idx_chromo
                    GA_history_list.append([invalid_ind[chromo_idx], fitness_for_idx_chromo])
                    

            ## log 기록 - initialize (= 0th generation)
            self.train_log[0] = GA_history_list

            self.save_log()

            # This is just to assign the crowding distance to the individuals
            # no actual selection is done
            pop = self.toolbox.select(pop, len(pop))

            
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
            npop = len(train_log_past['0'])
            
            start_gen = niter  # niter = 11 이면, log 상에 0 ~ 10번까지 기록되어있는 것.
            
            # self.train_log 에 읽어온 로그 넣어놓기 (OrderedDict())
            for i in range(niter):
                self.train_log[str(i)] = train_log_past[str(i)]

            self.save_log()
            
            # population 읽어오기
            # train_log 에서 last population 읽어오기
            last_population = train_log_past[str(int(niter)-1)]   # list of [chromosome, [-val_accuracy, flops]]

            # last population으로 population 만들기
            pop = self.toolbox.population_load(last_population)

            # fitness values 도 읽어오기
            for i in range(len(last_population)):    
                pop[i].fitness.values = last_population[i][1]  # [-val_accuracy, flops]
            
            # This is just to assign the crowding distance to the individuals
            # no actual selection is done
            pop = self.toolbox.select(pop, len(pop))
            
        now = datetime.datetime.now()
        now_str = now.strftime('%Y-%m-%d %H:%M:%S')
        print("Initialization is finished at", now_str)
        logging.info("Initialion is finished at " + now_str)

        init_time = time.time() - init_start_time
        logging.info("Initialization time = " + str(init_time) + "s")
        print()
        
        if self.RANDOM_SEARCH == False:
            ###################################
            # Begin GA
            ###################################
            # Begin the generational process
            for gen in range(start_gen, self.ngen+1):  # self.ngen 남은 횟수를 이어서 돌리기
                ##### 3.1. log 기록
                now = datetime.datetime.now()
                now_str = now.strftime('%Y-%m-%d %H:%M:%S')
                print("#####", gen, "th generation starts at", now_str)
                logging.info("#####" + str(gen) + "th generation starts at" + now_str)

                start_gen_time = time.time()

                ##### 3.2. Offspring pool 생성 후, crossover(=mate) & mutation
                # Vary the population
                offspring = tools.selTournamentDCD(pop, len(pop))
                offspring = [self.toolbox.clone(ind) for ind in offspring]

                # ::2, 1::2 즉, 짝수번째 크로모좀과 홀수번쨰 크로모좀들 차례로 선택하면서 cx, mut 적용
                # e.g. 0번, 1번 ind를 cx, mut   // 2번, 3번 ind를 cx, mut // ...
                for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                    # crossover (=mate)
                    self.toolbox.mate(ind1, ind2, self.cxpb)
                    
                    # mutation
                    self.toolbox.mutate(ind1, mutpb=self.mutpb)
                    self.toolbox.mutate(ind2, mutpb=self.mutpb)
                    del ind1.fitness.values, ind2.fitness.values

                ##### 3.3. Evaluation
                # Evaluate the individuals with an invalid fitness
                print("\t Evaluation...")
                start_time = time.time()

                # fitness values (= accuracy, flops) 모음
                GA_history_list = []

                invalid_ind = [ind for ind in offspring]
                # pop size = 20 고정
                # 크로모좀 4개씩 gpu 하나씩 배정해서 training 하기
                for i in [0, 4, 8, 12, 16]:
                    # i ~ i+1
                    # i ~ (i+3) 크로모좀을 각각 training 하기 - 먼저 끝났으면 기다리도록.
                    # 로그도 다르게 떨궈야하네.

                    eval_time_for_4_chromo = time.time()

                    # 각각 training
                    # i ~ i+3
                    fitness_dict = evaluate_Multiprocess(ind_list=[invalid_ind[i], invalid_ind[i+1], invalid_ind[i+2], invalid_ind[i+3] ],
                                                          args_train=self.args_train,
                                                         stage_pool_path_list=self.stage_pool_path_list,
                                                         data_path=self.data_path,
                                                         log_file_name=self.log_file_name)

                    # <= evaluate() returns  (-prec, flops), NN_model
                    eval_time_for_4_chromo = time.time() - eval_time_for_4_chromo
                    print('\t\t [eval_time_for_4_chromo: %.3fs]' % eval_time_for_4_chromo, i, '~', (i+3), 'chromo is evaluated.')
                    logging.info('\t\t [eval_time_for_4_chromo: %.3fs] %03d ~ %03d th chromo is evaluated.' % (eval_time_for_4_chromo, i, i+3))

                    for fit_idx, chromo_idx in zip([0, 1, 2, 3], [i, i+1, i+2, i+3]):
                        fitness_for_idx_chromo = fitness_dict[fit_idx]
                        invalid_ind[chromo_idx].fitness.values = fitness_for_idx_chromo
                        GA_history_list.append([invalid_ind[chromo_idx], fitness_for_idx_chromo])


                # 전체 크로모좀 evaluation 했으니 cuda cache 한번 clear 해주기
                torch.cuda.empty_cache()

                ## log 기록
                self.train_log[gen] = GA_history_list

                self.save_log()

                eval_time_for_one_generation = time.time() - start_time
                print("\t Evaluation ends (Time : %.3f)" % eval_time_for_one_generation)

                ##### Select the next generation population
                pop = self.toolbox.select(pop + offspring, self.pop_size)

                gen_time = time.time() - start_gen_time
                print('\t [gen_time: %.3fs]' % gen_time, gen, 'th generation is finished.')

                logging.info('\t Gen [%03d/%03d] -- evals: %03d, evals_time: %.4fs, gen_time: %.4fs' % (
                    gen, self.ngen, len(invalid_ind), eval_time_for_one_generation, gen_time))

        elif self.RANDOM_SEARCH == True:
            ###################################
            # Begin Random Search
            ###################################
            print("###################################")
            print("########## Random Search ##########")
            print("###################################")            
            # Begin the generational process
            for gen in range(start_gen, self.ngen+1):  # self.ngen 남은 횟수를 이어서 돌리기
                ##### 3.1. log 기록
                now = datetime.datetime.now()
                now_str = now.strftime('%Y-%m-%d %H:%M:%S')
                print("#####", gen, "th generation starts at", now_str)
                logging.info("#####" + str(gen) + "th generation starts at" + now_str)

                start_gen_time = time.time()
                
                # Random initialization
                offspring = self.toolbox.population(n=self.pop_size)

                ## Evaluation
                # Evaluate the individuals with an invalid fitness
                print("\t Evaluation...")
                start_time = time.time()

                # fitness values (= accuracy, flops) 모음
                GA_history_list = []

                invalid_ind = [ind for ind in offspring]
                # pop size = 20 고정
                # 크로모좀 4개씩 gpu 하나씩 배정해서 training 하기
                for i in [0, 4, 8, 12, 16]:
                    # i ~ i+1
                    # i ~ (i+3) 크로모좀을 각각 training 하기 - 먼저 끝났으면 기다리도록.
                    # 로그도 다르게 떨궈야하네.

                    eval_time_for_4_chromo = time.time()

                    # 각각 training
                    # i ~ i+3
                    fitness_dict = evaluate_Multiprocess(ind_list=[invalid_ind[i], invalid_ind[i+1], invalid_ind[i+2], invalid_ind[i+3] ],
                                                          args_train=self.args_train,
                                                         stage_pool_path_list=self.stage_pool_path_list,
                                                         data_path=self.data_path,
                                                         log_file_name=self.log_file_name)

                    # <= evaluate() returns  (-prec, flops), NN_model
                    eval_time_for_4_chromo = time.time() - eval_time_for_4_chromo
                    print('\t\t [eval_time_for_4_chromo: %.3fs]' % eval_time_for_4_chromo, i, '~', (i+3), 'chromo is evaluated.')
                    logging.info('\t\t [eval_time_for_4_chromo: %.3fs] %03d ~ %03d th chromo is evaluated.' % (eval_time_for_4_chromo, i, i+3))

                    for fit_idx, chromo_idx in zip([0, 1, 2, 3], [i, i+1, i+2, i+3]):
                        fitness_for_idx_chromo = fitness_dict[fit_idx]
                        invalid_ind[chromo_idx].fitness.values = fitness_for_idx_chromo
                        GA_history_list.append([invalid_ind[chromo_idx], fitness_for_idx_chromo])


                # 전체 크로모좀 evaluation 했으니 cuda cache 한번 clear 해주기
                torch.cuda.empty_cache()

                ## log 기록
                self.train_log[gen] = GA_history_list

                self.save_log()

                eval_time_for_one_generation = time.time() - start_time
                print("\t Evaluation ends (Time : %.3f)" % eval_time_for_one_generation)

                gen_time = time.time() - start_gen_time
                print('\t [gen_time: %.3fs]' % gen_time, gen, 'th generation is finished.')

                logging.info('\t Gen [%03d/%03d] -- evals: %03d, evals_time: %.4fs, gen_time: %.4fs' % (
                    gen, self.ngen, len(invalid_ind), eval_time_for_one_generation, gen_time))            
            

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
    
    trainer = rwns_train(json_file=args.params)

    trainer.train()
    trainer.save_log()
