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
from utils_kyy.utils_hr import make_random_graph_ex
from utils_kyy.create_toolbox_hr import create_toolbox_for_NSGA_RWNN_hr, evaluate_hr_full_train

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

        self.log_path = '../logs/' + self.run_code + '_' + self.name + '/'
        # self.log_file_name : Initialize 부터 GA 진행상황 등 코드 전체에 대한 logging
        self.log_file_name = self.log_path + 'logging.log'
        # self.train_log_file_name : fitness (= flops, val_accuracy). 즉 GA history 를 저장 후, 나중에 사용하기 위한 logging.
        self.train_log_file_name = self.log_path + 'train_logging.log'
        self.checkpoint_path = '../checkpoints/' + self.name
        

        if not os.path.isdir(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        if not os.path.isdir(self.log_path):
            os.makedirs(self.log_path)

        logging.basicConfig(filename=self.log_file_name, level=logging.INFO)
        logging.info('[Start] Rwns_train class is initialized.')
        logging.info('Start to write log.')


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

    def create_toolbox(self):
        now = datetime.datetime.now()
        now_str = now.strftime('%Y-%m-%d %H:%M:%S')
        logging.info("[GA] Creating Tool Box " + now_str)
        return create_toolbox_for_NSGA_RWNN_hr(self.args_train, self.data_path, self.log_file_name)

    def train(self):
        ###################################
        # 1. Initialize the population.  (toolbox.population은 creator.Individual n개를 담은 list를 반환. (=> population)
        ###################################
        now = datetime.datetime.now()
        now_str = now.strftime('%Y-%m-%d %H:%M:%S')
        print("[GA] Initialion starts ...")
        logging.info("[GA] Initialion starts at " + now_str)
        init_start_time = time.time()

        pop = self.toolbox.population(n=self.pop_size)

        # fitness values (= accuracy, flops) 모음
        GA_history_list = []  # (choromosome, accuracy, flops) 이렇게 담아놓기
        # e.g.  [ [[1,3,4], 40, 1000], [[2,6,10], 30%, 2000], ...  ]

        ###################################
        # 2. Evaluate the population (with an invalid fitness)
        ###################################
        invalid_ind = [ind for ind in pop] ## 이 부분은 왜한건지 iterator를 list로 만들기 위해?
        for idx, ind in enumerate(invalid_ind):
            print("processing chromosome",idx)
            fitness, ind_model = evaluate_hr_full_train(ind, args_train=self.args_train,
                                             data_path=self.data_path, log_file_name=self.log_file_name)
            ind.fitness.values = fitness
            GA_history_list.append([ind, fitness])

        ## log 기록 - initialize (= 0th generation)
        self.train_log[0] = GA_history_list

        self.save_log()

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

        ###################################
        # 3. Begin GA
        ###################################
        # Begin the generational process
        for gen in range(1, self.ngen):
            ##### 3.1. log 기록
            now = datetime.datetime.now()
            now_str = now.strftime('%Y-%m-%d %H:%M:%S')
            print("#####", gen, "th generation starts at", now_str)
            logging.info("#####" + str(gen) + "th generation starts at" + now_str)

            start_gen = time.time()

            ##### 3.2. Offspring pool 생성 후, crossover(=mate) & mutation
            # Vary the population
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            # ::2, 1::2 즉, 짝수번째 크로모좀과 홀수번쨰 크로모좀들 차례로 선택하면서 cx, mut 적용
            # e.g. 0번, 1번 ind를 cx, mut   // 2번, 3번 ind를 cx, mut // ...
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= self.cxpb:
                    self.toolbox.mate(ind1, ind2)

                self.toolbox.mutate(ind1, indpb=self.mutpb)
                self.toolbox.mutate(ind2, indpb=self.mutpb)
                del ind1.fitness.values, ind2.fitness.values


            ##### 3.3. Evaluation
            # Evaluate the individuals with an invalid fitness
            print("\t Evaluation...")
            start_time = time.time()
            # fitness values (= accuracy, flops) 모음
            GA_history_list = []

            invalid_ind = [ind for ind in offspring]

            for idx, ind in enumerate(invalid_ind):
                print("processing chromosome",idx)
                fitness, ind_model = evaluate_hr_full_train(ind, args_train=self.args_train,
                                                 data_path=self.data_path,
                                                            log_file_name=self.log_file_name, generation=gen ,idx=idx,save_model_path = self.checkpoint_path)
                # <= evaluate() returns  (-prec, flops), NN_model

                ind.fitness.values = fitness
                GA_history_list.append([ind, fitness])

            ## log 기록
            self.train_log[gen] = GA_history_list

            self.save_log()

            eval_time_for_one_generation = time.time() - start_time
            print("\t Evaluation ends (Time : %.3f)" % eval_time_for_one_generation)

            ##### Select the next generation population
            pop = self.toolbox.select(pop + offspring, self.pop_size)

            gen_time = time.time() - start_gen
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
    trainer.create_toolbox()
    trainer.train()
    trainer.save_log()
