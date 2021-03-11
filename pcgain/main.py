from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

from data_loader import data_loader
from pc_gain import PC_GAIN
from utils import rmse_loss
import os


def main (args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    data_name = args.data_name
    miss_rate = args.miss_rate
    
    gain_parameters = {'batch_size': args.batch_size,
        'hint_rate': args.hint_rate,
        'alpha': args.alpha,
        'beta': args.beta,
        'lambda_': args.lambda_,
        'k': args.k,
        'iterations': args.iterations,
        'cluster_species':args.cluster_species}
    
    # Load data and introduce missingness
    data_x, miss_data_x, data_M = data_loader(data_name, miss_rate)
    
    row , col = miss_data_x.shape
    five_len = row//5           
    ## 5-cross validations impute missing data
    for i in range(5):
        incomplete_data_x = np.vstack((miss_data_x[0:i*five_len:,] , miss_data_x[(i+1)*five_len:row:,]))

        complete_data_x = np.vstack((data_x[0:i*five_len:,] , data_x[(i+1)*five_len:row:,]))
        
        data_m = np.vstack((data_M[0:i*five_len:,] , data_M[(i+1)*five_len:row:,]))
        imputed_data = PC_GAIN(incomplete_data_x , gain_parameters , data_m)
        
        rmse = str(np.round(rmse_loss (complete_data_x, imputed_data, data_m), 4))
        print('RMSE Performance: ',rmse)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        choices=['credit', 'news', 'letter'],
        default = 'letter',
        type=str)
    parser.add_argument(
        '--miss_rate',
        help='missing data probability',
        default = 0.5,
        type=float)
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch',
        default=64,
        type=int)
    parser.add_argument(
        '--hint_rate',
        help='hint probability',
        default=0.9,
        type=float)
    parser.add_argument(
        '--alpha',
        help='hyperparameter',
        default=100,
        type=float)
    parser.add_argument(
        '--beta',
        help='hyperparameter',
        default=20,
        type=float)        
    parser.add_argument(
        '--lambda_',
        help='hyperparameter',
        default=0.4,
        type=float)
    parser.add_argument(
        '--k',
        help='hyperparameter',
        default=4,
        type=float)         
    parser.add_argument(
        '--iterations',
        help='number of training interations',
        default=10000,
        type=int)
    parser.add_argument(
        '--cluster_species',
        choices=['KM' , 'SC' , 'AC' , 'KMPP'],
        help='cluster species',
        default = 'KM',
        type=str)                   
    args = parser.parse_args()
    main(args)
                               
            
            
            
            