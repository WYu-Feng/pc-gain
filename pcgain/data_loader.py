import numpy as np
from utils import binary_sampler
import random
import copy

def data_loader (data_name, miss_rate):
    file_name = 'datasets/' + data_name +'.csv'
    complete_data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
    no, dim = complete_data_x.shape
    np.random.shuffle(complete_data_x)
    
    #Limit the amount of data
    if no > 10000:
        complete_data_x = complete_data_x[0:10000,0:dim-1]
        no = 10000
        dim = dim -1 
    else:
        complete_data_x = complete_data_x[0:no,0:dim-1]
        dim = dim -1
        
    data_m = binary_sampler(1-miss_rate, no, dim)
    incomplete_data_x = complete_data_x.copy()
    incomplete_data_x[data_m == 0] = np.nan
    return complete_data_x, incomplete_data_x, data_m
