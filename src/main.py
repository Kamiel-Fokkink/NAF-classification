# -*- coding: utf-8 -*-

################## Importing libraries ####################
import os
print(os.getcwd())

import sys

sys.path.insert(0,"Preprocessing/")
sys.path.insert(0,"Modeling/")
sys.path.insert(0,"Collection/")
sys.path.insert(0,"Augmentation")


import Preprocessing
import Modeling
import Collection
import Augmentation

import argparse
import json
from time import time


## Use parser to get arguments from the command line in order to launch only selected steps
parser = argparse.ArgumentParser(description='NAF Classification Project',
epilog="Special thanks to all developers on the project: Kamiel Fokkink, Sai Abhishikth Ayyadevara, Gloria Tang, \
            Jiahe Zhu, Zidi Yang, Mojun Guo, Ran Ding and little XY")

parser.add_argument('--step', help='integer that tells which step to run', default=-1)
parser.add_argument('--step_from',
help='integer that tells from which step to run main. It then run all the steps from step_from',
default=0)
parser.add_argument('--step_list', help='list of integer that tells which steps to run', default=[])

parser.add_argument("--pathconf", help="path to conf file", default="../params/config/config.json")

args = parser.parse_args()
step = int(args.step)
step_from = int(args.step_from)
step_list = args.step_list
path_conf = args.pathconf

conf = json.load(open(path_conf, 'r'))



def main(step_list, NB_STEP_TOT, path_conf = '../config/config.json'):
    """
    Main function launching step by step the ML Pipeline
    Args:
        ##logger: Logger file
        step_list: List of steps to be executed
        NB_STEP_TOT: By default = number of total step to laucnh them all if no specific steps are given
        path_conf: path of the config file
    """
    START = time()

    #Computation of the steps to complete
    if len(step_list) > 0:
        step_list = eval(step_list)
    
    if (step == -1) and (len(step_list) == 0):
        step_list = list(range(step_from, NB_STEP_TOT + 1))
    
    print(step_list)
    #logger.debug('Steps to execute :' + ', '.join(map(str,step_list)))
    
    #Reading conf file
    conf = json.load(open(path_conf, 'r'))

    #Launch of each selected step
    if (step == 1) or (1 in step_list):
        #logger.debug("Beginning of step 1 - Loading and Preprocessing given dataset")

        # Preprocessing of the given datasets
        Preprocessing.main_preprocessing(conf)

        #logger.debug("End of step 1 ")

        print("Great job there we go!")

    if (step == 2) or (2 in step_list):
        #logger.debug("Beginning of step 2 - Collecting and Preprocessing external dataset")

        # Collecting external datasets

        # Preprocessing external datasets

        # Writing of external datasets

        #logger.debug("End of step 2")

    if (step == 3) or (3 in step_list):
        #logger.debug("Beginning of step 3 - Training model")

        #logger.debug("End of step 3")

    if (step == 4) or (4 in step_list):
        #logger.debug("Beginning of step 4 - Making predictions with model")

        #logger.debug("End of step 4")
 
if __name__ == '__main__':
    try:
        main(step_list, NB_STEP_TOT = 420, path_conf=path_conf)
    
    except Exception as e:
        #logger.error("Error during execution", exc_info=True)
        print(e)