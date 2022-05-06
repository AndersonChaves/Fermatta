# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# tf.enable_eager_execution()

import tensorflow as tf
import pandas as pd
from djensemble import DJEnsemble
from datetime import datetime
from online.configuration_manager import ConfigurationManager
import time

def print_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    return time.time()

def run_experiment(time_weight):
    start = print_time()
    djensemble = DJEnsemble(ConfigurationManager("Q1/query1.config"), time_weight)
    result, result_by_tile = djensemble.ensemble()
    print("DJEnsemble:", result, result_by_tile )
    end = print_time()
    print("Total time: ", end - start, " seconds")

if __name__ == '__main__':
    for time_weight in [0.0]:
        run_experiment(time_weight)
