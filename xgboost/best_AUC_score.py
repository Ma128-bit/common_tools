import numpy as np
import matplotlib.pyplot as plt
global n_splits = 5

def best_AUC_in_category(name, category):
    for i in range(n_splits):
        
