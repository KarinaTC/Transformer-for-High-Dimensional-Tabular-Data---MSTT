import sys
from pathlib import Path
# Main root of repo MSTT
PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))

## Importar librerías 
import numpy as np
import torch
import time
import torch.nn.functional as F
import random
import json
import torch.optim
import torch.nn as nn

from tqdm.auto import tqdm # Loading bar    
from torch.utils.data import  DataLoader
import matplotlib.pyplot as plt

from src.model import MSTT
import src.utils as utils
from src import preprocess_data_tab
import src.training as training
import src.testing as testing
from pathlib import Path
from itertools import product

import pandas as pd
import copy
import os
import gc
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score

## --------------------- CONFIGURATIONS ---------------------
SAVE_DICTIONARY = False
NUM_WORKERS = min(4, os.cpu_count())
print("num_workers:", NUM_WORKERS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed),
    torch.manual_seed(seed)     # For CPU
    # For GPU (CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # si usas más de una GPU

    #  For dataloader
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


MODEL_HYPERPARAMS = {
    "pos_norm": False,
    "bias_st_cat": True,
    "device" :  DEVICE, 
}

TRAINING_ARGS =  {
    "patience_early_stop":  10,
    "learning_rate" : 5e-4,
    "weight_decay" : 1e-4,
    "batch_size" : 64,
    "epochs" :  200
}
