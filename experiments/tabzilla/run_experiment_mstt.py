import yaml
import os
from pathlib import Path
import sys

# Define project and configuration paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_DIR = Path(__file__).resolve().parent / "configs"

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
import experiments.tabzilla.preprocess_data_tab as preprocess_data_tab
import src.training as training
import src.testing as testing
from pathlib import Path
from itertools import product

import pandas as pd
import copy
import gc
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
#from TabZilla import tabzilla_datasets
from experiments.tabzilla.tabzilla_dataset import TabularDataset



# Load experiment configuration
with open(CONFIG_DIR / "config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)
# Load hyperparameter configuration
with open(CONFIG_DIR / "hyperparameters.yaml", "r", encoding="utf-8") as file:
    hyper = yaml.safe_load(file)


dataset_config = config["dataset"] # Dataset configuration
dataloader_config = config["dataloader"] # DataLoader configuration
experiment_config = config["experiment"] # Experiment configuration
model_config = config["model"] # Model configuration
training_config = config["training"] # Model configuration

hyperparameters = hyper["hyperparameters"]
experiment = hyper["experiment"]


## --------------------- CONFIGURATIONS ---------------------
SAVE_DICTIONARY = experiment_config["save_result"]
NUM_WORKERS = dataloader_config["num_workers"]
DEVICE = experiment_config["device"]
SEED = experiment_config["seed"]


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

def experiment_training(num_hyper=experiment["number_of_configurations_per_split"], random_search = True):    

    ## --------------------- Read Dataset ---------------------
    dataset_path = dataset_config["dataset_path"] #  Path to the example dataset
    dataset_path = (PROJECT_ROOT / dataset_path).resolve()

    # Verify that the dataset directory exists
    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_path}"
            )

    # Verify that all required dataset files exist
    required_files = [
        "metadata.json",
        "split_indeces.npy.gz",
        "X.npy.gz",
        "y.npy.gz"
        ]

    for filename in required_files:
        file_path = dataset_path / filename

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Required dataset file not found: {file_path}"
            )

    # Load the dataset using TabZilla
    dataset = TabularDataset.read(dataset_path)

    ## --------------------- Model Configuration ---------------------
    # Get type of classification problem
    num_classes = dataset.get_metadata()['num_classes']
    if num_classes == 1:
        task_type = "binary"
        binary_class_state = True
    else:
        task_type = "multiclass"
        binary_class_state = False
    print("Task type: ", task_type)

    # Define loss function
    if task_type == "multiclass":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    print("Loss function: ", criterion)

    ## Defining number of experiments 
    trial_data = [] 

    for i, split_dictionary in enumerate(dataset.split_indeces):
        print("--------------- Split " ,i,"---------------")
        # --------------------------------------------------
        # Safe initialization
        # --------------------------------------------------
        stt_model = None
        optimizer = None
        early_stopping = None
        trainable_params = None

        peak_gpu_memory_allocated = None
        peak_gpu_memory_reserved = None

        try:
            train_index = split_dictionary["train"]
            val_index = split_dictionary["val"]
            test_index = split_dictionary["test"]

            if random_search:
                # Select random hyperparameter
                hyper_1 = random.choices(hyperparameters['num_learned_features'],k=num_hyper)
                hyper_2 = random.choices(hyperparameters['num_heads'],k=num_hyper)
                hyper_3 = random.choices(hyperparameters['dim_model'],k=num_hyper)
          
            else:
                keys_hyper = hyperparameters.keys()
                values_hyper = hyperparameters.values()
                combinations_hyper = [dict(zip(keys_hyper, v)) for v in product(*values_hyper)]
                hyper_1 = [c['num_learned_features'] for c in combinations_hyper]
                hyper_2 = [c['num_heads'] for c in combinations_hyper]
                hyper_3 = [c['dim_model'] for c in combinations_hyper] 

            #  --------------------- Preprocess data --------------------- 
            print("Preprocessing data")
            X_train,X_val,X_test,y_train,y_val,y_test,num_idx,cat_idx,cat_cardinalities,n_classes = preprocess_data_tab.preprocess_all_data(dataset,train_index,val_index, test_index,scaler=True)
            num_features = X_train.shape[1]   
            print(X_train.shape)  

            data = preprocess_data_tab.create_dict_of_the_data(X_train,
                                                               y_train,
                                                               X_val,
                                                               y_val,
                                                               X_test,
                                                               y_test,
                                                               task_type,
                                                               len(num_idx),
                                                               len(cat_idx),
                                                               num_idx=num_idx,
                                                               cat_idx=cat_idx)
    
            #  --------------------- Dataloader --------------------- 
            # Seed of the split
            set_seed(SEED)
            g = torch.Generator()
            g.manual_seed(SEED)
            train_dataset = utils.CustomDictDataset(data['train'],
                                                                  binary_class = binary_class_state)
            train_dataloader = DataLoader(train_dataset, 
                                          batch_size = dataloader_config['batch_size'], 
                                          shuffle = True, 
                                          num_workers = NUM_WORKERS,
                                          worker_init_fn = seed_worker(), 
                                          generator = g, 
                                          pin_memory = dataloader_config["pin_memory"])

            test_dataset = utils.CustomDictDataset(data['test'],
                                                                 binary_class = binary_class_state)
            test_dataloader = DataLoader(test_dataset, 
                                         batch_size = dataloader_config['batch_size'], 
                                         shuffle = False, 
                                         num_workers = NUM_WORKERS, 
                                         worker_init_fn = seed_worker(), 
                                         generator = g, 
                                         pin_memory = dataloader_config["pin_memory"],
                                         drop_last = dataloader_config['drop_last'])

            val_dataset = utils.CustomDictDataset(data['val'],
                                                                binary_class = binary_class_state)
            val_dataloader = DataLoader(val_dataset, 
                                        batch_size = dataloader_config['batch_size'], 
                                        shuffle = False, 
                                        num_workers = NUM_WORKERS,
                                        worker_init_fn = seed_worker(), 
                                        generator = g, 
                                        pin_memory = dataloader_config["pin_memory"],
                                        drop_last = dataloader_config['drop_last'])

             ##  --------------------- Training ---------------------    
            
            for j in range(len(hyper_1)):
                # Define model  
                print("Random hyperparameters")
                print(f"Num learned features: {hyper_1[j]}, Num heads: {hyper_2[j]}, Model dimention: {hyper_3[j]}")

                model_config.update({
                    "cat_cardinalities": cat_cardinalities,
                    "num_features" : num_features,
                    "output_dim_supmlp" : num_classes,
                    "d_model": hyper_3[j],
                    "num_learned_features": hyper_1[j],
                    "num_heads": hyper_2[j],
                    "hidden_dim_attmlp": hyper_3[j]*2,
                    "device":  DEVICE,
                    })

            
                # Define model
                print('Intializing model .......')
                mstt_model = MSTT(**model_config).to(DEVICE)

                # Define optimizer
                optimizer = torch.optim.AdamW(mstt_model.parameters(), 
                                              lr = training_config["learning_rate"], 
                                              weight_decay = training_config['weight_decay'])
                early_stopping = training.EarlyStopping(patience = training_config['patience_early_stop'], 
                                                        verbose=False)

                trainable_params = sum(
                    p.numel()
                    for p in mstt_model.parameters()
                    if p.requires_grad
                )
                
        
                epoch_train_time = [] 
                batch_train_time = [] 
                batch_test_time = [] 
                sample_test_time = []     

                # --------- PEAK GPU MEMORY MEASUREMENT --------------
                # Reset AFTER the model/optimizer are created and
                # immediately BEFORE training starts.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(DEVICE)
                    torch.cuda.reset_peak_memory_stats(DEVICE)
                    initial_gpu_memory_allocated = (torch.cuda.memory_allocated(DEVICE)/ 1024**3)
                    initial_gpu_memory_reserved = (torch.cuda.memory_reserved(DEVICE)/ 1024**3)
            
                else:
                    initial_gpu_memory_allocated = None
                    initial_gpu_memory_reserved = None
            
                # ---   TIME MEASUREMENT ---
                trial_start = time.perf_counter()
                
                # ------------------ Traning ----------------------
                for epoch in tqdm(range(training_config['epochs'])):
                    #print(f"Epoch: {epoch}\n--------" )
                    train_loss,train_acc,efficiency_train = training.train(mstt_model,train_dataloader,criterion,optimizer,binary_class=binary_class_state,device=DEVICE,verbose=False)
                    val_loss,val_acc,efficiency_test = training.validation(mstt_model,val_dataloader,criterion,binary_class=binary_class_state,device=DEVICE,verbose=False)
                
                    epoch_train_time.append(efficiency_train["epoch_train_time"]) # tiempo de computo puro
                    batch_train_time.append(efficiency_train["batch_time_mean"]) # tiempo de computo puro
                    batch_test_time.append(efficiency_test["mean_batch_forward_time"]) 
                    sample_test_time.append(efficiency_test["mean_forward_time_per_example"])  
                    # Check early stopping
                    early_stopping.step(val_loss, val_acc, mstt_model,epoch)
                
                    if early_stopping.should_stop:
                        #print(f"Early stopping triggered at epoch {epoch+1}")
                        break
    
                # ==== FINISH CUDA OPERATIONS  =======
                if torch.cuda.is_available():
                    torch.cuda.synchronize(DEVICE)
                trial_end = time.perf_counter()
                trial_total_time = trial_end - trial_start
            
                # =============PEAK GPU MEMORY==============
                # This is measured before test inference.
                # It includes the training + validation loop.
                if torch.cuda.is_available():
                    peak_gpu_memory_allocated = (torch.cuda.max_memory_allocated(DEVICE)/ 1024**3)
                    peak_gpu_memory_reserved = (torch.cuda.max_memory_reserved(DEVICE)/ 1024**3)
            
                print(f"Peak allocated GPU memory: "
                    f"{peak_gpu_memory_allocated:.3f} GB"
                    if peak_gpu_memory_allocated is not None
                    else "GPU memory: N/A")
                        
                
                # ------------------ Testing ----------------------
                mstt_model.load_state_dict(early_stopping.best_model_state)
                y_true,y_preds = testing.evaluate_model(mstt_model,test_dataloader,DEVICE,binary_class_state)
                test_acc = balanced_accuracy_score(y_true, y_preds)
                print("Validation Accuracy_Accuracy: ",val_acc)
                #print("Test_Accuracy: ",test_acc)
                
                trial_data.append({
                    "split": i+1,
                    "status": "success",
                    "num_learned_feat": hyper_1,
                    "num_heads": hyper_2,
                    "model_dim": hyper_3,
                    "hidden_dim_attmlp":  hyper_3*2,
                    "train_acc": train_acc,
                    "best_val_loss": early_stopping.best_loss,
                    "best_val_acc": early_stopping.best_acc,
                    "best_epoch": early_stopping.best_epoch,
                    "test_acc": test_acc,
                    "number_parameter": trainable_params,
                    "number_features": num_features,
                    "epochs_trained": epoch + 1,
                    "total_training_time": trial_total_time,
                    "mean_epoch_training_time": np.mean(epoch_train_time),
                    "mean_batch_training_time": np.mean(batch_train_time),
                    "mean_batch_forward_time": np.mean(batch_test_time),
                    "mean_forward_time_per_example": np.mean(sample_test_time),
                    "initial_gpu_memory_allocated_gb":initial_gpu_memory_allocated,
                    "initial_gpu_memory_reserved_gb":initial_gpu_memory_reserved,
                    "peak_gpu_memory_allocated_gb":peak_gpu_memory_allocated,
                    "peak_gpu_memory_reserved_gb":peak_gpu_memory_reserved})

        # ===  OUT OF MEMORY ====
        except torch.OutOfMemoryError as e:
            print(f"\nCUDA OOM in split {i + 1}")
            print(e)
            oom_peak_allocated = None
            oom_peak_reserved = None
            if torch.cuda.is_available():
                try:
                    oom_peak_allocated = (torch.cuda.max_memory_allocated(DEVICE) / 1024**3)
                    oom_peak_reserved = (torch.cuda.max_memory_reserved(DEVICE)/ 1024**3)
                except Exception:
                    pass
        
        
            trial_data.append({
                "split": i+1,
                "status": "OOM",
                "num_learned_feat": hyper_1,
                "num_heads": hyper_2,
                "model_dim": hyper_3,
                "hidden_dim_attmlp":  hyper_3*2,
                "number_features":
                    num_features if "num_features" in locals()
                    else None,
                "number_parameter": trainable_params,
                "peak_gpu_memory_allocated_gb": oom_peak_allocated,
                "peak_gpu_memory_reserved_gb": oom_peak_reserved,
                "test_acc": None,
                "error": str(e)})
        
        # ==================================================
        # ALWAYS CLEAN GPU/CPU MEMORY
        # ==================================================
        finally:
            if early_stopping is not None:
                del early_stopping
            if optimizer is not None:
                del optimizer
            if stt_model is not None:
                del stt_model
        
            gc.collect()
        
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return trial_data

        
    


if __name__ == "__main__":
    trial_data = experiment_training()
    if SAVE_DICTIONARY:
        trial_data.append(dataset_config)
        trial_data.append(training_config)
        trial_data.append(experiment_config)
        trial_data.append(dataloader_config) 
        trial_data.append(("hidden_dim_supmlp: ", model_config["hidden_dim_supmlp"]))
        trial_data.append(("pos_norm: ", model_config["pos_norm"]))
        trial_data.append(("bias_st_cat: ", model_config["bias_st_cat"]))
        trial_data.append(("positional_enc_status: ", model_config["positional_enc_status"]))
        trial_data.append(("divisor: ", model_config["divisor"]))
        result_path = PROJECT_ROOT / experiment_config["path_save_result"]               
        with open(result_path, 'w') as f:
            json.dump(trial_data, f, indent=4)  
