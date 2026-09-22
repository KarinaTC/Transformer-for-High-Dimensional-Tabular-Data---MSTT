import torch
import torch.nn as nn
import numpy as np
import copy
import time
from sklearn.metrics import balanced_accuracy_score

class EarlyStopping:
    def __init__(self, patience = 5, 
                 min_delta = 0.0, 
                 verbose = False, 
                 save_path = None):
        """
        patience: number of epochs to wait without improvement
        min_delta: minimum improvement to count as improvement
        save_path: if provided, saves the best model here
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.save_path = save_path

        self.best_loss = float("inf")
        self.best_acc = 0.0
        self.counter = 0
        self.should_stop = False
        self.best_model_state = None
        self.best_epoch =  None
        self.first_time_flag = True

    def step(self, current_loss, current_acc=None, model=None, epoch=None):
        """Call this after each validation epoch"""

        if self.first_time_flag:
            self.best_loss = current_loss
            if current_acc is not None:
                self.best_acc = current_acc
            if epoch is not None:
                self.best_epoch = epoch
            self.counter = 0
            self.first_time_flag = False

            # Save best model if requested
            if model is not None and self.save_path is not None:
                torch.save(model.state_dict(), self.save_path)
            elif model is not None:
                # Keep in memory
                self.best_model_state = copy.deepcopy(model.state_dict())
            return

        improved = current_loss < self.best_loss - self.min_delta
        if improved:
            self.best_loss = current_loss
            if current_acc is not None:
                self.best_acc = current_acc
            if epoch is not None:
                self.best_epoch = epoch
            self.counter = 0

            # Save best model if requested
            if model is not None and self.save_path is not None:
                torch.save(model.state_dict(), self.save_path)
            elif model is not None:
                # Keep in memory
                self.best_model_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement in {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                self.should_stop = True



def train(model: nn.Module, 
          dataloader,
          loss_fn,
          optimizer,
          device,
          binary_class=True,
          verbose=True):
    time_train_per_batch = []
    model.train()
    train_loss = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    epoch_start = time.perf_counter()

    for batch in dataloader:
         ## Measure Training Time per Epoch
        if torch.cuda.is_available(): torch.cuda.synchronize()
        batch_start = time.perf_counter()
        
        labels = batch["target"].to(device, non_blocking=True)
        cont_feat = batch.get("x_cont", None)
        cat_feat = batch.get("x_cat", None)
        
        if cont_feat is not None: cont_feat = cont_feat.to(device, non_blocking=True)
        if cat_feat is not None: cat_feat = cat_feat.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        y_pred_logs = model(cont_feat, cat_feat)
        
        if binary_class == True:
            y_pred_logs = y_pred_logs.squeeze() # Quita dimensiones extra de [bacth,1] a [batch]
            loss = loss_fn(y_pred_logs, labels)
        else:
            loss = loss_fn(y_pred_logs, labels)
          
        # 4. Backpropagate
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
        # 5. Update weights
        optimizer.step()

        if torch.cuda.is_available(): torch.cuda.synchronize()
        batch_end= time.perf_counter()

        #Track time & loss
        time_train_per_batch.append(batch_end - batch_start)
        train_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
      
        if binary_class:
            with torch.no_grad():
                preds = torch.round(torch.sigmoid(y_pred_logs))
        else:
            preds = y_pred_logs.argmax(dim=1)
            
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    epoch_end = time.perf_counter()

    # Cálculos finales (fuera del cronómetro)
    epoch_loss = train_loss / total_samples
    y_pred = torch.cat(all_preds).int().numpy()
    y_true = torch.cat(all_labels).int().numpy()
    epoch_acc = balanced_accuracy_score(y_true, y_pred)

    # Promedio descartando el primer batch (Warm-up)
    batch_time_mean = np.mean(time_train_per_batch[1:]) if len(time_train_per_batch) > 1 else time_train_per_batch[0]
    epoch_train_time = epoch_end - epoch_start

    if verbose:
        print(f"Epoch train time: {epoch_train_time:.7f}s")
        print(f"Train loss: {epoch_loss:.4f} | Train acc: {epoch_acc*100.:.4f}%")
    return epoch_loss, epoch_acc, {"batch_time_mean":batch_time_mean,"epoch_train_time":epoch_train_time}



def validation(model: nn.Module,
         dataloader,
         loss_fn,
         device,
         binary_class=True,
         verbose=True):
    model.eval()
    test_loss = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    forward_batch_times = []
    # Sincronización para tiempo real de GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    val_start = time.perf_counter()

    with torch.inference_mode():
        for batch in dataloader:

            labels = batch["target"].to(device, non_blocking=True)
            
            cont_feat = batch.get("x_cont", None)
            cat_feat = batch.get("x_cat", None)
        
            if cont_feat is not None: cont_feat = cont_feat.to(device, non_blocking=True)
            if cat_feat is not None: cat_feat = cat_feat.to(device, non_blocking=True)

            # --- MEDICIÓN DEL FORWARD (Inicio) ---
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t_fwd_start = time.perf_counter()
            
            test_pred_logs = model(cont_feat, cat_feat)

            if torch.cuda.is_available(): torch.cuda.synchronize()
            t_fwd_end = time.perf_counter()

            forward_batch_times.append(t_fwd_end - t_fwd_start)

            if binary_class == True:
                test_pred_logs = test_pred_logs.squeeze()
                loss = loss_fn(test_pred_logs, labels)
                test_pred_prob = torch.sigmoid(test_pred_logs)
                test_pred = torch.round(test_pred_prob)
            else:
                loss = loss_fn(test_pred_logs, labels)
                test_pred = torch.softmax(test_pred_logs, dim=1).argmax(dim=1)

            test_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            all_preds.append(test_pred.detach().cpu())
            all_labels.append(labels.detach().cpu())

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    val_end = time.perf_counter()

    # Cálculos finales
    epoch_loss = test_loss / total_samples
    y_pred = torch.cat(all_preds).int().numpy()
    y_true = torch.cat(all_labels).int().numpy()
    epoch_acc = balanced_accuracy_score(y_true, y_pred)

    # --- CÁLCULOS DE EFICIENCIA FINAL ---
    # Promedio por batch (descartando el primero si quieres más precisión)
    mean_batch_forward_time = np.mean(forward_batch_times[1:]) if len(forward_batch_times) > 1 else forward_batch_times[0]
    
    # Tiempo por ejemplo (Latencia pura de inferencia)
    # Dividimos la suma total de los forwards entre el total de filas procesadas
    mean_forward_time_per_example = np.sum(forward_batch_times) / total_samples
    
    val_duration = val_end - val_start
    
    if verbose:
        print(f"Val loss: {epoch_loss:.4f} | Val acc: {epoch_acc*100.:.4f}% | Time: {val_duration:.4f}s")
    return epoch_loss, epoch_acc,{"val_duration":val_duration,"mean_batch_forward_time": mean_batch_forward_time,
        "mean_forward_time_per_example": mean_forward_time_per_example}