import torch
import torch.nn as nn
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.style.use('ggplot')



def evaluate_model(model: nn.Module,
                   dataloader,
                   device,
                   binary_class = True):
    y_preds = []
    y_true = []
    model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            y_test = batch["target"].to(device, non_blocking=True)
            
            cont_feat = batch.get("x_cont", None)
            cat_feat = batch.get("x_cat", None)
        
            if cont_feat is not None: cont_feat = cont_feat.to(device, non_blocking=True)
            if cat_feat is not None: cat_feat = cat_feat.to(device, non_blocking=True)
            
            try:   
                y_pred_logs = model(cont_feat, cat_feat)
                if binary_class == True:
                    y_pred = torch.sigmoid(y_pred_logs)
                    y_pred = torch.round(y_pred) # y_pred>0.50 -> y_pred=1 y y_pred<=0.50 -> y_pred=0
                else:
                    y_pred = torch.softmax(y_pred_logs, dim=1).argmax(dim=1)
                y_preds.append(y_pred.cpu())
                y_true.append(y_test.cpu())
            except:
                break
        
    y_preds = torch.cat(y_preds)
    y_true = torch.cat(y_true)
    return y_true,y_preds


def plot_confussion_matrix(y_true,
                           y_preds):
    cf_matrix = confusion_matrix(y_true, y_preds)
    plt.figure(figsize=(8,6))
    # Modificar el tamaño del texto
    sns.set(font_scale = 1.1)
    # Plot Matriz de confusión con heatmaps
    # Parámetros:
    # - first param - Matriz de confusión en un formato array
    # - annot = True: Muestra los números en cada celda del heatmap
    # - fmt = 'd': Muestra los números como enteros.
    ax = sns.heatmap(cf_matrix, annot=True, fmt='d', )
    # set x-axis label and ticks.
    ax.set_xlabel("Predicción", fontsize=14, labelpad=20)
    # set y-axis label and ticks
    ax.set_ylabel("Valor real", fontsize=14, labelpad=20)
    # set plot title
    ax.set_title("Matriz de confusión del clasificador", fontsize=14, pad=20)
    plt.show()