
import torch
from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np

def calculate_metrics(y_true, y_pred):

    device = y_true.device
    y_pred = y_pred.to(device)
    y_pred_binary = (y_pred > 0.4).float()

    TP = torch.sum((y_true == 1) & (y_pred_binary == 1)).item()
    TN = torch.sum((y_true == 0) & (y_pred_binary == 0)).item()
    FP = torch.sum((y_true == 0) & (y_pred_binary == 1)).item()
    FN = torch.sum((y_true == 1) & (y_pred_binary == 0)).item()

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-10)
    sensitivity = TP / (TP + FN + 1e-10)  
    precision = TP / (TP + FP + 1e-10)   
    specificity = TN / (TN + FP + 1e-10)  
    F1_score = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-10)
    REC = sensitivity

    return accuracy, precision, F1_score, REC

def cal_auc_aupr(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    au_prc = auc(recall, precision)
    return roc_auc, au_prc