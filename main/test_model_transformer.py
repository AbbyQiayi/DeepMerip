import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             precision_recall_curve, accuracy_score, 
                             auc, mean_squared_error, mutual_info_score)
from scipy.stats import pearsonr, entropy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformer import MultiTaskModel  # Your custom model import

# Modified distance correlation calculation using batching
def batched_distance_correlation(x, y, batch_size=10000):
    """Calculate distance correlation in batches to save memory"""
    n = len(x)
    dcov = 0.0
    var_x = 0.0
    var_y = 0.0
    
    for i in range(0, n, batch_size):
        batch_x = x[i:i+batch_size]
        batch_y = y[i:i+batch_size]
        
        # Calculate distance matrices for batch
        a = np.abs(batch_x[:, None] - batch_x[None, :])
        b = np.abs(batch_y[:, None] - batch_y[None, :])
        
        # Calculate centered distances
        a_row_mean = a.mean(axis=1, keepdims=True)
        a_col_mean = a.mean(axis=0, keepdims=True)
        a_mean = a.mean()
        A = a - a_row_mean - a_col_mean + a_mean
        
        b_row_mean = b.mean(axis=1, keepdims=True)
        b_col_mean = b.mean(axis=0, keepdims=True)
        b_mean = b.mean()
        B = b - b_row_mean - b_col_mean + b_mean
        
        # Update covariance and variances
        dcov += (A * B).sum()
        var_x += (A * A).sum()
        var_y += (B * B).sum()
        
    return np.sqrt(dcov / np.sqrt(var_x * var_y))

# Helper functions for new metrics
def calculate_kl_divergence(true, pred, num_bins=10):
    """Calculate KL divergence between true and predicted distributions"""
    min_val = min(np.min(true), np.min(pred))
    max_val = max(np.max(true), np.max(pred))
    bins = np.linspace(min_val, max_val, num_bins)
    
    hist_true, _ = np.histogram(true, bins=bins)
    hist_pred, _ = np.histogram(pred, bins=bins)
    
    epsilon = 1e-10
    hist_true = hist_true.astype(float) + epsilon
    hist_pred = hist_pred.astype(float) + epsilon
    
    prob_true = hist_true / np.sum(hist_true)
    prob_pred = hist_pred / np.sum(hist_pred)
    
    return entropy(prob_true, prob_pred)

def calculate_mutual_info(true, pred, num_bins=10):
    """Calculate mutual information between true and predicted values"""
    min_val = min(np.min(true), np.min(pred))
    max_val = max(np.max(true), np.max(pred))
    bins = np.linspace(min_val, max_val, num_bins)
    
    true_discrete = np.digitize(true, bins)
    pred_discrete = np.digitize(pred, bins)
    
    return mutual_info_score(true_discrete, pred_discrete)

# Function to load data
def load_data(cell_line, downsample_rate):
    file_path = f"autodl-tmp/Data_prep_{downsample_rate}_{cell_line}.npz"
    data = np.load(file_path)

    # Log transformation
    x_cov = np.log2(data['x_IP'] + 1) - np.log2(data['x_Input'] + 1)
    y_cov = np.log2(data['y_IP'] + 1) - np.log2(data['y_Input'] + 1)

    # Threshold processing
    threshold = 1e-10
    x_mask = (data['x_IP'] == threshold) & (data['x_Input'] == threshold)
    y_mask = (data['y_IP'] == threshold) & (data['y_Input'] == threshold)

    x_cov = np.where(x_mask, 0, x_cov)
    y_cov = np.where(y_mask, 0, y_cov)

    # Transpose arrays
    x_cov, x_binary, y_cov, y_binary = (arr.T for arr in (x_cov, data['x_binary'], y_cov, data['y_binary']))

    return x_cov, x_binary, y_cov, y_binary, x_cov.shape[1]

# Dataset preparation
def prepare_dataset(x_cov, y_cov, y_binary, sample_length):
    if isinstance(x_cov, np.ndarray):
        inputs_tensor = torch.from_numpy(x_cov.reshape(-1, 2, sample_length)).float()
    else:
        inputs_tensor = x_cov.float()

    reg_tensor = torch.from_numpy(y_cov.reshape(-1, 1, sample_length)).float()
    cls_tensor = torch.from_numpy(y_binary.reshape(-1, 1, sample_length)).float()

    return TensorDataset(inputs_tensor, reg_tensor, cls_tensor)

# Test set preparation
def test_set_prep(test_cell_line, downsample):
    x_cov, x_binary, y_cov, y_binary, sample_length = load_data(test_cell_line, downsample)
    x_train = np.concatenate((np.expand_dims(x_cov, axis=1), np.expand_dims(x_binary, axis=1)), axis=1)
    test_dataset = prepare_dataset(x_train, y_cov, y_binary, sample_length)
    return DataLoader(test_dataset, batch_size=64, shuffle=False)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
downsamples = ['0.05dn', '0.1dn', '0.2dn', '0.3dn']
test_cell_lines = ['SYSY']

# Initialize results storage
results = {
    'cell_line': [],
    'downsample': [],
    'pearson': [],
    'mse': [],
    'dcor': [],
    'kl_div': [],
    'mutual_info': [],
    'auc': [],
    'auprc': [],
    'runtime': []
}

# Main evaluation loop
for downsample in downsamples:
    for test_cell_line in test_cell_lines:
        start_time = time.time()
        
        # Data preparation
        test_loader = test_set_prep(test_cell_line, downsample)
        _, _, _, _, sample_length = load_data(test_cell_line, downsample)
        
        # Model setup
        model_save_path = f'autodl-tmp/transformer_model_0.1dn.pth'
        model = MultiTaskModel(sample_length).to(device)
        model.load_state_dict(torch.load(model_save_path))
        model.eval()

        # Inference
        all_reg_outputs, all_reg_labels = [], []
        all_cls_outputs, all_cls_labels = [], []
        
        with torch.no_grad():
            for inputs, reg_labels, cls_labels in test_loader:
                inputs = inputs.float().to(device)
                
                # Forward pass
                reg_output, cls_output = model(inputs)
                
                # Collect outputs
                mask = (inputs[:, 0, :] != 0).unsqueeze(1).cpu()
                all_reg_outputs.append(reg_output[mask].cpu().numpy())
                all_reg_labels.append(reg_labels[mask].cpu().numpy())
                all_cls_outputs.append(cls_output[mask].cpu().numpy())
                all_cls_labels.append(cls_labels[mask].cpu().numpy())

        # Concatenate results
        all_reg_outputs = np.concatenate(all_reg_outputs)
        all_reg_labels = np.concatenate(all_reg_labels)
        all_cls_outputs = np.concatenate(all_cls_outputs)
        all_cls_labels = np.concatenate(all_cls_labels)

        # Calculate metrics
        runtime = time.time() - start_time
        pearson_corr, _ = pearsonr(all_reg_outputs, all_reg_labels)
        mse = mean_squared_error(all_reg_labels, all_reg_outputs)
        dcor_val = batched_distance_correlation(all_reg_outputs, all_reg_labels)
        kl_div = calculate_kl_divergence(all_reg_labels, all_reg_outputs)
        mi = calculate_mutual_info(all_reg_labels, all_reg_outputs)
        cls_auc = roc_auc_score(all_cls_labels, all_cls_outputs)
        precision, recall, _ = precision_recall_curve(all_cls_labels, all_cls_outputs)
        cls_auprc = auc(recall, precision)

        # Store results
        results['cell_line'].append(test_cell_line)
        results['downsample'].append(downsample)
        results['pearson'].append(pearson_corr)
        results['mse'].append(mse)
        results['dcor'].append(dcor_val)
        results['kl_div'].append(kl_div)
        results['mutual_info'].append(mi)
        results['auc'].append(cls_auc)
        results['auprc'].append(cls_auprc)
        results['runtime'].append(runtime)

        # Print results
        print(f"\nResults for {test_cell_line} ({downsample}):")
        print(f"Regression Metrics:")
        print(f"  Pearson: {pearson_corr:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  dCor: {dcor_val:.4f}")
        print(f"  KL Divergence: {kl_div:.4f}")
        print(f"  Mutual Info: {mi:.4f}")
        print(f"Classification Metrics:")
        print(f"  AUC: {cls_auc:.4f}")
        print(f"  AUPRC: {cls_auprc:.4f}")
        print(f"Runtime: {runtime:.2f} seconds")

# Convert results to DataFrame and save
results_df = pd.DataFrame(results)
print("\nFinal Results Summary:")
print(results_df)
results_df.to_csv('evaluation_results.csv', index=False)