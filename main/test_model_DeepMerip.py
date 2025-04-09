import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, accuracy_score, auc
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from DeepTrans import MultiTaskModel  # Ensure this matches the training model definition
import time  # Added for runtime measurement
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif  # Added for mutual information

def load_data(cell_line, downsample_rate):
    # Existing code unchanged
    file_path = f"autodl-tmp/Data_prep_{downsample_rate}_{cell_line}.npz"
    data = np.load(file_path, allow_pickle=True)

    x_cov = np.log2(data['x_IP'] + 1) - np.log2(data['x_Input'] + 1)
    y_cov = np.log2(data['y_IP'] + 1) - np.log2(data['y_Input'] + 1)

    threshold = 1e-10
    x_mask = (data['x_IP'] == threshold) & (data['x_Input'] == threshold)
    y_mask = (data['y_IP'] == threshold) & (data['y_Input'] == threshold)

    x_cov = np.where(x_mask, 0, x_cov)
    y_cov = np.where(y_mask, 0, y_cov)

    x_cov, x_binary, y_cov, y_binary = (arr.T for arr in (x_cov, data['x_binary'], y_cov, data['y_binary']))
    sample_length = x_cov.shape[1]
    
    return x_cov, x_binary, y_cov, y_binary, sample_length

def prepare_dataloaders(cell_line, downsample_rate, batch_size):
    # Existing code unchanged
    x_cov, x_binary, y_cov, y_binary, sample_length = load_data(cell_line, downsample_rate)
    x_test = np.stack([x_cov, x_binary], axis=1)  # Shape (num_samples, 2, seq_len)
    
    y_cov_reshaped = y_cov.reshape(-1, 1, sample_length)
    y_binary_reshaped = y_binary.reshape(-1, 1, sample_length)
    
    test_dataset = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_cov_reshaped, dtype=torch.float32),
        torch.tensor(y_binary_reshaped, dtype=torch.float32)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader, sample_length

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

downsamples = ['0.05dn', '0.1dn', '0.2dn', '0.3dn']
test_cell_lines = ['SYSY']

# Initialize results lists for new metrics
p_corr_results = []
mse_results = []
dcor_results = []
kl_results = []
mi_reg_results = []
mi_cls_results = []
auc_results = []
auprc_results = []
runtime_results = []

for downsample in downsamples:
    for test_cell_line in test_cell_lines:
        test_loader, sample_length = prepare_dataloaders(test_cell_line, downsample, 64)
        
        model = MultiTaskModel(
            sample_length, 
            d_model=64, 
            nhead=4, 
            num_layers=3, 
            dim_feedforward=256, 
            dropout=0.1
        ).to(device)
        
        model_save_path = f'autodl-tmp/deepmerip_model_0.1dn.pth'
        model.load_state_dict(torch.load(model_save_path))
        model.eval()

        all_reg_outputs, all_reg_labels = [], []
        all_cls_outputs, all_cls_labels = [], []
        total_time = 0.0  # Track total inference time

        with torch.no_grad():
            for inputs, reg_labels, cls_labels in test_loader:
                inputs = inputs.to(device)
                reg_targets = reg_labels.to(device)
                cls_targets = cls_labels.to(device)

                # Create masks
                mask_padding = (inputs[:, 0, :] != 0)
                threshold = 1e-10
                label_zeros = (inputs[:, 0, :] == threshold)

                # Apply masks
                inputs[label_zeros.unsqueeze(1).expand(-1, 2, -1)] = 0
                reg_targets[label_zeros.unsqueeze(1)] = 0
                cls_targets[label_zeros.unsqueeze(1)] = 0

                # Time the forward pass
                start_time = time.time()
                reg_output, cls_output = model(inputs)
                if device.type == 'cuda':
                    torch.cuda.synchronize()  # Ensure accurate timing on GPU
                batch_time = time.time() - start_time
                total_time += batch_time

                # Collect valid positions
                mask = mask_padding.unsqueeze(1)
                all_reg_outputs.extend(reg_output[mask].cpu().numpy())
                all_reg_labels.extend(reg_targets[mask].cpu().numpy())
                all_cls_outputs.extend(cls_output[mask].cpu().numpy())
                all_cls_labels.extend(cls_targets[mask].cpu().numpy())

        # Calculate regression metrics
        reg_pearson_corr, _ = pearsonr(all_reg_outputs, all_reg_labels)
        mse = np.mean((np.array(all_reg_outputs) - np.array(all_reg_labels)) ** 2)
        reg_mi = mutual_info_regression(np.array(all_reg_outputs).reshape(-1, 1), np.array(all_reg_labels))[0]

        # Calculate classification metrics
        cls_probs = torch.sigmoid(torch.tensor(all_cls_outputs)).numpy()
        cls_probs = np.clip(cls_probs, 1e-12, 1 - 1e-12)  # Avoid log(0)
        kl = - (np.array(all_cls_labels) * np.log(cls_probs) + (1 - np.array(all_cls_labels)) * np.log(1 - cls_probs))
        kl_divergence = np.mean(kl)
        cls_auc = roc_auc_score(all_cls_labels, all_cls_outputs)
        precision, recall, _ = precision_recall_curve(all_cls_labels, all_cls_outputs)
        cls_auprc = auc(recall, precision)
        cls_mi = mutual_info_classif(cls_probs.reshape(-1, 1), np.array(all_cls_labels).ravel(), discrete_features=False)[0]

        # Append results
        p_corr_results.append(reg_pearson_corr)
        mse_results.append(mse)
        kl_results.append(kl_divergence)
        mi_reg_results.append(reg_mi)
        mi_cls_results.append(cls_mi)
        auc_results.append(cls_auc)
        auprc_results.append(cls_auprc)
        runtime_results.append(total_time)

        # Print all metrics
        print(f"Test Results - {test_cell_line} {downsample}:")
        print(f"Pearson: {reg_pearson_corr:.4f}, MSE: {mse:.4f}, MI (reg): {reg_mi:.4f}")
        print(f"AUC: {cls_auc:.4f}, AUPRC: {cls_auprc:.4f}, KL Div: {kl_divergence:.4f}, MI (cls): {cls_mi:.4f}")
        print(f"Runtime: {total_time:.2f} seconds\n")