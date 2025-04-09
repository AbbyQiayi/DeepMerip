import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, accuracy_score, auc
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from ResNet import MultiTaskModel  # Assuming this is your custom model


# Function to load data for a given cell line and downsampling rate
def load_data(cell_line, downsample_rate):
    # Load the preprocessed data from .npz file
    file_path = f"autodl-tmp/Data_prep_{downsample_rate}_{cell_line}.npz"
    data = np.load(file_path)

    # Log transformation of IP and Input coverage
    x_cov = np.log2(data['x_IP'] + 1) - np.log2(data['x_Input'] + 1)
    y_cov = np.log2(data['y_IP'] + 1) - np.log2(data['y_Input'] + 1)

    # Set a threshold to define conditions for further processing
    threshold = 1e-10

    # Create masks where IP and Input coverage values are equal to the threshold
    x_mask = (data['x_IP'] == threshold) & (data['x_Input'] == threshold)
    y_mask = (data['y_IP'] == threshold) & (data['y_Input'] == threshold)

    # Count how many sites match the condition
    num_x_conditional = np.sum(x_mask)
    num_y_conditional = np.sum(y_mask)

    # Set specific covariate values to 0 based on the mask
    x_cov = np.where(x_mask, np.full(x_cov.shape, 0), x_cov)
    y_cov = np.where(y_mask, np.full(y_cov.shape, 0), y_cov)

    # Transpose the data arrays for compatibility with further processing steps
    x_cov, x_binary, y_cov, y_binary = (arr.T for arr in (x_cov, data['x_binary'], y_cov, data['y_binary']))

    print(f"Number of x_conditional points: {num_x_conditional}")
    print(f"Number of y_conditional points: {num_y_conditional}")

    return x_cov, x_binary, y_cov, y_binary, x_cov.shape[1]


# Function to prepare dataset for PyTorch
def prepare_dataset(x_cov, y_cov, y_binary, sample_length):
    # Convert NumPy arrays to PyTorch tensors, reshaping as needed
    if isinstance(x_cov, np.ndarray):
        inputs_tensor = torch.from_numpy(x_cov.reshape(-1, 2, sample_length)).float()
    else:
        inputs_tensor = x_cov.float()  # If already a tensor, just ensure it is a float type

    # Reshape covariate and binary data as well
    reg_tensor = torch.from_numpy(y_cov.reshape(-1, 1, sample_length)).float()
    cls_tensor = torch.from_numpy(y_binary.reshape(-1, 1, sample_length)).float()

    # Create a TensorDataset combining inputs and targets
    dataset = TensorDataset(inputs_tensor, reg_tensor, cls_tensor)

    return dataset


# Function to prepare test set based on downsampling rate
def test_set_prep(test_cell_line, downsample):
    # Load and process the test data
    x_cov, x_binary, y_cov, y_binary, sample_length = load_data(test_cell_line, downsample)

    # Concatenate covariates and binary labels along the channel dimension
    x_train = np.concatenate((np.expand_dims(x_cov, axis=1), np.expand_dims(x_binary, axis=1)), axis=1)

    # Prepare dataset
    test_dataset = prepare_dataset(x_train, y_cov, y_binary, sample_length)

    # Create a DataLoader to batch and shuffle the test data
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return test_loader


# Check if a GPU is available for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the downsampling rates and test cell lines to evaluate
downsamples = ['0.05dn', '0.1dn','0.2dn','0.3dn'] # change it with your down sample rate
test_cell_lines = ['SYSY'] # change it with your test cell line

# Initialize lists to store results for further analysis
p_corr_results = []
auc_results = []
auprc_results = []

# Loop through the downsampling rates and test cell lines
for downsample in downsamples:
    for test_cell_line in test_cell_lines:
        # Prepare test data loader
        test_loader = test_set_prep(test_cell_line=test_cell_line, downsample=downsample)

        # Load the sample length for model initialization
        _, _, _, _, sample_length = load_data(test_cell_line, downsample)

        # Load the pre-trained model
        model_save_path = f'autodl-tmp/resnet_model_0.1dn.pth'  # Path to the saved model
        model = MultiTaskModel(sample_length).to(device)
        model.load_state_dict(torch.load(model_save_path))

        model.eval()  # Set the model to evaluation mode

        # Initialize lists to store outputs and labels
        all_test_reg_outputs, all_test_reg_labels = [], []
        all_test_cls_outputs, all_test_cls_labels = [], []
        all_reg_outputs, all_reg_labels = [], []
        all_cls_outputs, all_cls_labels = [], []
        saved_reg_outputs = []
        saved_cls_outputs = []

        # Disable gradient computation for evaluation
        with torch.no_grad():
            for inputs, reg_labels, cls_labels in test_loader:
                inputs = inputs.to(device)
                reg_targets = reg_labels.to(device)
                cls_targets = cls_labels.to(device)

                # Ensure inputs are float tensors
                inputs = inputs.float()
                reg_labels = reg_labels.float()
                cls_labels = cls_labels.float()

                # Skip batch if the size does not match
                if reg_labels.size(0) != test_loader.batch_size:
                    continue  # Skip incomplete batches

                # Create a mask for inputs where the first channel is non-zero (to handle padding)
                mask_padding = (inputs[:, 0, :] != 0)
                mask_padding_inputs = mask_padding.unsqueeze(1).expand_as(reg_labels)
                mask_padding_reg = mask_padding.unsqueeze(1).expand_as(reg_labels)

                # Define a threshold for zero masking
                threshold = 1e-10

                # Mask input values below the threshold
                label_zeros = (inputs[:, 0, :] == threshold)
                label_zeros_inputs = label_zeros.unsqueeze(1).expand_as(inputs)
                label_zeros_reg = label_zeros.unsqueeze(1).expand_as(reg_labels)

                inputs[label_zeros_inputs] = 0

                # Perform forward pass through the model
                reg_output, cls_output = model(inputs)

                # Save outputs before any masking
                saved_reg_outputs.append(reg_output.cpu().numpy())
                saved_cls_outputs.append(cls_output.cpu().numpy())

                # Apply padding mask and collect outputs/labels
                mask_padding = mask_padding.cpu()
                all_reg_outputs.append(reg_output[mask_padding.unsqueeze(1)].cpu().numpy().flatten())
                all_reg_labels.append(reg_labels[mask_padding.unsqueeze(1)].cpu().numpy().flatten())
                all_cls_outputs.append(cls_output[mask_padding.unsqueeze(1)].cpu().numpy().flatten())
                all_cls_labels.append(cls_labels[mask_padding.unsqueeze(1)].cpu().numpy().flatten())

            # Concatenate saved outputs
            all_saved_reg_outputs = np.concatenate(saved_reg_outputs)
            all_saved_cls_outputs = np.concatenate(saved_cls_outputs)

            # Optionally, save outputs to a file
            np.save(f"saved_{test_cell_line}_reg_outputs_{downsample}.npy", all_saved_reg_outputs)
            np.save(f"saved_{test_cell_line}_cls_outputs_{downsample}.npy", all_saved_cls_outputs)

            # Concatenate all outputs and labels across batches
            all_reg_outputs = np.concatenate(all_reg_outputs)
            all_reg_labels = np.concatenate(all_reg_labels)
            all_cls_outputs = np.concatenate(all_cls_outputs)
            all_cls_labels = np.concatenate(all_cls_labels)

            # Check for NaN or infinite values in regression outputs/labels
            nan_in_outputs = np.isnan(all_reg_outputs)
            inf_in_outputs = np.isinf(all_reg_outputs)
            nan_in_labels = np.isnan(all_reg_labels)
            inf_in_labels = np.isinf(all_reg_labels)

            # Replace NaN or infinite values with zeros
            all_reg_outputs[nan_in_outputs | inf_in_outputs] = 0
            all_reg_labels[nan_in_labels | inf_in_labels] = 0

        # Calculate performance metrics for regression and classification tasks
        reg_pearson_corr, _ = pearsonr(all_reg_outputs, all_reg_labels)
        cls_auc = roc_auc_score(all_cls_labels, all_cls_outputs)
        precision, recall, _ = precision_recall_curve(all_cls_labels, all_cls_outputs)
        cls_auprc = auc(recall, precision)

        # Print performance results for each test case
        print(f"cell_line={test_cell_line}")
        print(f"downsample={downsample}")
        print(f"Regression Pearson Correlation: {reg_pearson_corr}, "
              f"Classification AUC: {cls_auc}, Classification AUPRC: {cls_auprc}")
