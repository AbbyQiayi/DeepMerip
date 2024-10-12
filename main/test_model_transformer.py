import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, accuracy_score, auc
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformer import MultiTaskModel  # Assuming a custom transformer model is used


# Function to load the data for a given cell line and downsampling rate
def load_data(cell_line, downsample_rate):
    # Load .npz file with processed data based on the cell line and downsample rate
    file_path = f"../data/dataset/Data_prep_{downsample_rate}_{cell_line}.npz"
    data = np.load(file_path)

    # Log-transforming IP and Input coverage to create x_cov and y_cov (feature inputs for the model)
    x_cov = np.log2(data['x_IP'] + 1) - np.log2(data['x_Input'] + 1)
    y_cov = np.log2(data['y_IP'] + 1) - np.log2(data['y_Input'] + 1)

    # Setting a threshold for determining special cases where both IP and Input are below threshold
    threshold = 1e-10

    # Create masks for positions where both IP and Input are equal to the threshold
    x_mask = (data['x_IP'] == threshold) & (data['x_Input'] == threshold)
    y_mask = (data['y_IP'] == threshold) & (data['y_Input'] == threshold)

    # Count the number of positions that meet the threshold condition
    num_x_conditional = np.sum(x_mask)
    num_y_conditional = np.sum(y_mask)

    # Set covariate values to zero for the positions that meet the threshold condition
    x_cov = np.where(x_mask, np.full(x_cov.shape, 0), x_cov)
    y_cov = np.where(y_mask, np.full(y_cov.shape, 0), y_cov)

    # Transpose the data arrays to align with expected input formats for downstream tasks
    x_cov, x_binary, y_cov, y_binary = (arr.T for arr in (x_cov, data['x_binary'], y_cov, data['y_binary']))

    print(f"Number of x_conditional points: {num_x_conditional}")
    print(f"Number of y_conditional points: {num_y_conditional}")

    return x_cov, x_binary, y_cov, y_binary


# Function to prepare data loaders for test data, combining inputs into the required format
def prepare_dataloaders(cell_line, downsample_rate, batch_size):
    # Load processed data for the given cell line and downsampling rate
    x_cov, x_binary, y_cov, y_binary = load_data(cell_line, downsample_rate)

    # Combine x_cov and x_binary into a single array for model input (two channels)
    x_test = np.stack((x_cov, x_binary), axis=-1)

    # Create a TensorDataset combining the inputs (x_test), regression labels (y_cov), and classification labels (y_binary)
    test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32),
                                 torch.tensor(y_cov, dtype=torch.float32),
                                 torch.tensor(y_binary, dtype=torch.float32))

    # Create a DataLoader for the test set, which loads batches of data during evaluation
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return test_loader


# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define different downsampling rates to test
downsamples = ['0.01dn', '0.02dn', '0.03dn', '0.05dn', '0.1dn']
# Define the cell line to be used for testing
test_cell_lines = ['NEB']

# Lists to store evaluation results
p_corr_results = []
auc_results = []
auprc_results = []

# Loop through each downsampling rate and test it on each test cell line
for downsample in downsamples:
    for test_cell_line in test_cell_lines:
        # Prepare the data loaders for the test set
        test_loader = prepare_dataloaders(test_cell_line, downsample, 64)

        # Model configuration parameters
        sample_length = 5371
        embed_dim = 32  # Embedding dimension for the transformer, must be divisible by num_heads
        input_dim = 2
        num_heads = 8
        num_encoder_layers = 4
        num_decoder_layers = 4
        dropout = 0.1

        # Load a pre-trained model
        model_save_path = '../model/transformer_model_0.1dn.pth'  # Replace with actual model path
        model = MultiTaskModel(input_dim, num_heads, num_encoder_layers, num_decoder_layers, embed_dim, dropout).to(
            device)
        model.load_state_dict(torch.load(model_save_path))

        model.eval()  # Set the model to evaluation mode

        # Lists to store outputs and labels for both regression and classification tasks
        all_test_reg_outputs, all_test_reg_labels = [], []
        all_test_cls_outputs, all_test_cls_labels = [], []
        all_reg_outputs, all_reg_labels = [], []
        all_cls_outputs, all_cls_labels = []
        saved_reg_outputs = []
        saved_cls_outputs = []

        # Iterate through the test set using the data loader
        with torch.no_grad():
            for inputs, reg_labels, cls_labels in test_loader:
                inputs = inputs.to(device)
                reg_targets = reg_labels.to(device)
                cls_targets = cls_labels.to(device)

                # Ensure the inputs are float tensors
                inputs = inputs.float()
                reg_labels = reg_labels.float().unsqueeze(-1)  # Add extra dimension for regression labels
                cls_labels = cls_labels.float().unsqueeze(-1)  # Add extra dimension for classification labels

                # Skip batches that don't match the expected size
                if reg_labels.size(0) != test_loader.batch_size:
                    continue

                # Create a mask where the first channel is non-zero (indicating valid positions)
                mask_padding = (inputs[:, :, 0] != 0)

                # Define a threshold for masking
                threshold = 1e-10

                # Mask inputs and labels where both IP and Input are below the threshold
                label_zeros = (inputs[:, :, 0] == threshold)

                # Adjust the shape of the label_zeros mask if needed
                if label_zeros.shape != inputs.shape[:2]:
                    print("Shape mismatch detected, adjusting label_zeros shape...")
                    label_zeros = label_zeros[:, :, None]  # Adjust shape if necessary

                # Set inputs and labels to zero at masked positions
                inputs[label_zeros] = 0
                reg_targets[label_zeros] = 0
                cls_targets[label_zeros] = 0

                # Forward pass through the model to get predictions
                reg_output, cls_output = model(inputs)

                # Save the raw outputs before masking
                saved_reg_outputs.append(reg_output.cpu().numpy())
                saved_cls_outputs.append(cls_output.cpu().numpy())

                # Mask and collect the outputs and labels for valid positions
                mask_padding = mask_padding.cpu()
                all_reg_outputs.append(reg_output[mask_padding].cpu().numpy().flatten())
                all_reg_labels.append(reg_labels[mask_padding].cpu().numpy().flatten())
                all_cls_outputs.append(cls_output[mask_padding].cpu().numpy().flatten())
                all_cls_labels.append(cls_labels[mask_padding].cpu().numpy().flatten())

            # Concatenate the saved outputs for further analysis
            all_saved_reg_outputs = np.concatenate(saved_reg_outputs)
            all_saved_cls_outputs = np.concatenate(saved_cls_outputs)

            # Optionally, save the model predictions to files for future use
            np.save(f"saved_{test_cell_line}_reg_outputs_{downsample}.npy", all_saved_reg_outputs)
            np.save(f"saved_{test_cell_line}_cls_outputs_{downsample}.npy", all_saved_cls_outputs)

            # Concatenate all outputs and labels for evaluation
            all_reg_outputs = np.concatenate(all_reg_outputs)
            all_reg_labels = np.concatenate(all_reg_labels)
            all_cls_outputs = np.concatenate(all_cls_outputs)
            all_cls_labels = np.concatenate(all_cls_labels)

            # Check for NaN or infinite values in the regression outputs and labels
            nan_in_outputs = np.isnan(all_reg_outputs)
            inf_in_outputs = np.isinf(all_reg_outputs)
            nan_in_labels = np.isnan(all_reg_labels)
            inf_in_labels = np.isinf(all_reg_labels)

            # Replace NaN and infinite values with zeros
            all_reg_outputs[nan_in_outputs | inf_in_outputs] = 0
            all_reg_labels[nan_in_labels | inf_in_labels] = 0

        # Calculate evaluation metrics for the regression and classification tasks
        reg_pearson_corr, _ = pearsonr(all_reg_outputs, all_reg_labels)  # Pearson correlation for regression
        cls_auc = roc_auc_score(all_cls_labels, all_cls_outputs)  # AUC for classification
        precision, recall, _ = precision_recall_curve(all_cls_labels, all_cls_outputs)  # Precision-recall curve
        cls_auprc = auc(recall, precision)  # Area under the precision-recall curve (AUPRC)

        # Print out the results for each test configuration
        print(f"cell_line={test_cell_line}")
        print(f"downsample={downsample}")
        print(f"Regression Pearson Correlation: {reg_pearson_corr}, "
              f"Classification AUC: {cls_auc}, Classification AUPRC: {cls_auprc}")
