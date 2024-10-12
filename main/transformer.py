import torch
import torch.nn as nn
import numpy as np
import time
import logging
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


# Function to load data from .npz file and preprocess it
def load_data(cell_line, downsample_rate):
    # Load the data from the specified file
    file_path = f"../data/dataset/Data_prep_{downsample_rate}_{cell_line}.npz"
    data = np.load(file_path)

    # Calculate log2 transformed coverage for IP and Input samples
    x_cov = np.log2(data['x_IP'] + 1) - np.log2(data['x_Input'] + 1)
    y_cov = np.log2(data['y_IP'] + 1) - np.log2(data['y_Input'] + 1)

    # Set a threshold to mask very low values
    threshold = 1e-10

    # Create masks where both x_IP and x_Input are equal to the threshold
    x_mask = (data['x_IP'] == threshold) & (data['x_Input'] == threshold)
    y_mask = (data['y_IP'] == threshold) & (data['y_Input'] == threshold)

    # Count the number of masked points
    num_x_conditional = np.sum(x_mask)
    num_y_conditional = np.sum(y_mask)

    # Set specific covariate values to zero based on the mask
    x_cov = np.where(x_mask, np.full(x_cov.shape, 0), x_cov)
    y_cov = np.where(y_mask, np.full(y_cov.shape, 0), y_cov)

    # Transpose arrays to match the model input requirements
    x_cov, x_binary, y_cov, y_binary = (arr.T for arr in (x_cov, data['x_binary'], y_cov, data['y_binary']))

    # Print information about the number of conditional points
    print(f"Number of x_conditional points: {num_x_conditional}")
    print(f"Number of y_conditional points: {num_y_conditional}")

    return x_cov, x_binary, y_cov, y_binary


# Function to prepare dataloaders for training and validation
def prepare_dataloaders(cell_line, downsample_rate, batch_size):
    # Load and process the data
    x_cov, x_binary, y_cov, y_binary = load_data(cell_line, downsample_rate)

    # Combine x_cov and x_binary into a single input array
    x_train = np.stack((x_cov, x_binary), axis=-1)

    # Split the data into training and validation sets
    x_train, x_val, y_cov_train, y_cov_val, y_binary_train, y_binary_val = train_test_split(
        x_train, y_cov, y_binary, test_size=0.2, random_state=42
    )

    # Print the shapes of training and validation inputs
    print("train_inputs shape:", x_train.shape)  # (20411, 5371, 2)
    print("val_inputs shape:", y_cov_val.shape)  # (5103, 5371)

    # Create TensorDatasets for training and validation
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                                  torch.tensor(y_cov_train, dtype=torch.float32),
                                  torch.tensor(y_binary_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_cov_val, dtype=torch.float32),
                                torch.tensor(y_binary_val, dtype=torch.float32))

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader


# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the multi-task Transformer model
class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_encoder_layers, num_decoder_layers, embed_dim=32, dropout=0.1):
        super(MultiTaskModel, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Project input dimension to embed_dim
        self.input_projection = nn.Linear(input_dim, embed_dim)

        # Transformer encoder layers
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        # Transformer decoder layers for regression
        self.decoder_reg_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout
        )
        self.decoder_reg = nn.TransformerDecoder(self.decoder_reg_layer, num_layers=num_decoder_layers)

        # Transformer decoder layers for classification
        self.decoder_cls_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout
        )
        self.decoder_cls = nn.TransformerDecoder(self.decoder_cls_layer, num_layers=num_decoder_layers)

        # Output heads for regression and classification tasks
        self.reg_head = nn.Linear(embed_dim, 1)  # Regression head
        self.cls_head = nn.Linear(embed_dim, 1)  # Classification head with sigmoid

    def forward(self, src):
        # Project input to embedding dimension
        src = self.input_projection(src)

        # Pass through the encoder
        memory = self.encoder(src)

        # Pass memory through both regression and classification decoders
        reg_output = self.decoder_reg(memory, memory)
        reg_output = self.reg_head(reg_output)

        cls_output = self.decoder_cls(memory, memory)
        cls_output = self.cls_head(cls_output)
        cls_output = torch.sigmoid(cls_output)  # Apply sigmoid for binary classification

        return reg_output, cls_output


if __name__ == "__main__":
    # Define downsample rates and model parameters
    downsamples = ['0.1dn']
    train_cell_line = 'NEB'
    embed_dim = 32  # Must be divisible by num_heads
    input_dim = 2
    num_heads = 8
    num_encoder_layers = 4
    num_decoder_layers = 4
    dropout = 0.1

    for downsample in downsamples:
        print(f"Running model for downsampling rate: {downsample}")
        # Prepare dataloaders for the given cell line and downsample rate
        train_loader, val_loader = prepare_dataloaders(train_cell_line, downsample, 14)

        # Initialize the multi-task Transformer model
        model = MultiTaskModel(input_dim, num_heads, num_encoder_layers, num_decoder_layers, embed_dim, dropout).to(
            device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)
        reg_loss_fn = nn.MSELoss()
        cls_loss_fn = nn.BCELoss()

        num_epochs = 10
        for epoch in range(num_epochs):
            start_time = time.time()  # Start timing for epoch

            # Training phase
            model.train()
            total_loss = 0
            for inputs, reg_targets, cls_targets in train_loader:
                inputs, reg_targets, cls_targets = inputs.to(device), reg_targets.to(device), cls_targets.to(device)
                optimizer.zero_grad()

                inputs = inputs.float()
                reg_targets = reg_targets.float().unsqueeze(-1)
                cls_targets = cls_targets.float().unsqueeze(-1)

                # Skip batches with size mismatch
                if reg_targets.size(0) != train_loader.batch_size:
                    continue  # Skip this batch

                optimizer.zero_grad()

                # Mask to ignore zero inputs
                mask_padding = (inputs[:, :, 0] != 0)

                # Apply threshold to mask specific labels and targets
                threshold = 1e-10
                label_zeros = (inputs[:, :, 0] == threshold)
                inputs[label_zeros] = 0
                reg_targets[label_zeros] = 0
                cls_targets[label_zeros] = 0

                # Forward pass through the model
                reg_output, cls_output = model(inputs)

                # Compute losses for both tasks
                reg_loss = reg_loss_fn(reg_output[mask_padding], reg_targets[mask_padding])
                cls_loss = cls_loss_fn(cls_output[mask_padding], cls_targets[mask_padding])
                loss = reg_loss + cls_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Print epoch duration and loss
            epoch_duration = time.time() - start_time
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {total_loss / len(train_loader)}, Duration: {epoch_duration:.2f} seconds")

            # Validation phase
            model.eval()
            all_reg_outputs, all_reg_labels = [], []
            all_cls_outputs, all_cls_labels = [], []
            with torch.no_grad():
                for inputs, reg_labels, cls_labels in val_loader:
                    inputs, reg_labels, cls_labels = inputs.to(device), reg_labels.to(device), cls_labels.to(device)

                    inputs = inputs.float()
                    reg_labels = reg_labels.float().unsqueeze(-1)
                    cls_labels = cls_labels.float().unsqueeze(-1)

                    if reg_labels.size(0) != val_loader.batch_size:
                        continue  # Skip this batch

                    # Mask for ignoring padding values
                    mask_padding = (inputs[:, :, 0] != 0)
                    label_zeros = (inputs[:, :, 0] == threshold)
                    inputs[:, :, 0] = torch.where(label_zeros, torch.zeros_like(inputs[:, :, 0]), inputs[:, :, 0])

                    reg_output, cls_output = model(inputs)

                    mask_padding = mask_padding.cpu()
                    all_reg_outputs.append(reg_output[mask_padding].cpu().numpy().flatten())
                    all_reg_labels.append(reg_labels[mask_padding].cpu().numpy().flatten())
                    all_cls_outputs.append(cls_output[mask_padding].cpu().numpy().flatten())
                    all_cls_labels.append(cls_labels[mask_padding].cpu().numpy().flatten())

                # Concatenate outputs and labels for evaluation
                all_reg_outputs = np.concatenate(all_reg_outputs)
                all_reg_labels = np.concatenate(all_reg_labels)
                all_cls_outputs = np.concatenate(all_cls_outputs)
                all_cls_labels = np.concatenate(all_cls_labels)

                # Handle NaN and inf values in outputs and labels
                all_reg_outputs[np.isnan(all_reg_outputs) | np.isinf(all_reg_outputs)] = 0
                all_reg_labels[np.isnan(all_reg_labels) | np.isinf(all_reg_labels)] = 0

            # Calculate regression correlation and classification AUC/AUPRC
            reg_corr = pearsonr(all_reg_outputs, all_reg_labels)[0]
            cls_auc = roc_auc_score(all_cls_labels, all_cls_outputs)
            precision, recall, _ = precision_recall_curve(all_cls_labels, all_cls_outputs)
            cls_auprc = auc(recall, precision)

            # Print evaluation metrics for this epoch
            print(
                f"Epoch {epoch + 1}, Regression Correlation: {reg_corr}, Classification AUC: {cls_auc}, AUPRC: {cls_auprc}")

        # Save the trained model
        model_save_path = f'../model/transformer_model_{downsample}.pth'
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
