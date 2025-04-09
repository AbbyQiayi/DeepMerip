import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import TensorDataset, DataLoader


# Helper functions for Mutual Information and KL Divergence
def compute_mi(x, y, bins=20):
    hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = hist_2d / hist_2d.sum()
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    mi = 0.0
    for i in range(pxy.shape[0]):
        for j in range(pxy.shape[1]):
            if pxy[i, j] > 0:
                mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j] + 1e-10))
    return mi


def compute_kl_div(cls_probs, cls_labels, eps=1e-10):
    kl = cls_labels * np.log((cls_labels + eps) / (cls_probs + eps)) + (1 - cls_labels) * np.log(
        (1 - cls_labels + eps) / (1 - cls_probs + eps))
    return np.mean(kl)

def load_data(cell_line, downsample_rate):
    file_path = f"autodl-tmp/Data_prep_{downsample_rate}_{cell_line}.npz"
    data = np.load(file_path,allow_pickle=True)

    x_cov = np.log2(data['x_IP'] + 1) - np.log2(data['x_Input'] + 1)
    y_cov = np.log2(data['y_IP'] + 1) - np.log2(data['y_Input'] + 1)

    # set the threshold
    threshold = 1e-10

    # mask x_IP and x_Input
    x_mask = (data['x_IP'] == threshold) & (data['x_Input'] == threshold)
    y_mask = (data['y_IP'] == threshold) & (data['y_Input'] == threshold)

    # debug: sum the pos meeting conditions
    num_x_conditional = np.sum(x_mask)
    num_y_conditional = np.sum(y_mask)

    # set x_cov and y_cov based on mask
    x_cov = np.where(x_mask, np.full(x_cov.shape, 0), x_cov)
    y_cov = np.where(y_mask, np.full(y_cov.shape, 0), y_cov)

    x_cov, x_binary, y_cov, y_binary = (arr.T for arr in (x_cov, data['x_binary'], y_cov, data['y_binary']))

    print(f"Number of x_conditional points: {num_x_conditional}")
    print(f"Number of y_conditional points: {num_y_conditional}")

    return x_cov, x_binary, y_cov, y_binary, x_cov.shape[1]


def prepare_dataset(x_cov, y_cov, y_binary, sample_length):
    # Convert NumPy arrays directly to PyTorch tensors only if they are not already tensors
    if isinstance(x_cov, np.ndarray):
        inputs_tensor = torch.from_numpy(x_cov.reshape(-1, 2, sample_length)).float()
    else:
        inputs_tensor = x_cov.float()  # Assuming x_cov is already a tensor

    reg_tensor = torch.from_numpy(y_cov.reshape(-1, 1, sample_length)).float()
    cls_tensor = torch.from_numpy(y_binary.reshape(-1, 1, sample_length)).float()

    return TensorDataset(inputs_tensor, reg_tensor, cls_tensor)


def prepare_dataloaders(cell_line, downsample_rate, batch_size):
    x_cov, x_binary, y_cov, y_binary, sample_length = load_data(cell_line, downsample_rate)
    x_train = np.concatenate((np.expand_dims(x_cov, axis=1), np.expand_dims(x_binary, axis=1)), axis=1)

    train_inputs, val_inputs, train_reg_labels, val_reg_labels, train_cls_labels, val_cls_labels = train_test_split(
        x_train, y_cov, y_binary, test_size=0.2, random_state=42
    )

    train_dataset = prepare_dataset(train_inputs, train_reg_labels, train_cls_labels, sample_length)
    val_dataset = prepare_dataset(val_inputs, val_reg_labels, val_cls_labels, sample_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size // 2, shuffle=False)

    return train_loader, val_loader, sample_length


# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Positional Encoding Module
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.size(1) > self.max_len:
            raise ValueError(f"Sequence length {x.size(1)} exceeds max_len {self.max_len}")
        x = x + self.pe[:, :x.size(1), :]
        return x


# New Decoder Block with Residual Connection
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Residual connection
        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out + residual

# Modified Model with Transformer Encoder and CNN Decoder
class MultiTaskModel(nn.Module):
    def __init__(self, sample_length, d_model=64, nhead=4, num_layers=3,
                 dim_feedforward=256, dropout=0.1):
        super(MultiTaskModel, self).__init__()
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(2, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=sample_length)

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)

        # Enhanced CNN Decoder with Residual Blocks
        self.decoder = nn.Sequential(
            DecoderBlock(d_model, 256, kernel_size=5, padding=2, dropout=0.25),
            DecoderBlock(256, 128, kernel_size=5, padding=2, dropout=0.2),
            DecoderBlock(128, 64, kernel_size=5, padding=2),
            nn.Conv1d(64, 1, kernel_size=5, padding=2)
        )

        # Additional components
        self.sigmoid = nn.Sigmoid()
        self.final_activation = nn.ReLU()  # For regression output

    def forward(self, x):
        # Input shape: (B, 2, L)
        B, C, L = x.size()

        # Transformer processing
        x = x.permute(0, 2, 1)  # (B, L, 2)
        x = self.input_proj(x)  # (B, L, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)  # (B, d_model, L)

        # CNN Decoder with residuals
        x = self.decoder(x)  # (B, 1, L)

        # Output processing
        reg = self.final_activation(x)  # ReLU for positive outputs
        cls = self.sigmoid(x)
        return reg, cls


# The rest of the code remains mostly the same except for validation metrics

if __name__ == "__main__":
    downsamples = ['0.1dn']
    train_cell_line = 'SYSY'
    for downsample in downsamples:
        print(f"Running model for downsampling rate: {downsample}")
        train_loader, val_loader, sample_length = prepare_dataloaders(train_cell_line, downsample, 2)
        model = MultiTaskModel(sample_length).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.1)
        reg_loss_fn = nn.MSELoss()
        cls_loss_fn = nn.BCELoss()

        num_epochs = 50
        for epoch in range(num_epochs):
            start_time_epoch = time.time()  # Start time for the epoch
            model.train()
            all_reg_outputs, all_reg_labels = [], []
            all_cls_outputs, all_cls_labels = [], []

            for inputs, reg_targets, cls_targets in train_loader:
                inputs, reg_targets, cls_targets = inputs.to(device), reg_targets.to(device), cls_targets.to(device)

                inputs = inputs.float()
                reg_targets = reg_targets.float()
                cls_targets = cls_targets.float()

                if reg_targets.size(0) != train_loader.batch_size:
                    continue  # Skip this batch

                optimizer.zero_grad()

                # Mask where the first channel is zero
                mask_padding = (inputs[:, 0, :] != 0)
                mask_padding_inputs = mask_padding.unsqueeze(1).expand_as(inputs)
                mask_padding_reg = mask_padding.unsqueeze(1).expand_as(reg_targets)

                # Define the threshold
                threshold = 1e-10

                # Modify the arrays
                label_zeros = (inputs[:, 0, :] == threshold)

                # Reshape and expand the mask to match the shape of inputs and reg_targets
                label_zeros_inputs = label_zeros.unsqueeze(1).expand_as(inputs)
                label_zeros_reg = label_zeros.unsqueeze(1).expand_as(reg_targets)

                inputs[label_zeros_inputs] = 0
                # reg_targets[label_zeros_reg] = 0

                reg_output, cls_output = model(inputs)

                reg_loss = reg_loss_fn(reg_output[mask_padding_reg], reg_targets[mask_padding_reg])
                cls_loss = cls_loss_fn(cls_output[mask_padding.unsqueeze(1)], cls_targets[mask_padding.unsqueeze(1)])
                total_loss = reg_loss + cls_loss
                total_loss.backward()
                optimizer.step()


            print(f"\nEpoch {epoch + 1}")
            print(f"Training Loss: {total_loss.item() / len(val_loader)}")
            print(f"Epoch {epoch + 1} Training Completed")


            # Validation phase
            model.eval()
            all_reg_outputs, all_reg_labels = [], []
            all_cls_outputs, all_cls_labels = [], []
            with torch.no_grad():
                for inputs, reg_labels, cls_labels in val_loader:
                    inputs = inputs.to(device)
                    reg_targets = reg_labels.to(device)
                    cls_targets = cls_labels.to(device)

                    inputs = inputs.float()
                    reg_labels = reg_labels.float()
                    cls_labels = cls_labels.float()

                    if reg_labels.size(0) != val_loader.batch_size:
                        continue  # Skip this batch

                    # Mask where the first channel is zero
                    mask_padding = (inputs[:, 0, :] != 0)
                    mask_padding_inputs = mask_padding.unsqueeze(1).expand_as(reg_labels)
                    mask_padding_reg = mask_padding.unsqueeze(1).expand_as(reg_labels)

                    # Define the threshold
                    threshold = 1e-10

                    # Modify the arrays
                    label_zeros = (inputs[:, 0, :] == threshold)
                    # num_label_zeros = torch.sum(label_zeros).item()
                    # print(f"Number of label_zeros: {num_label_zeros}")

                    # Reshape and expand the mask to match the shape of inputs and reg_targets
                    label_zeros_inputs = label_zeros.unsqueeze(1).expand_as(inputs)
                    label_zeros_reg = label_zeros.unsqueeze(1).expand_as(reg_labels)

                    inputs[label_zeros_inputs] = 0

                    reg_output, cls_output = model(inputs)

                    mask_padding = mask_padding.cpu()
                    all_reg_outputs.append(reg_output[mask_padding.unsqueeze(1)].cpu().numpy().flatten())
                    all_reg_labels.append(reg_labels[mask_padding.unsqueeze(1)].cpu().numpy().flatten())
                    all_cls_outputs.append(cls_output[mask_padding.unsqueeze(1)].cpu().numpy().flatten())
                    all_cls_labels.append(cls_labels[mask_padding.unsqueeze(1)].cpu().numpy().flatten())

                # Concatenate all outputs and labels outside the loop
                all_reg_outputs = np.concatenate(all_reg_outputs)
                all_reg_labels = np.concatenate(all_reg_labels)
                all_cls_outputs = np.concatenate(all_cls_outputs)
                all_cls_labels = np.concatenate(all_cls_labels)

                # Check for NaN and inf values in all_reg_outputs and all_reg_labels
                nan_in_outputs = np.isnan(all_reg_outputs)
                inf_in_outputs = np.isinf(all_reg_outputs)
                nan_in_labels = np.isnan(all_reg_labels)
                inf_in_labels = np.isinf(all_reg_labels)

                # Handle NaN and inf values (Option 1: Replace with zeros)
                all_reg_outputs[nan_in_outputs | inf_in_outputs] = 0
                all_reg_labels[nan_in_labels | inf_in_labels] = 0


            # Calculate metrics
            reg_pearson_corr, _ = pearsonr(all_reg_outputs, all_reg_labels)
            mi = compute_mi(all_reg_outputs, all_reg_labels, bins=20)
            cls_auc = roc_auc_score(all_cls_labels, all_cls_outputs)
            precision, recall, _ = precision_recall_curve(all_cls_labels, all_cls_outputs)
            cls_auprc = auc(recall, precision)
            kl_div = compute_kl_div(all_cls_outputs, all_cls_labels)

            print(f"Validation - Regression Pearson: {reg_pearson_corr}, MI: {mi}")
            print(f"Classification AUC: {cls_auc}, AUPRC: {cls_auprc}, KL Div: {kl_div}")
            
        # Save the trained model
        model_save_path = f'autodl-tmp/deeptrans_modelres_{downsample}.pth'
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")