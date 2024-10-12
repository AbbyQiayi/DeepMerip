import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def load_data(cell_line, downsample_rate):
    file_path = f"../data/dataset/Data_prep_{downsample_rate}_{cell_line}.npz"
    data = np.load(file_path)

    x_cov = np.log2(data['x_IP'] + 1) - np.log2(data['x_Input'] + 1)
    y_cov = np.log2(data['y_IP'] + 1) - np.log2(data['y_Input'] + 1)

    # 设定阈值
    threshold = 1e-10

    # 为 x_IP 和 x_Input 等于阈值的情况创建掩码
    x_mask = (data['x_IP'] == threshold) & (data['x_Input'] == threshold)
    y_mask = (data['y_IP'] == threshold) & (data['y_Input'] == threshold)

    # 统计符合条件的位点数量
    num_x_conditional = np.sum(x_mask)
    num_y_conditional = np.sum(y_mask)

    # 根据掩码设置特定位点的 x_cov 和 y_cov 值
    x_cov = np.where(x_mask, np.full(x_cov.shape, 0), x_cov)
    y_cov = np.where(y_mask, np.full(y_cov.shape, 0), y_cov)

    # 转置处理以符合后续处理需求
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


class MultiTaskModel(nn.Module):
    def __init__(self, sample_length):
        super(MultiTaskModel, self).__init__()
        self.shared_conv1 = nn.Conv1d(in_channels=2, out_channels=256, kernel_size=5, padding=2)
        self.shared_conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, padding=2)
        self.shared_conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, padding=2)
        self.shared_conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding=2)
        self.shared_conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.shared_conv6 = nn.Conv1d(in_channels=256, out_channels=64, kernel_size=7, padding=3)
        self.shared_conv7 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=5, padding=2)

        # Residual adjustment layers to ensure channel matching
        self.res_adjust1 = nn.Conv1d(in_channels=2, out_channels=256, kernel_size=1)  # After shared_conv2
        self.res_adjust2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1)  # After shared_conv4

        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Shared layers
        identity = x

        # Layer 1
        x = self.shared_conv1(x)
        x = self.drop1(x)
        x = self.relu(x)
        x = self.bn1(x)

        # Layer 2
        x = self.shared_conv2(x)
        x = self.drop1(x)
        x = self.relu(x)
        x = self.bn1(x)

        # Adding residual connection from identity after layer 2
        x += self.res_adjust1(identity)  # Adjust identity to have 256 channels

        identity = x  # Updating identity to the output after the first two layers

        # Layer 3
        x = self.shared_conv3(x)
        x = self.drop2(x)
        x = self.relu(x)
        x = self.bn2(x)

        # Layer 4
        x = self.shared_conv4(x)
        x = self.drop2(x)
        x = self.relu(x)
        x = self.bn2(x)

        # Adding residual connection from identity after layer 4
        x += self.res_adjust2(identity)  # Adjust identity to have 128 channels

        # Layer 5
        x = self.shared_conv5(x)
        x = self.drop2(x)
        x = self.relu(x)
        x = self.bn1(x)

        # Layer 6
        x = self.shared_conv6(x)
        x = self.drop2(x)
        x = self.relu(x)
        x = self.bn3(x)

        # Layer 7
        x = self.shared_conv7(x)

        # Regression head
        reg = x

        # Classification head
        cls = self.sigmoid(x)

        return reg, cls


if __name__ == "__main__":
    # Model, optimizer, and loss functions

    downsamples = ['0.1dn']
    train_cell_line = 'NEB'
    for downsample in downsamples:
        print(f"Running model for downsampling rate: {downsample}")
        # Initialize the DataLoader for the current downsampling rate
        train_loader, val_loader,sample_length = prepare_dataloaders(train_cell_line, downsample,64)

        # Initialize the model
        model = MultiTaskModel(sample_length)
        model = model.to(device)

        # Initialize optimizer and loss functions
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.1)
        reg_loss_fn = nn.MSELoss()
        cls_loss_fn = nn.BCELoss()

        # Training and validation loop
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
            cls_auc = roc_auc_score(all_cls_labels, all_cls_outputs)
            precision, recall, _ = precision_recall_curve(all_cls_labels, all_cls_outputs)
            cls_auprc = auc(recall, precision)

            print(f"Validation - Regression Pearson Correlation:{reg_pearson_corr}, "
                f"Validation - Classification AUC: {cls_auc}, Classification AUPRC: {cls_auprc}")
            # At the end of the epoch, calculate the time taken
            end_time_epoch = time.time()
            print(f"Epoch {epoch + 1} completed in {end_time_epoch - start_time_epoch:.2f} seconds")
        # After training, save the model's state dictionary
        model_save_path = f'../model/resnet_model_{downsample}.pth'
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")





