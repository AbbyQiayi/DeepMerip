import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# Function to process data, convert it to .npz format and save
def process_npz_data(path_to_data_dir, save_npz_path, cell_line, down_sample_exten):

    # Define the file name based on the cell line and down-sampling extension
    exten = down_sample_exten
    file_name = f"peak_{cell_line}_IP_vs_INPUT_{exten}.npz"
    save_file_path = os.path.join(save_npz_path, file_name)  # Full path to save the npz file
    load_file_path = os.path.join(path_to_data_dir, "peak_" + cell_line + "_IP_vs_INPUT_" + exten + '.csv')  # Full path to load CSV

    # Check if the .npz file already exists to avoid reprocessing
    if os.path.exists(save_file_path):
        print("Loaded .npz data:", {save_file_path})

    else:
        # If the file does not exist, read the CSV and process the data
        print(f"Extracting patterns from {cell_line} with {down_sample_exten} down-sample")

        # Read CSV into a pandas DataFrame
        data = pd.read_csv(load_file_path, low_memory=False)
        df = pd.DataFrame(data)

        # Define a small number to add to zero values for stability
        small_number = 1e-10

        # Add the small number where both 'IP_cov' and 'input_cov' are zero to avoid issues
        df.loc[(df['IP_cov'] == 0) & (df['input_cov'] == 0), ['IP_cov', 'input_cov']] += small_number
        # Set the 'peakCalling_result' to 0 where both 'IP_cov' and 'input_cov' are zero
        df.loc[(df['IP_cov'] == 0) & (df['input_cov'] == 0), 'peakCalling_result'] = 0

        # Group the data by 'Gene_indx'
        grouped = df.groupby('Gene_indx')

        # Create a dictionary to hold n x 2 matrices for each gene index
        data_to_save = {}
        for gene_indx, group in grouped:
            # Extract relevant columns ('IP_cov', 'input_cov', 'peakCalling_result') as a numpy array
            matrices = group[['IP_cov', 'input_cov', 'peakCalling_result']].values
            data_to_save[str(gene_indx)] = matrices  # Store the matrix by gene index

        # Save the dictionary as a .npz file
        np.savez(save_file_path, **data_to_save)
        print(f"Saved {cell_line} {down_sample_exten}.npz data")

    return save_file_path

# Function to pad the matrices to the same length
def pad_matrix(mat, max_len):
    # Calculate padding length based on the max length of the matrices
    pad_len = max_len - mat.shape[0]
    # Ensure padding is evenly distributed before and after
    pad_before = pad_len // 2  # Integer division
    pad_after = pad_len - pad_before  # Remaining padding

    # Pad the matrix with zeros and return
    return np.pad(mat, ((pad_before, pad_after), (0, 0)), 'constant')

# Function to pad and reshape data from .npz files for model input
def pad_and_reshape(data):
    # Find the maximum length of matrices across all genes
    max_len = max([data[key].shape[0] for key in data.files])

    # Pad and reshape data for IP, input, and binary labels
    IP = [pad_matrix(data[key], max_len)[:, 0] for key in data.files]  # IP coverage
    Input = [pad_matrix(data[key], max_len)[:, 1] for key in data.files]  # Input coverage
    binary = [pad_matrix(data[key], max_len)[:, 2] for key in data.files]  # Binary classification (peakCalling_result)

    # Stack the padded matrices into numpy arrays
    IP = np.stack(IP, axis=-1)
    Input = np.stack(Input, axis=-1)
    binary = np.stack(binary, axis=-1)
    return IP, Input, binary

# Function to prepare and save the dataset
def prep(path_to_data_dir, save_data_path, cell_line, downsample_exten):

    # Process and load down-sampled data and original data
    down_sample_data_path = process_npz_data(path_to_data_dir, save_data_path, cell_line, downsample_exten)
    original_data_path = process_npz_data(path_to_data_dir, save_data_path, cell_line, 'None')
    down_sample_data = np.load(down_sample_data_path)
    original_data = np.load(original_data_path)

    # Pad and reshape the down-sampled data and original data
    x_IP, x_Input, x_binary = pad_and_reshape(down_sample_data)  # Down-sampled data
    y_IP, y_Input, y_binary = pad_and_reshape(original_data)  # Original data

    # Save the prepared data into a new .npz file
    output_path = os.path.join(save_data_path, f"Data_prep_{downsample_exten}_{cell_line}.npz")
    np.savez(output_path, x_IP=x_IP, x_Input=x_Input, x_binary=x_binary, y_IP=y_IP, y_Input=y_Input, y_binary=y_binary)
    print(f"Saved x_cov and y_cov to {output_path}")

# Main function to call the preparation process for different cell lines and extensions
def main():
    path_to_data_dir = "../data/exomePeak2_out"  # Path to the raw data
    save_data_path = "../data/dataset"  # Path to save processed datasets

    # Define the cell lines and down-sampling extensions to process
    cell_lines = ['mm'] # change to your own cell line
    extensions = ['None', '.1', '.2', '.3', '.01', '.02', '.03', '.05'] # change to your own downsample rate

    # Iterate through each cell line and down-sampling extension
    for cell_line in cell_lines:
        for exten in extensions:
            print(f"Processing for cell line: {cell_line}, down_sample: {exten}")
            # Prepare data for the current cell line and down-sampling extension
            prep(path_to_data_dir, save_data_path, cell_line, exten)

# Call the main function if the script is executed
if __name__ == "__main__":
    main()
