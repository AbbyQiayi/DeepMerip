import numpy as np
import pandas as pd
import os


# Function to process .npz data from input files and save the results
def process_npz_data(path_to_data_dir, save_npz_path, cell_line, down_sample_exten):
    # Define file extensions and paths for saving and loading
    exten = down_sample_exten
    file_name = f"peak_{cell_line}_IP_vs_INPUT_{exten}.npz"
    save_file_path = os.path.join(save_npz_path, file_name)
    load_file_path = os.path.join(path_to_data_dir, "peak_" + cell_line + "_IP_vs_INPUT_" + exten + '.csv')

    # Check if the .npz file already exists
    if os.path.exists(save_file_path):
        print("Loaded .npz data:", {save_file_path})

    else:
        print(f"Extracting patterns from {cell_line} with {down_sample_exten}")

        # Load the data from the CSV file
        data = pd.read_csv(load_file_path, low_memory=False)
        df = pd.DataFrame(data)

        # Avoid log of zero by adding a small value to prevent log(0)
        epsilon_IP = 1
        epsilon_INPUT = 1
        df['log_diff'] = np.log2(df['IP_cov'] + epsilon_IP) - np.log2(df['input_cov'] + epsilon_INPUT)

        # Add a small number to 'log_diff' where its value is 0 to avoid issues
        small_number = 1e-10
        df.loc[df['log_diff'] == 0, 'log_diff'] += small_number

        # Group data by Gene index
        grouped = df.groupby('Gene_indx')

        # Extract matrices (n x 3) containing 'log_diff', 'peakCalling_result', and 'pvalue'
        data_to_save = {}
        for gene_indx, group in grouped:
            matrices = group[['log_diff', 'peakCalling_result', 'pvalue']].values
            data_to_save[str(gene_indx)] = matrices  # n*3 matrix

        # Save the grouped data into an .npz file
        np.savez(save_file_path, **data_to_save)
        print(f"Saved {cell_line} {down_sample_exten}.npz data")

    return save_file_path


# Function to pad matrices for each gene index
def pad_matrix(mat, max_len):
    pad_len = max_len - mat.shape[0]
    pad_before = pad_len // 2  # Use integer division to calculate padding
    pad_after = pad_len - pad_before  # Ensure the total padding equals pad_len

    return np.pad(mat, ((pad_before, pad_after), (0, 0)), 'constant')


# Function to pad and reshape data for use in model input
def pad_and_reshape(data, binary):
    # Find the maximum matrix length to pad all matrices to the same length
    max_len = max([data[key].shape[0] for key in data.files])

    # List of X (input features) extracted from each gene's data
    X_list = [pad_matrix(data[key], max_len)[:, 0] for key in data.files]

    # Define which column in the matrix to use as Y (target variable)
    if binary:
        y_column_index = 1  # Binary classification uses peakCalling_result
    else:
        y_column_index = 2  # Use pvalue for regression

    # List of Y (target values) extracted from each gene's data
    Y_list = [pad_matrix(data[key], max_len)[:, y_column_index] for key in data.files]

    # Stack the matrices into arrays with consistent shapes for X and Y
    X = np.stack(X_list, axis=-1)
    Y = np.stack(Y_list, axis=-1)

    return X, Y


# Function to process data for coverage and binary classification
def prep(path_to_data_dir, save_data_path, cell_line, downsample_exten):
    # Process and load the down-sampled and original .npz data
    down_sample_data_path = process_npz_data(path_to_data_dir, save_data_path, cell_line, downsample_exten)
    original_data_path = process_npz_data(path_to_data_dir, save_data_path, cell_line, 'None')

    # Load the data from the .npz files
    down_sample_data = np.load(down_sample_data_path)
    original_data = np.load(original_data_path)

    # Prepare X and Y for coverage-based regression (log_diff and p-value)
    x_cov, _ = pad_and_reshape(down_sample_data, binary=False)
    y_cov, _ = pad_and_reshape(original_data, binary=False)

    # Prepare X and Y for binary classification (log_diff and peakCalling_result)
    x_binary, y_binary = pad_and_reshape(original_data, binary=True)

    # Save the prepared data into an .npz file
    output_path = os.path.join(save_data_path, f"cov_regression_cls_{downsample_exten}_{cell_line}.npz")
    np.savez(output_path, x_cov=x_cov, x_binary=x_binary, y_cov=y_cov, y_binary=y_binary)
    print(f"Saved x_cov and y_cov to {output_path}")


# Main function to run the process on different cell lines and downsample extensions
def main():
    path_to_data_dir = "../data/exomePeak2_out/HEK293"
    save_data_path = "/../data/dataset"

    # Define cell lines and extensions to process
    cell_lines = ['HEK293T']
    extensions = ['0.04dn', '0.16dn', '0.23dn']

    # Loop through each cell line and extension
    for cell_line in cell_lines:
        for exten in extensions:
            print(f"Processing for cell line: {cell_line}, down_sample: {exten}")
            # Process data for both coverage and binary classification
            prep(path_to_data_dir, save_data_path, cell_line, exten)


if __name__ == "__main__":
    main()
