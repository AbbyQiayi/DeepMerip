
# DeepMerip: A Convolutional Deep Learning Approach for Denoising Exome-Wide MeRIP-Seq Data Due to Low Sequencing Depth

## Introduction
**DeepMerip** is a computational tool designed to enhance exome-wide MeRIP-Seq data quality by addressing the challenges posed by low sequencing depth. The tool uses a convolutional neural network (CNN) with residual connection to denoise input coverage data, improving the identification of m6A-modified sites and reducing biases associated with technical artifacts such as low read coverage.

MeRIP-Seq (Methylated RNA Immunoprecipitation sequencing) is a critical method for detecting post-transcriptional RNA modifications, such as N6-methyladenosine (m6A). However, due to experimental limitations like low sequencing depth, raw MeRIP-Seq data often contain noise that hampers accurate peak calling and downstream analysis. DeepMerip leverages deep learning techniques to overcome these limitations, providing a robust pipeline for denoising and peak calling.

## Features
- **Convolutional Neural Networks (CNNs) with residual connection** for denoising sequencing data.
- **Support for low-coverage sequencing datasets** with improved sensitivity for m6A site identification.
- **Integration with exomePeak2** for downstream peak calling analysis.
- **Easy-to-use command line interface** for seamless data input and denoising workflows.
- **Support for both coverage and binary classification outputs.**

## Installation
To use DeepMerip, ensure you have the following dependencies installed:

### Requirements
- Python 3.x
- PyTorch (deep learning backend)
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Other specific libraries used in your pipeline, such as h5py for .npz file handling

## Usage

### Data Preparation
Before running DeepMerip, ensure that your data is preprocessed correctly. The tool expects input coverage data and binary peak calling results in `.npz` format. These files typically include the following arrays:
- **IP_cov**: Immunoprecipitation coverage data.
- **input_cov**: Input coverage data (control).
- **peakCalling_result**: Binary classification indicating peak locations.

Use the provided data preparation scripts in `data_prep.py` and `data_format_trans.py` to convert your raw data into the correct format.

### Denoising with DeepMerip
Once your data is prepared, you can run the deep learning model for denoising by Deep_merip.py

### Model Testing
You can evaluate the performance of the denoising model using test datasets by test_model_DeepMerip.py


For additional performance analysis, you can evaluate the transformer-based model test_model_transformer.py:

## File Descriptions
- `data_format_trans.py`: Script for transforming genomic position data for peak calling.
- `data_prep.py`: Script for preparing and preprocessing MeRIP-Seq data into the required format for DeepMerip.
- `Deep_merip.py`: Main script for running the convolutional neural network model.
- `test_model_DeepMerip.py`: Script for testing the CNN-based DeepMerip model.
- `test_model_transformer.py`: Script for testing a transformer-based model as an alternative to the CNN.
- `transformer.py`: Implementation of a transformer model for potential performance comparison.


