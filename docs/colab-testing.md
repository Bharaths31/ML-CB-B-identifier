# Google Colab Testing & Evaluation

For automated large-scale evaluation of exported models on Google Colab, the project provides a dedicated testing notebook.

**Notebook location:** `colab/cattle_buffalo_tester.ipynb`

## Features

This notebook allows you to:

1. **Test Exported Models**: Load your exported ONNX model (`lite2_fp32.onnx`).
2. **Single & Batch Inference**: Run inference on single images or entire zipped batches.
3. **HTML Reports**: Generate comprehensive HTML evaluation reports containing confusion matrices, per-breed accuracy tables, and misclassified examples.
4. **Large-Scale Kaggle Evaluation**: Automatically download the complete Kaggle dataset (`algsoch/breed-cattle-buffalo`) directly to Colab and run inference on all images, yielding full dataset metrics and per-breed accuracy tables.

## Usage

### Option 1: Uploading local test split

The repository includes a helper script `create_test_eval_zip.py` that you can run locally to bundle your generated test split images into a zip file (`test_eval_images.zip`).
Upload this file to your Colab session for batch evaluation.

### Option 2: Full Dataset Evaluation via Kaggle

Using the **§9 — Large-Scale Kaggle Dataset Evaluation** section in the notebook, you can supply your Kaggle API credentials. The notebook will download the entire dataset and map breeds to your model's class maps automatically, providing a full-scale performance report.
