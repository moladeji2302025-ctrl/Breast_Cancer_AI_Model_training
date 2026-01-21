# Breast Cancer ANN Architectures Experiment

This repository contains an implementation of an experiment to evaluate the effectiveness of various Artificial Neural Network (ANN) architectures for breast cancer prediction.

## Overview

The experiment compares four different ANN architectures:
- **Baseline Architecture (No Hidden Layer):** Equivalent to logistic regression
- **One Hidden Layer:** Single dense layer with 64 neurons
- **Two Hidden Layers:** Two dense layers (128, 64 neurons)
- **Three Hidden Layers:** Three dense layers (256, 128, 64 neurons)

## Dataset

The experiment uses the [Breast Cancer Wisconsin Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) from scikit-learn, which contains:
- 569 samples
- 30 features (computed from digitized images of breast mass)
- 2 classes: malignant (0) and benign (1)

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Dependencies:
- TensorFlow >= 2.12.0
- NumPy >= 1.23.0
- scikit-learn >= 1.2.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

## Usage

Run the main experiment:

```bash
python breast_cancer_ann_experiment.py
```

This will:
1. Run the default experiment with an 80/20 train/test split
2. Run additional experiments with varying test splits (10%, 20%, 30%, 40%)
3. Display evaluation metrics for each architecture
4. Generate confusion matrices for each configuration

## Features

### Data Preprocessing
- Automatic train/test split with configurable ratios
- Feature standardization using StandardScaler
- Stratified sampling to maintain class distribution

### Model Architectures
All models use:
- ReLU activation for hidden layers
- Sigmoid activation for output layer
- Adam optimizer
- Binary cross-entropy loss

### Evaluation Metrics
Each model is evaluated on:
- **Accuracy (ACC):** Overall correctness
- **Precision (Prec):** True positives / (True positives + False positives)
- **Recall (Rec):** True positives / (True positives + False negatives)
- **F1 Score:** Harmonic mean of precision and recall

### Visualization
- Confusion matrices for each architecture and test split
- Saved as PNG files for further analysis

## Code Structure

The implementation includes reusable functions for:
- `preprocess_data()` - Data loading and preprocessing
- `create_baseline_model()` - Baseline architecture creation
- `create_one_hidden_layer_model()` - Single hidden layer model
- `create_two_hidden_layers_model()` - Two hidden layers model
- `create_three_hidden_layers_model()` - Three hidden layers model
- `evaluate_model()` - Metric calculation
- `plot_confusion_matrix()` - Confusion matrix visualization
- `train_and_evaluate_architecture()` - Complete training and evaluation pipeline
- `run_experiment()` - Main experiment runner
- `run_experiments_with_varying_splits()` - Multiple experiments with different splits

## Output

The script generates:
1. Console output with detailed metrics for each architecture
2. Confusion matrix plots saved as PNG files
3. Comparison tables across different test splits

## Example Output

```
SUMMARY OF RESULTS
================================================================
Architecture                                  ACC      Prec     Rec      F1      
--------------------------------------------------------------------------------
Baseline (No Hidden Layer)                   0.9561   0.9577   0.9714   0.9645
One Hidden Layer (64 units)                  0.9649   0.9714   0.9714   0.9714
Two Hidden Layers (128, 64 units)            0.9737   0.9857   0.9714   0.9785
Three Hidden Layers (256, 128, 64 units)     0.9737   0.9857   0.9714   0.9785
================================================================
```

## License

This project is open source and available for educational purposes.