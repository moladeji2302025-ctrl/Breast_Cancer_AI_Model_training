# Usage Examples

This document provides examples of how to use the breast cancer ANN experiment code.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Quick Test

Run a quick test with reduced epochs to verify installation:

```bash
python test_experiment.py
```

### 3. Run Full Experiment

Run the complete experiment with all architectures and varying test splits:

```bash
python breast_cancer_ann_experiment.py
```

## Using Individual Functions

You can also import and use individual functions in your own scripts:

```python
import numpy as np
import tensorflow as tf
from breast_cancer_ann_experiment import (
    preprocess_data,
    create_baseline_model,
    create_one_hidden_layer_model,
    create_two_hidden_layers_model,
    create_three_hidden_layers_model,
    evaluate_model,
    plot_confusion_matrix
)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Preprocess the data
X_train, X_test, y_train, y_test = preprocess_data(test_size=0.2)

# Create and train a model
model = create_one_hidden_layer_model(input_dim=X_train.shape[1], hidden_units=64)
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
metrics, y_pred = evaluate_model(model, X_test, y_test)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")

# Plot confusion matrix
plot_confusion_matrix(
    metrics['confusion_matrix'],
    "My Custom Model",
    save_path="my_confusion_matrix.png",
    show_plot=True  # Set to True to display the plot
)
```

## Customizing Experiments

### Run with Different Test Splits

```python
from breast_cancer_ann_experiment import run_experiment

# Run experiment with 30% test data
results = run_experiment(test_size=0.3, epochs=100, batch_size=32)
```

### Create Custom Architecture

```python
from tensorflow import keras
from tensorflow.keras import layers

def create_custom_model(input_dim):
    model = keras.Sequential([
        layers.Dense(512, activation='relu', input_dim=input_dim),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Use it with the existing pipeline
from breast_cancer_ann_experiment import preprocess_data, train_and_evaluate_architecture

X_train, X_test, y_train, y_test = preprocess_data(test_size=0.2)
model = create_custom_model(X_train.shape[1])

metrics, history = train_and_evaluate_architecture(
    "Custom Deep Model with Dropout",
    model,
    X_train, X_test, y_train, y_test,
    epochs=100,
    batch_size=32,
    verbose=1
)
```

## Understanding the Output

### Console Output

The experiment prints:
1. **Configuration** - Test size, epochs, batch size
2. **Training Progress** - Results for each architecture
3. **Summary Table** - Comparison of all models
4. **Comparison** - Results across different test splits

### Generated Files

The experiment generates PNG files for confusion matrices:
- `confusion_matrix_baseline_no_hidden_layer_test_0.2.png`
- `confusion_matrix_one_hidden_layer_64_units_test_0.2.png`
- `confusion_matrix_two_hidden_layers_128_64_units_test_0.2.png`
- `confusion_matrix_three_hidden_layers_256_128_64_units_test_0.2.png`

These files are saved in the current directory and can be used for analysis and reporting.

## Expected Results

With the default configuration, you should see:
- **Baseline Model**: ~93-95% accuracy
- **One Hidden Layer**: ~95-97% accuracy
- **Two Hidden Layers**: ~96-98% accuracy
- **Three Hidden Layers**: ~96-98% accuracy

The models typically show high precision and recall for breast cancer detection.

## Troubleshooting

### Low Accuracy

If you're getting unusually low accuracy:
1. Check that you're using enough epochs (default is 100)
2. Ensure data preprocessing is working correctly
3. Verify the random seed is set for reproducibility

### Memory Issues

If you run out of memory:
1. Reduce the batch size
2. Reduce the number of neurons in hidden layers
3. Run fewer experiments at once

### Plot Not Showing

By default, plots are saved but not displayed. To show plots:
```python
plot_confusion_matrix(cm, title, save_path="file.png", show_plot=True)
```
