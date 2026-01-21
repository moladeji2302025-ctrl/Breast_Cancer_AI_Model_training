"""
Breast Cancer ANN Architectures Experiment

This script implements and evaluates various ANN architectures for breast cancer prediction
using the sklearn breast cancer dataset.

Architectures:
- Baseline: No hidden layers (logistic regression equivalent)
- One Hidden Layer: Single dense layer
- Two Hidden Layers: Two dense layers with varying neurons
- Three Hidden Layers: Three dense layers with varying dimensions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')


def preprocess_data(test_size=0.2, random_state=42):
    """
    Preprocess the breast cancer dataset.
    
    Args:
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Load the breast cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


def create_baseline_model(input_dim):
    """
    Create a baseline ANN model with no hidden layers (logistic regression equivalent).
    
    Args:
        input_dim (int): Number of input features
        
    Returns:
        keras.Model: Compiled baseline model
    """
    model = keras.Sequential([
        layers.Dense(1, activation='sigmoid', input_dim=input_dim)
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_one_hidden_layer_model(input_dim, hidden_units=64):
    """
    Create an ANN model with one hidden layer.
    
    Args:
        input_dim (int): Number of input features
        hidden_units (int): Number of neurons in the hidden layer
        
    Returns:
        keras.Model: Compiled model with one hidden layer
    """
    model = keras.Sequential([
        layers.Dense(hidden_units, activation='relu', input_dim=input_dim),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_two_hidden_layers_model(input_dim, hidden_units_1=128, hidden_units_2=64):
    """
    Create an ANN model with two hidden layers.
    
    Args:
        input_dim (int): Number of input features
        hidden_units_1 (int): Number of neurons in the first hidden layer
        hidden_units_2 (int): Number of neurons in the second hidden layer
        
    Returns:
        keras.Model: Compiled model with two hidden layers
    """
    model = keras.Sequential([
        layers.Dense(hidden_units_1, activation='relu', input_dim=input_dim),
        layers.Dense(hidden_units_2, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_three_hidden_layers_model(input_dim, hidden_units_1=256, hidden_units_2=128, hidden_units_3=64):
    """
    Create an ANN model with three hidden layers.
    
    Args:
        input_dim (int): Number of input features
        hidden_units_1 (int): Number of neurons in the first hidden layer
        hidden_units_2 (int): Number of neurons in the second hidden layer
        hidden_units_3 (int): Number of neurons in the third hidden layer
        
    Returns:
        keras.Model: Compiled model with three hidden layers
    """
    model = keras.Sequential([
        layers.Dense(hidden_units_1, activation='relu', input_dim=input_dim),
        layers.Dense(hidden_units_2, activation='relu'),
        layers.Dense(hidden_units_3, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model and calculate metrics.
    
    Args:
        model (keras.Model): Trained model
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return metrics, y_pred


def plot_confusion_matrix(cm, title, save_path=None):
    """
    Plot a confusion matrix.
    
    Args:
        cm (np.ndarray): Confusion matrix
        title (str): Title for the plot
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Malignant', 'Benign'],
                yticklabels=['Malignant', 'Benign'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def train_and_evaluate_architecture(model_name, model, X_train, X_test, y_train, y_test, 
                                    epochs=100, batch_size=32, verbose=0):
    """
    Train and evaluate a specific architecture.
    
    Args:
        model_name (str): Name of the architecture
        model (keras.Model): Model to train
        X_train, X_test, y_train, y_test: Training and testing data
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        verbose (int): Verbosity level
        
    Returns:
        tuple: (metrics, history)
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=verbose
    )
    
    # Evaluate the model
    metrics, y_pred = evaluate_model(model, X_test, y_test)
    
    # Print metrics
    print(f"\nResults for {model_name}:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    
    return metrics, history


def run_experiment(test_size=0.2, epochs=100, batch_size=32):
    """
    Run the complete experiment with all architectures.
    
    Args:
        test_size (float): Proportion of the dataset to include in the test split
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
    """
    print("\n" + "="*60)
    print("BREAST CANCER ANN ARCHITECTURES EXPERIMENT")
    print("="*60)
    print(f"\nExperiment Configuration:")
    print(f"  Test Size: {test_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(test_size=test_size)
    input_dim = X_train.shape[1]
    
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Testing samples: {X_test.shape[0]}")
    print(f"  Number of features: {input_dim}")
    
    # Define architectures
    architectures = {
        'Baseline (No Hidden Layer)': create_baseline_model(input_dim),
        'One Hidden Layer (64 units)': create_one_hidden_layer_model(input_dim, 64),
        'Two Hidden Layers (128, 64 units)': create_two_hidden_layers_model(input_dim, 128, 64),
        'Three Hidden Layers (256, 128, 64 units)': create_three_hidden_layers_model(input_dim, 256, 128, 64)
    }
    
    # Train and evaluate each architecture
    results = {}
    for name, model in architectures.items():
        metrics, history = train_and_evaluate_architecture(
            name, model, X_train, X_test, y_train, y_test,
            epochs=epochs, batch_size=batch_size, verbose=0
        )
        results[name] = metrics
        
        # Plot confusion matrix
        cm_title = f"Confusion Matrix - {name}\n(Test Size: {test_size})"
        plot_confusion_matrix(
            metrics['confusion_matrix'], 
            cm_title,
            save_path=f"confusion_matrix_{name.replace(' ', '_').replace('(', '').replace(')', '').lower()}_test_{test_size}.png"
        )
    
    # Summary of results
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    print(f"\n{'Architecture':<45} {'ACC':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}")
    print("-"*80)
    
    for name, metrics in results.items():
        print(f"{name:<45} {metrics['accuracy']:<8.4f} {metrics['precision']:<8.4f} "
              f"{metrics['recall']:<8.4f} {metrics['f1_score']:<8.4f}")
    
    print("="*60 + "\n")
    
    return results


def run_experiments_with_varying_splits():
    """
    Run experiments with varying test dataset splits.
    """
    test_sizes = [0.1, 0.2, 0.3, 0.4]
    
    print("\n" + "="*60)
    print("EXPERIMENTS WITH VARYING TEST SPLITS")
    print("="*60)
    
    all_results = {}
    for test_size in test_sizes:
        print(f"\n\n{'#'*60}")
        print(f"# Experiment with Test Size = {test_size}")
        print(f"{'#'*60}\n")
        
        results = run_experiment(test_size=test_size, epochs=100, batch_size=32)
        all_results[test_size] = results
    
    # Compare results across different splits
    print("\n" + "="*60)
    print("COMPARISON ACROSS DIFFERENT TEST SPLITS")
    print("="*60)
    
    for architecture in ['Baseline (No Hidden Layer)', 'One Hidden Layer (64 units)', 
                        'Two Hidden Layers (128, 64 units)', 'Three Hidden Layers (256, 128, 64 units)']:
        print(f"\n{architecture}:")
        print(f"{'Test Size':<12} {'ACC':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}")
        print("-"*50)
        
        for test_size in test_sizes:
            metrics = all_results[test_size][architecture]
            print(f"{test_size:<12.1f} {metrics['accuracy']:<8.4f} {metrics['precision']:<8.4f} "
                  f"{metrics['recall']:<8.4f} {metrics['f1_score']:<8.4f}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run default experiment with test_size=0.2
    print("\n" + "#"*60)
    print("# MAIN EXPERIMENT (Default Configuration)")
    print("#"*60)
    run_experiment(test_size=0.2, epochs=100, batch_size=32)
    
    # Run experiments with varying test splits
    run_experiments_with_varying_splits()
