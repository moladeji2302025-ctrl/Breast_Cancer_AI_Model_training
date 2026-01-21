"""
Quick Test Script for Breast Cancer ANN Experiment

This script runs a quick version of the experiment with reduced epochs
to verify that everything is working correctly.
"""

import numpy as np
import tensorflow as tf
from breast_cancer_ann_experiment import run_experiment

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("QUICK TEST - BREAST CANCER ANN EXPERIMENT")
    print("Running with reduced epochs for faster testing")
    print("="*60 + "\n")
    
    # Run a quick test with fewer epochs
    results = run_experiment(test_size=0.2, epochs=20, batch_size=32)
    
    print("\n✓ Quick test completed successfully!")
    print("  Run 'python breast_cancer_ann_experiment.py' for the full experiment")
