#!/usr/bin/env python3
"""
SVM Classification on Scattering Features

Trains SVM classifier on scattering features for T-phase seismic event classification.
Runs multiple experiments with different random seeds for robust results.
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "path/to/scattering/features"  # Path to scattering features
OUTPUT_BASE = "path/to/svm/results"  # SVM results output directory
CLASSES = ['Single T-Phase', 'Multiple T-Phase', 'No T-Phase']
SEEDS = [42, 123, 456]  # Random seeds for reproducibility

def run_experiment(seed, features, labels):
    """Run SVM experiment with given seed"""
    print(f"\n=== Running SVM experiment with seed {seed} ===")
    
    output_dir = os.path.join(OUTPUT_BASE, f"run_svm_seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)

    # Split train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM
    print("Training SVM...")
    svm_model = SVC(kernel='linear', class_weight='balanced', 
                   random_state=seed, probability=True)
    svm_model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = svm_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")

    # Save model and scaler
    joblib.dump(svm_model, os.path.join(output_dir, 'svm_model.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

    # Save predictions and targets
    np.save(os.path.join(output_dir, 'predictions.npy'), y_pred)
    np.save(os.path.join(output_dir, 'targets.npy'), y_test)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=CLASSES, digits=4)
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"SVM Classification Results (Seed: {seed})\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    np.save(os.path.join(output_dir, 'confusion_matrix.npy'), cm)
    
    # Save confusion matrix as text
    with open(os.path.join(output_dir, 'confusion_matrix.txt'), 'w') as f:
        f.write("Confusion Matrix:\n")
        f.write("Predicted ->   " + "   ".join([f"{cls:>12}" for cls in CLASSES]) + "\n")
        for i, true_class in enumerate(CLASSES):
            f.write(f"Actual {true_class:>12}: " + 
                   "   ".join([f"{cm[i,j]:>12}" for j in range(len(CLASSES))]) + "\n")

    print(f"Results saved to: {output_dir}")
    print(f"Classification Report:\n{report}")
    
    return accuracy

def main():
    """Main function"""
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    # Load features
    features_path = os.path.join(DATA_DIR, 'scatter_features.npy')
    labels_path = os.path.join(DATA_DIR, 'labels.npy')

    if not (os.path.exists(features_path) and os.path.exists(labels_path)):
        print("Error: Feature files not found!")
        print("Please run extract_scatter_features.py first.")
        return

    features = np.load(features_path)
    labels = np.load(labels_path)

    print(f"Loaded features: {features.shape}")
    print(f"Loaded labels: {labels.shape}")
    
    # Print class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  {CLASSES[label]}: {count} samples")

    # Run experiments with different seeds
    accuracies = []
    for seed in SEEDS:
        accuracy = run_experiment(seed, features, labels)
        accuracies.append(accuracy)

    # Summary statistics
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print(f"\n{'='*50}")
    print("SUMMARY RESULTS")
    print(f"{'='*50}")
    print(f"Individual accuracies: {[f'{acc:.4f}' for acc in accuracies]}")
    print(f"Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Results saved to: {OUTPUT_BASE}")
    
    # Save summary
    summary_path = os.path.join(OUTPUT_BASE, 'summary_results.txt')
    with open(summary_path, 'w') as f:
        f.write("SVM Classification Summary\n")
        f.write("=" * 30 + "\n")
        f.write(f"Seeds used: {SEEDS}\n")
        f.write(f"Individual accuracies: {accuracies}\n")
        f.write(f"Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n")

if __name__ == "__main__":
    main()
