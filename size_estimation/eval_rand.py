'''
Navid.Zarrabi@torontomu.ca
Jan 2026
Goal: 
Perform repeated stratified split evaluation with different random seeds
and calculate averaged metrics and confusion matrices for size estimation models.
'''

import numpy as np
import pandas as pd
import yaml
import os
import sys
import tensorflow as tf
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

from tensorflow.keras.callbacks import EarlyStopping
from model_architecture import create_model
from util import clean_radius, get_bin_label


def configure_gpu():
    """Configure TensorFlow to use CPU only."""
    # Force TensorFlow to use CPU only
    tf.config.set_visible_devices([], 'GPU')
    print("💻 Configured TensorFlow to use CPU only")
    return True

# Add parent directory to path to import stratified_split
# sys.path.append(str(Path(__file__).parent.parent / "data" / "data_splitter"))
from data_split import stratified_split_by_size, save_material_splits

def load_config():
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_stratified_splits_with_seed(seed, config):
    """Create stratified splits with a specific random seed."""
    print(f"Creating stratified splits with seed {seed}...")
    
    # Load the main dataset
    data_path = Path(__file__).parent.parent / "data" / "all_labeled_signals.csv"
    data = pd.read_csv(data_path)
    
    # Define size bin ranges from config
    bin_ranges = [(bin['min'], bin['max']) for bin in config['size_bins']]
    
    # Create splits with specific seed
    splits = stratified_split_by_size(data, bin_ranges, seed=seed)
    
    # Save splits to temporary directory with seed
    temp_dir = Path(__file__).parent.parent / "data" / f"temp_splits_seed_{seed}"
    temp_dir.mkdir(exist_ok=True)
    
    save_material_splits(splits, str(temp_dir))
    
    return temp_dir

def load_material_data(material, split_type, splits_dir):
    """Load data for a specific material and split type from given splits directory."""
    file_path = splits_dir / f"{material.lower()}_{split_type}.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Split file not found: {file_path}")
    
    data = pd.read_csv(file_path)
    
    # Clean radius values
    data['Radius'] = data['Radius'].apply(clean_radius)
    
    # Convert signal strings to numpy arrays
    def parse_signal(x):
        if isinstance(x, str):
            values = [val.strip() for val in x.split(',') if val.strip() and val.strip() != '-']
            try:
                return np.array([float(val) for val in values if val])
            except ValueError:
                return np.fromstring(x, sep=',', dtype=float)
        return x
    
    data['Signal'] = data['Signal'].apply(parse_signal)
    
    return data

def prepare_data(data, bin_ranges, max_length):
    """Prepare X and y from data."""
    # Pad signals to uniform length
    X = np.array([np.pad(signal, (0, max_length - len(signal))) for signal in data['Signal']])
    
    # Convert radius to bin labels
    y = data['Radius'].apply(lambda x: get_bin_label(x, bin_ranges))
    
    # Remove samples that don't fall into any bin
    valid_indices = y.notna()
    X = X[valid_indices]
    y = y[valid_indices]
    
    return X, y

def train_and_evaluate_single_seed(seed, config, materials, bin_ranges):
    """Train and evaluate models for all materials with a single random seed."""
    print(f"\n{'='*50}")
    print(f"EVALUATION WITH RANDOM SEED: {seed}")
    print(f"{'='*50}")
    
    # Create stratified splits with this seed
    splits_dir = create_stratified_splits_with_seed(seed, config)
    
    results = {}
    
    for material in materials:
        print(f"\nProcessing material: {material}")
        
        try:
            # Load data splits
            train_data = load_material_data(material, 'train', splits_dir)
            val_data = load_material_data(material, 'val', splits_dir)
            test_data = load_material_data(material, 'test', splits_dir)
            
            # Determine max signal length
            all_signals = (list(train_data['Signal']) + 
                          list(val_data['Signal']) + 
                          list(test_data['Signal']))
            max_length = max(len(signal) for signal in all_signals)
            
            # Prepare data
            X_train, y_train = prepare_data(train_data, bin_ranges, max_length)
            X_val, y_val = prepare_data(val_data, bin_ranges, max_length)
            X_test, y_test = prepare_data(test_data, bin_ranges, max_length)
            
            print(f"Data sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            
            # Check minimum samples (use a default value since config doesn't have min_samples)
            min_samples = 10  # Default minimum samples per material
            if len(X_train) < min_samples:
                print(f"Skipping {material} - not enough training samples ({len(X_train)} < {min_samples})")
                continue
            
            # Create and train model
            model = create_model((X_train.shape[1],), config['model']['num_classes'])
            
            # Early stopping
            early_stopping = EarlyStopping(
                monitor=config['training']['early_stopping']['monitor'],
                patience=config['training']['early_stopping']['patience'],
                restore_best_weights=config['training']['early_stopping']['restore_best_weights'],
                verbose=0  # Reduce verbosity
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=config['training']['epochs'],
                batch_size=config['training']['batch_size'],
                validation_data=(X_val, y_val),
                verbose=0,  # Reduce verbosity
                callbacks=[early_stopping]
            )
            
            # Evaluate on test set
            y_pred = model.predict(X_test, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred_classes)
            existing_labels = sorted(set(y_test.astype(int)))
            
            # Get precision, recall, f1-score
            precision, recall, f1, support = precision_recall_fscore_support(
                y_test, y_pred_classes, labels=existing_labels, average=None, zero_division=0
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred_classes, labels=existing_labels)
            
            # Store results
            results[material] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': support,
                'confusion_matrix': cm,
                'existing_labels': existing_labels,
                'y_test': y_test,
                'y_pred': y_pred_classes
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"Error processing {material}: {e}")
            continue
    
    # Clean up temporary directory
    import shutil
    shutil.rmtree(splits_dir)
    
    return results

def aggregate_results(all_results, materials, bin_ranges, config):
    """Aggregate results across all seeds."""
    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS ACROSS ALL SEEDS")
    print(f"{'='*60}")
    
    aggregated = {}
    
    for material in materials:
        if material not in all_results[0]:  # Skip if material wasn't processed
            continue
            
        print(f"\nMaterial: {material}")
        print("-" * 40)
        
        # Collect all metrics for this material
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        confusion_matrices = []
        
        for seed_results in all_results:
            if material in seed_results:
                accuracies.append(seed_results[material]['accuracy'])
                precisions.append(seed_results[material]['precision'])
                recalls.append(seed_results[material]['recall'])
                f1_scores.append(seed_results[material]['f1_score'])
                confusion_matrices.append(seed_results[material]['confusion_matrix'])
        
        # Calculate statistics
        acc_mean = np.mean(accuracies)
        acc_std = np.std(accuracies)
        
        # For precision, recall, f1 - handle per-class metrics
        precision_mean = np.mean(precisions, axis=0)
        precision_std = np.std(precisions, axis=0)
        recall_mean = np.mean(recalls, axis=0)
        recall_std = np.std(recalls, axis=0)
        f1_mean = np.mean(f1_scores, axis=0)
        f1_std = np.std(f1_scores, axis=0)
        
        # Average confusion matrix
        avg_confusion_matrix = np.mean(confusion_matrices, axis=0)
        
        # Get existing labels from first result
        existing_labels = all_results[0][material]['existing_labels']
        
        # Print results
        print(f"Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
        print("\nPer-class metrics:")
        for i, label in enumerate(existing_labels):
            bin_info = f"Bin {label+1} ({bin_ranges[label][0]}-{bin_ranges[label][1]})"
            print(f"  {bin_info}:")
            print(f"    Precision: {precision_mean[i]:.4f} ± {precision_std[i]:.4f}")
            print(f"    Recall:    {recall_mean[i]:.4f} ± {recall_std[i]:.4f}")
            print(f"    F1-score:  {f1_mean[i]:.4f} ± {f1_std[i]:.4f}")
        
        # Macro averages
        macro_precision = np.mean(precision_mean)
        macro_recall = np.mean(recall_mean)
        macro_f1 = np.mean(f1_mean)
        
        print(f"\nMacro averages:")
        print(f"  Precision: {macro_precision:.4f}")
        print(f"  Recall:    {macro_recall:.4f}")
        print(f"  F1-score:  {macro_f1:.4f}")
        
        # Store aggregated results
        aggregated[material] = {
            'accuracy_mean': acc_mean,
            'accuracy_std': acc_std,
            'precision_mean': precision_mean,
            'precision_std': precision_std,
            'recall_mean': recall_mean,
            'recall_std': recall_std,
            'f1_mean': f1_mean,
            'f1_std': f1_std,
            'avg_confusion_matrix': avg_confusion_matrix,
            'existing_labels': existing_labels,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1
        }
        
        # Plot averaged confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(avg_confusion_matrix, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=[f"Bin {i+1}" for i in existing_labels],
                    yticklabels=[f"Bin {i+1}" for i in existing_labels])
        plt.title(f'Averaged Confusion Matrix - {material}\n(Mean over {len(all_results)} seeds)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        
        # Save plot
        plot_dir = Path(config['output']['plot_dir'])
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / f'averaged_confusion_matrix_{material}.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved averaged confusion matrix to: {plot_path}")
    
    # Print averaged confusion matrices at the end
    print(f"\n{'='*80}")
    print("AVERAGED CONFUSION MATRICES ACROSS ALL SEEDS")
    print(f"{'='*80}")
    
    for material, results in aggregated.items():
        print(f"\nMaterial: {material}")
        print("-" * 60)
        
        avg_cm = results['avg_confusion_matrix']
        existing_labels = results['existing_labels']
        
        # Create header
        header = "True\\Pred"
        for i in existing_labels:
            header += f"\t Bin {i+1}"
        print(header)
        print("-" * 60)
        
        # Print each row
        for i, true_label in enumerate(existing_labels):
            row = f"Bin {true_label+1}\t"
            for j, pred_label in enumerate(existing_labels):
                row += f"\t{avg_cm[i,j]:6.1f}"
            print(row)
        
        # Print totals
        print("-" * 60)
        print(f"Total samples per run: {np.sum(avg_cm):.0f}")
        print(f"Average accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
    
    return aggregated

def main():
    # Configure GPU first
    configure_gpu()
    
    # Configuration
    config = load_config()
    bin_ranges = [(bin['min'], bin['max']) for bin in config['size_bins']]
    
    # Test parameters
    seeds = [42, 123, 456, 789, 2024]  # Different random seeds
    materials = ['GLASS', 'PE', 'PMMA', 'STEEL']
    
    print(f"Starting repeated evaluation with {len(seeds)} different random seeds: {seeds}")
    print(f"Materials to evaluate: {materials}")
    
    # Collect results from all seeds
    all_results = []
    
    for i, seed in enumerate(seeds):
        print(f"\n{'#'*60}")
        print(f"ITERATION {i+1}/{len(seeds)} - SEED {seed}")
        print(f"{'#'*60}")
        
        seed_results = train_and_evaluate_single_seed(seed, config, materials, bin_ranges)
        all_results.append(seed_results)
    
    # Aggregate and display results
    aggregated_results = aggregate_results(all_results, materials, bin_ranges, config)
    
    # Save aggregated results to file
    results_file = Path(config['output']['model_dir']) / 'repeated_evaluation_results.txt'
    
    # Create the output directory if it doesn't exist
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write(f"Repeated Stratified Evaluation Results\n")
        f.write(f"Seeds used: {seeds}\n")
        f.write(f"Number of repetitions: {len(seeds)}\n\n")
        
        for material, results in aggregated_results.items():
            f.write(f"Material: {material}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}\n")
            f.write(f"Macro Precision: {results['macro_precision']:.4f}\n")
            f.write(f"Macro Recall: {results['macro_recall']:.4f}\n")
            f.write(f"Macro F1-score: {results['macro_f1']:.4f}\n\n")
    
    print(f"\nDetailed results saved to: {results_file}")
    print("\nRepeated evaluation completed successfully!")

if __name__ == '__main__':
    main()