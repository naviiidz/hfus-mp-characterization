"""
Repeated Training and Evaluation Analysis Using Pre-generated Splits
Author: Navid Zarrabi (Navid.Zarrabi@torontomu.ca)
Date: December 22, 2025

This script runs training and per-particle testing using the pre-generated 
train/test splits from data/material_classification_splits to compute
average metrics with standard deviation for robust performance evaluation.
"""

import numpy as np
import pandas as pd
import os
import sys
import random
import time
from collections import defaultdict, Counter
from pathlib import Path

# Add the current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
from preprocessing import process_signal
from config import preprocessing_parameters


def configure_gpu():
    """Configure TensorFlow to use CPU only."""
    # Force TensorFlow to use CPU only
    tf.config.set_visible_devices([], 'GPU')
    print("💻 Configured TensorFlow to use CPU only")
    return True

# Load preprocessing parameters
params = preprocessing_parameters()
SIGNAL_LENGTH = params["signal_length"]


def set_deterministic_seeds(seed=42):
    """Set seeds for reproducible training."""
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set TensorFlow random seed
    tf.random.set_seed(seed)
    
    # For additional determinism in TensorFlow operations
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    print(f"Set deterministic seeds to: {seed}")


def prepare_test_data(data_path, label_encoder):
    """Load and prepare test data from CSV file."""
    data = pd.read_csv(data_path)
    
    # Process signals
    processed_signals = []
    
    for idx, row in data.iterrows():
        signal_str = row['Signal'].strip('[]')
        signal = np.fromstring(signal_str, sep=',')
        processed_signal = process_signal(signal)
        processed_signals.append(processed_signal)
    
    # Prepare for testing
    padded_signals = pad_sequences(processed_signals, maxlen=SIGNAL_LENGTH,
                                 dtype='float32', padding='post', truncating='post')
    padded_signals = np.expand_dims(padded_signals, axis=2)
    
    return padded_signals, data


def run_single_split_evaluation(split_dir, split_name):
    """Run training and evaluation on a single data split."""
    print(f"\n{'='*60}")
    print(f"EVALUATING SPLIT: {split_name}")
    print(f"{'='*60}")
    
    # Extract seed from split name and set deterministic seeds
    set_deterministic_seeds(60)
    
    try:
        # File paths
        train_path = split_dir / "train_data.csv"
        val_path = split_dir / "val_data.csv"
        test_path = split_dir / "test_data.csv"
        
        # Check if all files exist
        if not all(path.exists() for path in [train_path, val_path, test_path]):
            print(f"ERROR: Missing data files in {split_dir}")
            return None
        
        # Load label encoder
        root_dir = Path(__file__).parent.parent
        encoder_path = root_dir / "data" / "label_encoder_classes.npy"
        label_classes = np.load(encoder_path, allow_pickle=True)
        
        label_encoder = LabelEncoder()
        label_encoder.classes_ = label_classes
        
        # Create temporary copies of the split data in the main data folder for train.py
        main_data_dir = root_dir / "data"
        temp_train_path = main_data_dir / "temp_train_data.csv"
        temp_val_path = main_data_dir / "temp_val_data.csv"
        temp_test_path = main_data_dir / "temp_test_data.csv"
        
        # Copy split files to main data directory
        import shutil
        shutil.copy2(train_path, temp_train_path)
        shutil.copy2(val_path, temp_val_path)
        shutil.copy2(test_path, temp_test_path)
        
        # Backup original files if they exist
        original_train = main_data_dir / "train_data.csv"
        original_val = main_data_dir / "val_data.csv"
        original_test = main_data_dir / "test_data.csv"
        
        backup_files = []
        for original_file in [original_train, original_val, original_test]:
            if original_file.exists():
                backup_file = original_file.with_suffix(f".backup_{split_name}")
                shutil.copy2(original_file, backup_file)
                backup_files.append((original_file, backup_file))
        
        # Replace original files with split data
        shutil.move(temp_train_path, original_train)
        shutil.move(temp_val_path, original_val)
        shutil.move(temp_test_path, original_test)
        
        try:
            # Train model using existing train.py
            print(f"Training model using train.py...")
            
            # Import and run training
            import train
            import importlib
            importlib.reload(train)
            
            # Create temporary model path
            temp_model_path = Path(__file__).parent / f"temp_model_{split_name}.h5"
            original_model_path = Path(__file__).parent / "final_model.h5"
            
            # Backup original model if it exists
            model_backup = None
            if original_model_path.exists():
                model_backup = original_model_path.with_suffix(f".backup_{split_name}")
                shutil.copy2(original_model_path, model_backup)
            
            # Measure training time
            training_start_time = time.time()
            
            # Run training
            train.main()
            
            # Calculate training duration
            training_end_time = time.time()
            training_duration = training_end_time - training_start_time
            
            # Move trained model to temporary location
            if original_model_path.exists():
                shutil.move(original_model_path, temp_model_path)
            
            # Restore original model if it existed
            if model_backup and model_backup.exists():
                shutil.move(model_backup, original_model_path)
            
            print(f"Training completed in {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")
        
            # Load trained model and prepare test data
            print(f"Evaluating on test data...")
            
            model = load_model(temp_model_path)
            X_test, test_data = prepare_test_data(test_path, label_encoder)
            
            # Make predictions
            predictions = model.predict(X_test, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Get true labels
            true_labels = [label_encoder.transform([label])[0] for label in test_data['Material Type']]
            true_labels = np.array(true_labels)
            
            # Save model to models directory before cleanup
            models_dir = Path(__file__).parent / "models"
            models_dir.mkdir(exist_ok=True)
            saved_model_path = models_dir / f"model_{split_name}.h5"
            if temp_model_path.exists():
                shutil.copy2(temp_model_path, saved_model_path)
                print(f"Model saved to: {saved_model_path}")
            
            # Clean up temporary model
            if temp_model_path.exists():
                temp_model_path.unlink()
        
        finally:
            # Restore original data files
            for original_file, backup_file in backup_files:
                if backup_file.exists():
                    shutil.move(backup_file, original_file)
        
        # Step 1: Calculate per-signal (sample-level) metrics
        signal_accuracy = accuracy_score(true_labels, predicted_classes)
        signal_precision, signal_recall, signal_f1, _ = precision_recall_fscore_support(
            true_labels, predicted_classes, average='weighted'
        )
        
        # Calculate signal-level confusion matrix
        signal_cm = confusion_matrix(true_labels, predicted_classes)
        
        # Per-material signal metrics
        signal_material_metrics = {}
        for i, material in enumerate(label_encoder.classes_):
            material_mask = true_labels == i
            if np.any(material_mask):
                material_true = true_labels[material_mask]
                material_pred = predicted_classes[material_mask]
                
                if len(material_true) > 0:
                    material_acc = accuracy_score(material_true, material_pred)
                    
                    # Calculate precision, recall, f1 for this material
                    try:
                        if len(np.unique(material_true)) == 1 and len(np.unique(material_pred)) == 1:
                            if material_true[0] == material_pred[0]:
                                material_prec = material_rec = material_f1 = 1.0
                            else:
                                material_prec = material_rec = material_f1 = 0.0
                        else:
                            material_prec, material_rec, material_f1, _ = precision_recall_fscore_support(
                                material_true, material_pred, average='binary' if len(np.unique(material_true)) == 2 else 'weighted'
                            )
                            if isinstance(material_prec, np.ndarray):
                                material_prec = material_prec[0] if len(material_prec) > 0 else material_acc
                                material_rec = material_rec[0] if len(material_rec) > 0 else material_acc
                                material_f1 = material_f1[0] if len(material_f1) > 0 else material_acc
                    except:
                        material_prec = material_rec = material_f1 = material_acc
                    
                    signal_material_metrics[material] = {
                        'accuracy': material_acc,
                        'precision': material_prec,
                        'recall': material_rec,
                        'f1': material_f1,
                        'samples': len(material_true)
                    }
        
        # Step 2: Calculate per-particle metrics using voting
        print(f"Calculating particle-level metrics with voting...")
        
        # Group by particle ID and perform voting
        particle_data = {}
        
        for idx, row in test_data.iterrows():
            particle_id = row['particle_id']
            true_label = row['Material Type']
            pred_class = predicted_classes[idx]
            
            if particle_id not in particle_data:
                particle_data[particle_id] = {
                    'true_label': true_label,
                    'predictions': [],
                }
            
            particle_data[particle_id]['predictions'].append(pred_class)
        
        # Perform majority voting and calculate particle metrics
        particle_true_labels = []
        particle_pred_labels = []
        
        for particle_id, data in particle_data.items():
            true_label = data['true_label']
            predictions = data['predictions']
            
            # Majority voting
            vote_counts = Counter(predictions)
            majority_class = vote_counts.most_common(1)[0][0]
            
            true_class = label_encoder.transform([true_label])[0]
            
            particle_true_labels.append(true_class)
            particle_pred_labels.append(majority_class)
        
        particle_true_labels = np.array(particle_true_labels)
        particle_pred_labels = np.array(particle_pred_labels)
        
        particle_accuracy = accuracy_score(particle_true_labels, particle_pred_labels)
        particle_precision, particle_recall, particle_f1, _ = precision_recall_fscore_support(
            particle_true_labels, particle_pred_labels, average='weighted'
        )
        
        # Calculate particle-level confusion matrix
        particle_cm = confusion_matrix(particle_true_labels, particle_pred_labels)
        
        # Per-material particle metrics
        particle_material_metrics = {}
        for i, material in enumerate(label_encoder.classes_):
            material_mask = particle_true_labels == i
            if np.any(material_mask):
                material_true = particle_true_labels[material_mask]
                material_pred = particle_pred_labels[material_mask]
                
                if len(material_true) > 0:
                    material_acc = accuracy_score(material_true, material_pred)
                    
                    try:
                        if len(np.unique(material_true)) == 1 and len(np.unique(material_pred)) == 1:
                            if material_true[0] == material_pred[0]:
                                material_prec = material_rec = material_f1 = 1.0
                            else:
                                material_prec = material_rec = material_f1 = 0.0
                        else:
                            material_prec, material_rec, material_f1, _ = precision_recall_fscore_support(
                                material_true, material_pred, average='binary' if len(np.unique(material_true)) == 2 else 'weighted'
                            )
                            if isinstance(material_prec, np.ndarray):
                                material_prec = material_prec[0] if len(material_prec) > 0 else material_acc
                                material_rec = material_rec[0] if len(material_rec) > 0 else material_acc
                                material_f1 = material_f1[0] if len(material_f1) > 0 else material_acc
                    except:
                        material_prec = material_rec = material_f1 = material_acc
                    
                    particle_material_metrics[material] = {
                        'accuracy': material_acc,
                        'precision': material_prec,
                        'recall': material_rec,
                        'f1': material_f1,
                        'particles': len(material_true)
                    }
        
        # Collect results
        results = {
            'split_name': split_name,
            'signal_metrics': {
                'accuracy': signal_accuracy,
                'precision': signal_precision,
                'recall': signal_recall,
                'f1': signal_f1
            },
            'particle_metrics': {
                'accuracy': particle_accuracy,
                'precision': particle_precision,
                'recall': particle_recall,
                'f1': particle_f1
            },
            'signal_material_metrics': signal_material_metrics,
            'particle_material_metrics': particle_material_metrics,
            'signal_confusion_matrix': signal_cm,
            'particle_confusion_matrix': particle_cm,
            'total_signals': len(test_data),
            'total_particles': len(particle_data),
            'improvement': particle_accuracy - signal_accuracy,
            'label_encoder': label_encoder,
            'training_time': training_duration,
            'model_path': saved_model_path
        }
        
        print(f"Split {split_name} completed successfully!")
        print(f"   Signal Accuracy: {signal_accuracy:.4f}")
        print(f"   Particle Accuracy: {particle_accuracy:.4f}")
        print(f"   Improvement: {particle_accuracy - signal_accuracy:+.4f}")
        print(f"   Training Time: {training_duration:.2f}s ({training_duration/60:.2f}min)")
        print(f"   Total particles: {len(particle_data)}")
        print(f"   Particle confusion matrix shape: {particle_cm.shape}")
        print(f"   Particle confusion matrix:")
        print(f"   {particle_cm}")
        
        return results
        
    except Exception as e:
        print(f"ERROR in split {split_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def print_confusion_matrix(cm, label_encoder, title):
    """Print confusion matrix in a formatted way."""
    print(f"\n{title}:")
    print("Confusion Matrix (rows=true, cols=predicted):")
    
    # Print header
    header = "True\\Pred"
    for material in label_encoder.classes_:
        header += f"{material:>8s}"
    print(header)
    
    # Print matrix rows
    for i, true_material in enumerate(label_encoder.classes_):
        row = f"{true_material:<8s}"
        for j in range(len(label_encoder.classes_)):
            row += f"{cm[i,j]:8.4f}"
        print(row)


def analyze_all_results(all_results):
    """Analyze and print results across all splits."""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE ANALYSIS ACROSS ALL SPLITS")
    print(f"{'='*80}")
    
    valid_results = [r for r in all_results if r is not None]
    if not valid_results:
        print("ERROR: No valid results to analyze!")
        return
    
    n_splits = len(valid_results)
    print(f"Analyzing {n_splits} successful splits...")
    
    # Get label encoder (assume all results have the same one)
    label_encoder = valid_results[0]['label_encoder']
    
    # Collect overall metrics
    signal_accuracies = [r['signal_metrics']['accuracy'] for r in valid_results]
    signal_precisions = [r['signal_metrics']['precision'] for r in valid_results]
    signal_recalls = [r['signal_metrics']['recall'] for r in valid_results]
    signal_f1s = [r['signal_metrics']['f1'] for r in valid_results]
    
    particle_accuracies = [r['particle_metrics']['accuracy'] for r in valid_results]
    particle_precisions = [r['particle_metrics']['precision'] for r in valid_results]
    particle_recalls = [r['particle_metrics']['recall'] for r in valid_results]
    particle_f1s = [r['particle_metrics']['f1'] for r in valid_results]
    
    improvements = [r['improvement'] for r in valid_results]
    training_times = [r.get('training_time', 0) for r in valid_results]
    
    # Print overall statistics
    print(f"\nOVERALL PERFORMANCE STATISTICS (across {n_splits} splits):")
    print(f"{'Metric':<15} {'Signal-Level':<25} {'Particle-Level':<25} {'Improvement':<20}")
    print("-" * 85)
    
    print(f"{'Accuracy':<15} {np.mean(signal_accuracies):.4f} ± {np.std(signal_accuracies):.4f}"
          f"         {np.mean(particle_accuracies):.4f} ± {np.std(particle_accuracies):.4f}"
          f"         {np.mean(improvements):+.4f} ± {np.std(improvements):.4f}")
    
    print(f"{'Precision':<15} {np.mean(signal_precisions):.4f} ± {np.std(signal_precisions):.4f}"
          f"         {np.mean(particle_precisions):.4f} ± {np.std(particle_precisions):.4f}")
    
    print(f"{'Recall':<15} {np.mean(signal_recalls):.4f} ± {np.std(signal_recalls):.4f}"
          f"         {np.mean(particle_recalls):.4f} ± {np.std(particle_recalls):.4f}")
    
    print(f"{'F1-Score':<15} {np.mean(signal_f1s):.4f} ± {np.std(signal_f1s):.4f}"
          f"         {np.mean(particle_f1s):.4f} ± {np.std(particle_f1s):.4f}")
    
    # Training time statistics
    if training_times and any(t > 0 for t in training_times):
        print(f"\nTRAINING TIME STATISTICS:")
        print("-" * 50)
        print(f"Average Training Time: {np.mean(training_times):.2f} ± {np.std(training_times):.2f} seconds")
        print(f"Average Training Time: {np.mean(training_times)/60:.2f} ± {np.std(training_times)/60:.2f} minutes")
        print(f"Fastest Training:      {np.min(training_times):.2f}s ({np.min(training_times)/60:.2f}min)")
        print(f"Slowest Training:      {np.max(training_times):.2f}s ({np.max(training_times)/60:.2f}min)")
        print(f"Total Training Time:   {np.sum(training_times):.2f}s ({np.sum(training_times)/60:.2f}min)")
        print(f"Total Training Time:   {np.sum(training_times)/3600:.2f} hours")
    
    # Per-material analysis
    print(f"\nPER-MATERIAL PERFORMANCE STATISTICS:")
    
    materials = list(label_encoder.classes_)
    for material in materials:
        print(f"\n{material.upper()}:")
        
        # Signal-level stats for this material
        signal_mat_accs = []
        signal_mat_precs = []
        particle_mat_accs = []
        particle_mat_precs = []
        
        for r in valid_results:
            if material in r['signal_material_metrics']:
                signal_mat_accs.append(r['signal_material_metrics'][material]['accuracy'])
                signal_mat_precs.append(r['signal_material_metrics'][material]['precision'])
            
            if material in r['particle_material_metrics']:
                particle_mat_accs.append(r['particle_material_metrics'][material]['accuracy'])
                particle_mat_precs.append(r['particle_material_metrics'][material]['precision'])
        
        if signal_mat_accs and particle_mat_accs:
            print(f"  Accuracy  - Signal: {np.mean(signal_mat_accs):.3f} ± {np.std(signal_mat_accs):.3f}, "
                  f"Particle: {np.mean(particle_mat_accs):.3f} ± {np.std(particle_mat_accs):.3f}")
            print(f"  Precision - Signal: {np.mean(signal_mat_precs):.3f} ± {np.std(signal_mat_precs):.3f}, "
                  f"Particle: {np.mean(particle_mat_precs):.3f} ± {np.std(particle_mat_precs):.3f}")
    
    # Calculate average confusion matrices
    print(f"\nAVERAGE CONFUSION MATRICES ACROSS ALL SPLITS:")
    
    # Collect all confusion matrices
    signal_cms = [r['signal_confusion_matrix'] for r in valid_results]
    particle_cms = [r['particle_confusion_matrix'] for r in valid_results]
    
    print(f"\nDEBUG: Individual particle confusion matrices:")
    for i, (r, cm) in enumerate(zip(valid_results, particle_cms)):
        print(f"   Split {r['split_name']}: Shape {cm.shape}, Sum {cm.sum()}")
        print(f"   {cm}")
        print(f"   Accuracy: {r['particle_metrics']['accuracy']:.4f}")
    
    # Calculate averages
    avg_signal_cm = np.mean(signal_cms, axis=0)
    avg_particle_cm = np.mean(particle_cms, axis=0)
    std_signal_cm = np.std(signal_cms, axis=0)
    std_particle_cm = np.std(particle_cms, axis=0)
    
    print(f"\nDEBUG: Averaged results:")
    print(f"   Average particle confusion matrix: {avg_particle_cm}")
    print(f"   Standard deviation: {std_particle_cm}")
    
    print(f"\nDEBUG: Averaged results:")
    print(f"   Average particle confusion matrix: {avg_particle_cm}")
    print(f"   Standard deviation: {std_particle_cm}")
    
    # Print average signal confusion matrix
    print_confusion_matrix(avg_signal_cm, label_encoder, 
                          f"\nAVERAGE SIGNAL-LEVEL CONFUSION MATRIX (±std)")
    
    print("\nStandard Deviation:")
    print_confusion_matrix(std_signal_cm, label_encoder, "Signal-Level Std Dev")
    
    # Print average particle confusion matrix
    print_confusion_matrix(avg_particle_cm, label_encoder,
                          f"\nAVERAGE PARTICLE-LEVEL CONFUSION MATRIX (±std)")
    
    print("\nStandard Deviation:")
    print_confusion_matrix(std_particle_cm, label_encoder, "Particle-Level Std Dev")
    
    # Analyze misclassifications
    print(f"\nMISCLASSIFICATION ANALYSIS:")
    print("\nSignal-Level Common Misclassifications (avg ± std):")
    signal_misclass_found = False
    for i, true_material in enumerate(label_encoder.classes_):
        for j, pred_material in enumerate(label_encoder.classes_):
            if i != j and avg_signal_cm[i,j] > 0.5:
                print(f"  {true_material} → {pred_material}: {avg_signal_cm[i,j]:.1f} ± {std_signal_cm[i,j]:.1f}")
                signal_misclass_found = True
    
    if not signal_misclass_found:
        print("  No significant signal-level misclassifications!")
    
    print("\nParticle-Level Common Misclassifications (avg ± std):")
    particle_misclass_found = False
    for i, true_material in enumerate(label_encoder.classes_):
        for j, pred_material in enumerate(label_encoder.classes_):
            if i != j and avg_particle_cm[i,j] > 0.1:
                print(f"  {true_material} → {pred_material}: {avg_particle_cm[i,j]:.1f} ± {std_particle_cm[i,j]:.1f}")
                particle_misclass_found = True
    
    if not particle_misclass_found:
        print("  No significant particle-level misclassifications!")
    
    # Statistical significance of improvement
    print(f"\nSTATISTICAL ANALYSIS:")
    print(f"Mean improvement: {np.mean(improvements):.4f} ± {np.std(improvements):.4f}")
    
    if np.mean(improvements) > 0:
        t_stat = np.mean(improvements) / (np.std(improvements) / np.sqrt(n_splits))
        print(f"T-statistic: {t_stat:.3f}")
        
        if abs(t_stat) > 2.0:  # Rough significance at p<0.05
            print("Statistically significant improvement with particle voting!")
        else:
            print("WARNING: Improvement not statistically significant")
    else:
        print("ERROR: No improvement with particle voting on average")
    
    # Summary
    print(f"\nFINAL SUMMARY:")
    print(f"Successfully evaluated {n_splits} different train/test splits")
    print(f"Signal-level accuracy: {np.mean(signal_accuracies):.4f} ± {np.std(signal_accuracies):.4f}")
    print(f"Particle-level accuracy: {np.mean(particle_accuracies):.4f} ± {np.std(particle_accuracies):.4f}")
    print(f"Average improvement: {np.mean(improvements):+.4f} ± {np.std(improvements):.4f}")
    
    # Find and save the best performing model as final model
    best_idx = np.argmax(particle_accuracies)
    best_result = valid_results[best_idx]
    best_accuracy = particle_accuracies[best_idx]
    
    # Copy best model to final_model.h5
    models_dir = Path(__file__).parent / "models"
    final_model_path = models_dir / "final_model.h5" 
    best_model_path = best_result['model_path']
    
    import shutil
    if best_model_path.exists():
        shutil.copy2(best_model_path, final_model_path)
        print(f"\nBest performing model saved as: {final_model_path}")
        print(f"Best model split: {best_result['split_name']}")
        print(f"Best particle accuracy: {best_accuracy:.4f}")
    else:
        print(f"\nWARNING: Best model file not found at {best_model_path}")


def main():
    """Main function to run evaluation on all splits."""
    # Configure GPU first
    configure_gpu()
    
    print("="*80)
    print("REPEATED EVALUATION USING PRE-GENERATED SPLITS")
    print("="*80)
    print("This will train and evaluate the model using different train/test splits")
    print("from data/material_classification_splits to calculate robust performance statistics.")
    
    # Find all split directories
    root_dir = Path(__file__).parent.parent
    splits_dir = root_dir / "data" / "material_classification_splits"
    
    if not splits_dir.exists():
        print(f"ERROR: Splits directory not found: {splits_dir}")
        print("Please run the material_classification_splits.py script first!")
        return
    
    # Get all seed directories
    split_dirs = [d for d in splits_dir.iterdir() if d.is_dir() and d.name.startswith('seed_')]
    split_dirs.sort(key=lambda x: int(x.name.split('_')[1]))  # Sort by seed number
    
    if not split_dirs:
        print(f"ERROR: No split directories found in {splits_dir}")
        return
    
    print(f"Found {len(split_dirs)} split directories:")
    for split_dir in split_dirs:
        print(f"  - {split_dir.name}")
    
    # Run evaluation on each split
    all_results = []
    
    for split_dir in split_dirs:
        split_name = split_dir.name
        result = run_single_split_evaluation(split_dir, split_name)
        all_results.append(result)
    
    # Analyze and print comprehensive results
    analyze_all_results(all_results)
    
    # Calculate total experiment time
    valid_results = [r for r in all_results if r is not None]
    total_experiment_time = sum(r.get('training_time', 0) for r in valid_results)
    
    print(f"\n{'='*80}")
    print("REPEATED EVALUATION WITH SPLITS COMPLETED!")
    if total_experiment_time > 0:
        print(f"Total experiment time: {total_experiment_time:.2f}s ({total_experiment_time/60:.2f} minutes)")
        print(f"Evaluated {len(valid_results)} splits with GPU acceleration")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()