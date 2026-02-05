"""
Author: Navid Zarrabi (Navid.Zarrabi@torontomu.ca)
Date: May 26, 2025
Purpose: Train traditional ML models for material identification of microspheres.

Key Notes:
1. Modular design with separate functions for data loading, preprocessing, and model training
2. Configuration-driven approach using YAML
3. Comprehensive model evaluation and visualization
4. Support for multiple random seeds for robust evaluation
"""

import pandas as pd
import numpy as np
import ast
import random
from scipy.stats import kurtosis, skew, entropy
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import time
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from feature_extraction import fe_time, fe_freq, cal_corr, show_corr
import pickle

@dataclass
class ModelConfig:
    """Configuration for model training."""
    num_seeds: int
    random_state: int
    feature_correlation_threshold: float
    output_dir: Path
    train_data: Path = None
    val_data: Path = None
    test_data: Path = None

@dataclass
class ModelResult:
    """Container for model evaluation results."""
    # Particle-level metrics
    accuracy: float
    std_accuracy: float
    precision: float
    std_precision: float
    recall: float
    std_recall: float
    f1: float
    std_f1: float
    class_accuracies: np.ndarray
    std_class_accuracies: np.ndarray
    confusion_matrix: np.ndarray
    feature_importance: Dict[str, float] = None
    
    # Sample-level metrics
    sample_accuracy: float = 0.0
    sample_std_accuracy: float = 0.0
    sample_precision: float = 0.0
    sample_std_precision: float = 0.0
    sample_recall: float = 0.0
    sample_std_recall: float = 0.0
    sample_f1: float = 0.0
    sample_std_f1: float = 0.0

def load_config() -> ModelConfig:
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert paths to Path objects
    config['training']['output_dir'] = Path(__file__).parent.parent / config['training']['output_dir']
    
    # Data paths are optional since we use pre-existing splits
    return ModelConfig(**config['training'])

def setup_directories(config: ModelConfig) -> None:
    """Create necessary output directories."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / 'plots').mkdir(exist_ok=True)
    (config.output_dir / 'models').mkdir(exist_ok=True)

def load_data_from_splits(seed: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Load pre-existing train/val/test splits from material_classification_splits directory."""
    print(f"\nLoading data from pre-existing splits (seed={seed})...")
    
    # Load pre-existing splits
    base_path = Path(__file__).parent.parent
    splits_path = base_path / "data" / "material_classification_splits" / f"seed_{seed}"
    
    if not splits_path.exists():
        raise FileNotFoundError(f"Split directory not found: {splits_path}")
    
    # Load train/val/test data
    train_data = pd.read_csv(splits_path / "train_data.csv")
    val_data = pd.read_csv(splits_path / "val_data.csv")
    test_data = pd.read_csv(splits_path / "test_data.csv")
    
    print(f"Train set: {len(train_data)} samples")
    print(f"Val set: {len(val_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    
    def process_dataset(data):
        """Process a single dataset and extract features."""
        signals = data['Signal'].values
        material_types = data['Material Type'].values
        
        # Convert signals from strings to numpy arrays
        signals_array = [np.fromstring(signal, sep=',') for signal in signals]
        
        # Apply FFT to all signals
        signals_freq_array = [signal_to_frequency(signal) for signal in signals_array]
        
        # Extract features
        features_list = []
        for i in range(len(signals_array)):
            time_features = fe_time(signals_array[i])
            freq_features = fe_freq(signals_freq_array[i])
            all_features = list(time_features.values()) + list(freq_features.values())
            features_list.append(all_features)
        
        return np.array(features_list), material_types
    
    # Process all three datasets
    X_train, y_train = process_dataset(train_data)
    X_val, y_val = process_dataset(val_data)
    X_test, y_test = process_dataset(test_data)
    
    # Get test particle IDs for voting (if particle_id column exists)
    if 'particle_id' in test_data.columns:
        test_particle_ids = test_data['particle_id'].values
    else:
        # Create dummy particle IDs if not available
        test_particle_ids = np.arange(len(test_data))
        print("Warning: No particle_id column found. Creating dummy particle IDs.")
    
    # Get feature names (using train data as reference)
    signals_sample = [np.fromstring(train_data['Signal'].iloc[0], sep=',')]
    signals_freq_sample = [signal_to_frequency(signals_sample[0])]
    time_feature_names = [f"time_{k}" for k in fe_time(signals_sample[0]).keys()]
    freq_feature_names = [f"freq_{k}" for k in fe_freq(signals_freq_sample[0]).keys()]
    feature_names = time_feature_names + freq_feature_names
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names, test_particle_ids
    
    def process_dataset(data):
        """Process a single dataset and extract features."""
        signals = data['Signal'].values
        material_types = data['Material Type'].values
        
        # Convert signals from strings to numpy arrays
        signals_array = [np.fromstring(signal, sep=',') for signal in signals]
        
        # Apply FFT to all signals
        signals_freq_array = [signal_to_frequency(signal) for signal in signals_array]
        
        # Extract features
        features_list = []
        for i in range(len(signals_array)):
            time_features = fe_time(signals_array[i])
            freq_features = fe_freq(signals_freq_array[i])
            all_features = list(time_features.values()) + list(freq_features.values())
            features_list.append(all_features)
        
        return np.array(features_list), material_types
    
    # Process all three datasets
    X_train, y_train = process_dataset(train_data)
    X_val, y_val = process_dataset(val_data)
    X_test, y_test = process_dataset(test_data)
    
    # Get test particle IDs for voting
    test_particle_ids = test_data['particle_id'].values
    
    # Get feature names (using train data as reference)
    signals_sample = [np.fromstring(train_data['Signal'].iloc[0], sep=',')]
    signals_freq_sample = [signal_to_frequency(signals_sample[0])]
    time_feature_names = [f"time_{k}" for k in fe_time(signals_sample[0]).keys()]
    freq_feature_names = [f"freq_{k}" for k in fe_freq(signals_freq_sample[0]).keys()]
    feature_names = time_feature_names + freq_feature_names
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names, test_particle_ids

def signal_to_frequency(signal: np.ndarray) -> np.ndarray:
    """Convert signal to frequency domain."""
    fft_result = np.fft.fft(signal)
    return np.abs(fft_result)

def get_models(seed: str) -> Dict[str, Any]:
    """Initialize all models to be evaluated."""
    # Convert seed to integer for random state
    random_state = int(seed)
    
    return {
        'Random Forest': RandomForestClassifier(random_state=random_state),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=500),
        'Gradient Boosting': GradientBoostingClassifier(random_state=random_state),
        
        # Best performing Neural Network configuration
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(128, 64), 
            activation='tanh',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=random_state
        ),
        
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'SVM (RBF Kernel)': SVC(kernel='rbf', random_state=random_state),
        'Voting Classifier': VotingClassifier(estimators=[
            ('rf', RandomForestClassifier(random_state=random_state)),
            ('lr', LogisticRegression(random_state=random_state, max_iter=500)),
            ('gbm', GradientBoostingClassifier(random_state=random_state)),
        ], voting='soft')
    }

def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray,
                  label_encoder: LabelEncoder) -> ModelResult:
    """Evaluate a trained model and return results."""
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    # Get feature importance if available
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(label_encoder.classes_, model.feature_importances_))
    
    return ModelResult(
        accuracy=accuracy,
        std_accuracy=0.0,  # Will be updated later
        precision=report['weighted avg']['precision'],
        std_precision=0.0,
        recall=report['weighted avg']['recall'],
        std_recall=0.0,
        f1=report['weighted avg']['f1-score'],
        std_f1=0.0,
        class_accuracies=class_accuracies,
        std_class_accuracies=np.zeros_like(class_accuracies),
        confusion_matrix=cm,
        feature_importance=feature_importance
    )

def evaluate_model_with_voting(model: Any, X_test: np.ndarray, y_test: np.ndarray, 
                              test_particle_ids: np.ndarray, label_encoder: LabelEncoder, 
                              bootstrap_particles: bool = True) -> ModelResult:
    """Evaluate a trained model using particle-level voting and return both sample and particle level results."""
    y_pred = model.predict(X_test)
    
    # Calculate sample-level metrics first
    sample_accuracy = accuracy_score(y_test, y_pred)
    sample_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Convert predictions back to original labels for voting
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_test_labels = label_encoder.inverse_transform(y_test)
    
    # Create DataFrame for easier grouping
    import pandas as pd
    predictions_df = pd.DataFrame({
        'particle_id': test_particle_ids,
        'true_label': y_test_labels,
        'pred_label': y_pred_labels
    })
    
    # Group by particle and apply majority voting
    particle_results = []
    for particle_id, group in predictions_df.groupby('particle_id'):
        # Get the true label (should be the same for all samples from the same particle)
        true_label = group['true_label'].iloc[0]
        
        # Apply majority voting for predictions
        pred_counts = group['pred_label'].value_counts()
        predicted_label = pred_counts.index[0]  # Most frequent prediction
        
        particle_results.append({
            'particle_id': particle_id,
            'true_label': true_label, 
            'predicted_label': predicted_label
        })
    
    # Convert to DataFrame
    particle_df = pd.DataFrame(particle_results)
    
    # Optional: Bootstrap sampling of particles for increased variability
    if bootstrap_particles and len(particle_df) > 10:
        n_particles = len(particle_df)
        bootstrap_indices = np.random.choice(n_particles, size=n_particles, replace=True)
        particle_df = particle_df.iloc[bootstrap_indices].reset_index(drop=True)
    
    # Calculate particle-level metrics
    y_true_particles = particle_df['true_label'].values
    y_pred_particles = particle_df['predicted_label'].values
    
    # Encode particle-level predictions for sklearn metrics
    y_true_particles_encoded = label_encoder.transform(y_true_particles)
    y_pred_particles_encoded = label_encoder.transform(y_pred_particles)
    
    # Calculate metrics on particle level
    particle_accuracy = accuracy_score(y_true_particles_encoded, y_pred_particles_encoded)
    particle_report = classification_report(y_true_particles_encoded, y_pred_particles_encoded, output_dict=True)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true_particles_encoded, y_pred_particles_encoded)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    # Get feature importance if available
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(label_encoder.classes_, model.feature_importances_))
    
    print(f"    Sample-level: {len(y_test)} samples, Accuracy: {sample_accuracy:.4f}")
    print(f"    Particle-level: {len(y_true_particles)} particles, Accuracy: {particle_accuracy:.4f}")
    
    return ModelResult(
        # Particle-level metrics (primary)
        accuracy=particle_accuracy,
        std_accuracy=0.0,  # Will be updated later
        precision=particle_report['weighted avg']['precision'],
        std_precision=0.0,
        recall=particle_report['weighted avg']['recall'],
        std_recall=0.0,
        f1=particle_report['weighted avg']['f1-score'],
        std_f1=0.0,
        class_accuracies=class_accuracies,
        std_class_accuracies=np.zeros_like(class_accuracies),
        confusion_matrix=cm,
        feature_importance=feature_importance,
        
        # Sample-level metrics (secondary)
        sample_accuracy=sample_accuracy,
        sample_precision=sample_report['weighted avg']['precision'],
        sample_recall=sample_report['weighted avg']['recall'],
        sample_f1=sample_report['weighted avg']['f1-score']
    )

def plot_results(results: Dict[str, ModelResult], output_dir: Path) -> None:
    """Plot and save evaluation results."""
    # Plot accuracy comparison
    plt.figure(figsize=(12, 6))
    models = list(results.keys())
    accuracies = [r.accuracy for r in results.values()]
    std_accuracies = [r.std_accuracy for r in results.values()]
    
    plt.bar(models, accuracies, yerr=std_accuracies)
    plt.xticks(rotation=45)
    plt.title('Model Accuracy Comparison (Particle-Level with Voting)')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(output_dir / 'plots' / 'accuracy_comparison.png')
    plt.close()
    
    # Plot confusion matrices
    for model_name, result in results.items():
        plt.figure(figsize=(10, 8))
        sns.heatmap(result.confusion_matrix, annot=True, fmt='.2f', cmap='Blues')
        plt.title(f'Average Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(output_dir / 'plots' / f'confusion_matrix_{model_name}.png')
        plt.close()

def run_experiment(seed: str, config: ModelConfig) -> Tuple[Dict[str, ModelResult], LabelEncoder, List[str]]:
    """Run a single experiment with the given seed."""
    print(f"\n--- Running experiment with seed: {seed} ---")
    
    # Load pre-existing data splits
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, test_particle_ids = load_data_from_splits(seed)
    
    # Scale data (fit on train, transform all)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Initialize models
    models = get_models(seed)
    results = {}
    
    # Train and evaluate each model (using test set for final evaluation)
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train_encoded)
        
        # Evaluate on test set with particle-level voting
        results[name] = evaluate_model_with_voting(model, X_test_scaled, y_test_encoded, test_particle_ids, label_encoder)
        
        # Also evaluate on validation set for monitoring (sample-level)
        val_result = evaluate_model(model, X_val_scaled, y_val_encoded, label_encoder)
        print(f"  Validation Accuracy (sample-level): {val_result.accuracy:.4f}")
        
        # Save model and scaler
        model_path = config.output_dir / 'models' / f'{name}_seed_{seed}.pkl'
        scaler_path = config.output_dir / 'models' / f'{name}_scaler_seed_{seed}.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    
    return results, label_encoder, feature_names

def get_available_seeds() -> List[str]:
    """Get list of available seed directories from material_classification_splits."""
    base_path = Path(__file__).parent.parent
    splits_path = base_path / "data" / "material_classification_splits"
    
    if not splits_path.exists():
        raise FileNotFoundError(f"Material classification splits directory not found: {splits_path}")
    
    # Get all seed directories
    seed_dirs = [d.name for d in splits_path.iterdir() if d.is_dir() and d.name.startswith('seed_')]
    seeds = [d.replace('seed_', '') for d in seed_dirs]
    
    print(f"Found {len(seeds)} seed directories: {sorted(seeds)}")
    return sorted(seeds)

def main():
    """Main function to run the training pipeline."""
    # Load configuration
    config = load_config()
    setup_directories(config)
    
    print(f"\nRunning experiments with pre-existing data splits...")
    
    # Get available seeds from material_classification_splits directory
    available_seeds = get_available_seeds()
    
    # Limit to config.num_seeds if specified, otherwise use all available
    if config.num_seeds > 0 and config.num_seeds < len(available_seeds):
        seeds_to_use = available_seeds[:config.num_seeds]
        print(f"Using first {config.num_seeds} seeds: {seeds_to_use}")
    else:
        seeds_to_use = available_seeds
        print(f"Using all {len(seeds_to_use)} available seeds")
    
    # Run experiments (each with pre-existing data splits)
    all_results = {}
    label_encoder = None
    feature_names = None
    
    for seed in seeds_to_use:
        seed_results, label_encoder, feature_names = run_experiment(seed, config)
        
        # Aggregate results
        for model_name, result in seed_results.items():
            if model_name not in all_results:
                all_results[model_name] = []
            all_results[model_name].append(result)
    
    print(f"\nDataset Information (from last experiment):")
    print(f"Feature names: {feature_names}")
    
    # Calculate average results
    final_results = {}
    for model_name, results in all_results.items():
        final_results[model_name] = ModelResult(
            # Particle-level metrics
            accuracy=np.mean([r.accuracy for r in results]),
            std_accuracy=np.std([r.accuracy for r in results]),
            precision=np.mean([r.precision for r in results]),
            std_precision=np.std([r.precision for r in results]),
            recall=np.mean([r.recall for r in results]),
            std_recall=np.std([r.recall for r in results]),
            f1=np.mean([r.f1 for r in results]),
            std_f1=np.std([r.f1 for r in results]),
            class_accuracies=np.mean([r.class_accuracies for r in results], axis=0),
            std_class_accuracies=np.std([r.class_accuracies for r in results], axis=0),
            confusion_matrix=np.mean([r.confusion_matrix for r in results], axis=0),
            feature_importance=results[0].feature_importance,
            
            # Sample-level metrics
            sample_accuracy=np.mean([r.sample_accuracy for r in results]),
            sample_std_accuracy=np.std([r.sample_accuracy for r in results]),
            sample_precision=np.mean([r.sample_precision for r in results]),
            sample_std_precision=np.std([r.sample_precision for r in results]),
            sample_recall=np.mean([r.sample_recall for r in results]),
            sample_std_recall=np.std([r.sample_recall for r in results]),
            sample_f1=np.mean([r.sample_f1 for r in results]),
            sample_std_f1=np.std([r.sample_f1 for r in results])
        )
    
    # Plot and save results
    plot_results(final_results, config.output_dir)
    
    # Print final results
    print("\n" + "="*70)
    print(f"AVERAGE RESULTS ACROSS {len(seeds_to_use)} SEEDS: {seeds_to_use}")
    print("="*70)
    
    for model_name, result in final_results.items():
        # Debug: Show why std is zero
        model_results = all_results[model_name]
        particle_accuracies = [r.accuracy for r in model_results]
        sample_accuracies = [r.sample_accuracy for r in model_results]
        
        print(f"\n{model_name}:")
        print(f"  DEBUG INFO:")
        print(f"    Particle accuracies across seeds: {particle_accuracies}")
        print(f"    Sample accuracies across seeds: {sample_accuracies[:5]}...")  # Show first 5
        
        print(f"  PARTICLE-LEVEL METRICS:")
        print(f"    Accuracy: {result.accuracy:.4f} ± {result.std_accuracy:.4f}")
        print(f"    Precision: {result.precision:.4f} ± {result.std_precision:.4f}")
        print(f"    Recall: {result.recall:.4f} ± {result.std_recall:.4f}")
        print(f"    F1 Score: {result.f1:.4f} ± {result.std_f1:.4f}")
        
        print(f"  SAMPLE-LEVEL METRICS:")
        print(f"    Accuracy: {result.sample_accuracy:.4f} ± {result.sample_std_accuracy:.4f}")
        print(f"    Precision: {result.sample_precision:.4f} ± {result.sample_std_precision:.4f}")
        print(f"    Recall: {result.sample_recall:.4f} ± {result.sample_std_recall:.4f}")
        print(f"    F1 Score: {result.sample_f1:.4f} ± {result.sample_std_f1:.4f}")
        
        print(f"  CLASS-WISE ACCURACY (Particle-level):")
        for idx, (avg_acc, std_acc) in enumerate(zip(result.class_accuracies, result.std_class_accuracies)):
            class_name = label_encoder.inverse_transform([idx])[0]
            print(f"    {class_name}: {avg_acc:.4f} ± {std_acc:.4f}")

if __name__ == "__main__":
    main()