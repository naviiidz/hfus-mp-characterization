"""
Author: Navid Zarrabi (Navid.Zarrabi@torontomu.ca)
Date: May 26, 2025
Purpose: Train a CNN model for material identification of microspheres.

Key Notes:
1. Enhanced signal processing for improved material discrimination.
2. Balanced dataset creation with equal samples per class.
3. Proper train/test split to prevent data leakage.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from preprocessing import process_signal
from model import create_model
from visualize import plot_training_history, plot_confusion_matrix
from config import get_training_config, preprocessing_parameters
import os

def configure_gpu():
    """Configure TensorFlow for GPU usage with memory growth."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for GPU(s)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"🚀 GPU Configuration: Found {len(gpus)} GPU(s) - Memory growth enabled")
            print(f"   GPU Device(s): {[gpu.name for gpu in gpus]}")
            
            # Set mixed precision policy for better performance
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("   Mixed precision (float16) enabled for better performance")
            
            return True
        except RuntimeError as e:
            print(f"⚠️  GPU configuration error: {e}")
            return False
    else:
        print("❌ No GPU detected - training will use CPU")
        return False

# Load preprocessing parameters
params = preprocessing_parameters()
SAMPLING_RATE = params["sampling_rate"]
LOWCUT_FREQ = params["lowcut_freq"]
HIGHCUT_FREQ = params["highcut_freq"]
SIGNAL_LENGTH = params["signal_length"]
SAMPLES_PER_CLASS = params["samples_per_class"]


def main():
    """Train the CNN model for material identification."""
    # Configure GPU
    configure_gpu()
    
    # File paths
    # Get the directory of the current script
    root_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(root_dir)

    # Use the pre-split CSV files instead of splitting in code
    train_csv = os.path.join(parent_dir, "data", "train_data.csv")
    val_csv = os.path.join(parent_dir, "data", "val_data.csv")
    test_csv = os.path.join(parent_dir, "data", "test_data.csv")
    model_path = os.path.join(root_dir, "final_model.h5")
    encoder_path = os.path.join(parent_dir, "data", "label_encoder_classes.npy")

    # Verify paths
    print("\nVerifying paths...")
    print(f"Root directory exists: {os.path.exists(root_dir)}")
    print(f"Train CSV exists: {os.path.exists(train_csv)}")
    print(f"Val CSV exists: {os.path.exists(val_csv)}")
    print(f"Test CSV exists: {os.path.exists(test_csv)}")
    print(f"Current working directory: {os.getcwd()}")

    try:
        # Load pre-split datasets
        print("\nLoading pre-split datasets...")
        train_data = pd.read_csv(train_csv)
        val_data = pd.read_csv(val_csv)
        test_data = pd.read_csv(test_csv)
        
        print(f"Training set: {len(train_data)} samples")
        print(f"Validation set: {len(val_data)} samples") 
        print(f"Test set: {len(test_data)} samples")
        print("Columns:", train_data.columns.tolist())

        # Process each dataset separately
        def process_dataset(data, dataset_name):
            print(f"\nProcessing {dataset_name} dataset...")
            raw_signals = []
            signal_lengths = []
            
            for i, signal_str in enumerate(data['Signal'].values):
                try:
                    signal = np.fromstring(signal_str, sep=',')
                    raw_signals.append(signal)
                    signal_lengths.append(len(signal))
                except Exception as e:
                    print(f"Error converting {dataset_name} signal {i}: {str(e)}")
                    continue

            print(f"{dataset_name} signal statistics:")
            print(f"  Min length: {min(signal_lengths)}")
            print(f"  Max length: {max(signal_lengths)}")
            print(f"  Mean length: {np.mean(signal_lengths):.2f}")
            
            # Process signals
            processed_signals = []
            for i, signal in enumerate(raw_signals):
                try:
                    processed = process_signal(signal)
                    processed_signals.append(processed)
                    if i % 1000 == 0 and i > 0:
                        print(f"  Processed {i}/{len(raw_signals)} signals")
                except Exception as e:
                    print(f"Error processing {dataset_name} signal {i}: {str(e)}")
                    continue
            
            print(f"{dataset_name} processed signals: {len(processed_signals)}")
            return processed_signals, data['Material Type'].values

        # Process all three datasets
        train_signals, train_labels = process_dataset(train_data, "Training")
        val_signals, val_labels = process_dataset(val_data, "Validation")  
        test_signals, test_labels = process_dataset(test_data, "Test")

        print(f"\nUnique material types: {np.unique(train_labels)}")
        
        # Verify we have matching lengths
        assert len(train_signals) == len(train_labels), "Training signals and labels length mismatch"
        assert len(val_signals) == len(val_labels), "Validation signals and labels length mismatch"
        assert len(test_signals) == len(test_labels), "Test signals and labels length mismatch"

        print("\nPreparing signals for training...")
        padded_train = pad_sequences(train_signals, maxlen=SIGNAL_LENGTH,
                                   dtype='float32', padding='post', truncating='post')
        padded_val = pad_sequences(val_signals, maxlen=SIGNAL_LENGTH,
                                 dtype='float32', padding='post', truncating='post')
        padded_test = pad_sequences(test_signals, maxlen=SIGNAL_LENGTH,
                                  dtype='float32', padding='post', truncating='post')

        padded_train = np.expand_dims(padded_train, axis=2)
        padded_val = np.expand_dims(padded_val, axis=2)
        padded_test = np.expand_dims(padded_test, axis=2)

        print(f"Training data shape: {padded_train.shape}")
        print(f"Validation data shape: {padded_val.shape}")
        print(f"Testing data shape: {padded_test.shape}")

        # Encode labels
        print("\nEncoding labels...")
        label_encoder = LabelEncoder()
        
        # Fit label encoder on all labels to ensure consistency
        all_labels = np.concatenate([train_labels, val_labels, test_labels])
        label_encoder.fit(all_labels)
        
        train_labels_encoded = label_encoder.transform(train_labels)
        val_labels_encoded = label_encoder.transform(val_labels)
        test_labels_encoded = label_encoder.transform(test_labels)

        categorical_train = to_categorical(train_labels_encoded)
        categorical_val = to_categorical(val_labels_encoded)
        categorical_test = to_categorical(test_labels_encoded)
        
        print(f"Label classes: {label_encoder.classes_}")
        print(f"Training label distribution: {np.bincount(train_labels_encoded)}")
        print(f"Validation label distribution: {np.bincount(val_labels_encoded)}")
        print(f"Test label distribution: {np.bincount(test_labels_encoded)}")

        # Save label encoder classes
        np.save(encoder_path, label_encoder.classes_)
        print(f"\nLabel encoder classes saved to {encoder_path}")

        # Create and train model
        print("\nCreating model...")
        model = create_model(
            input_shape=(SIGNAL_LENGTH, 1),
            num_classes=len(label_encoder.classes_)
        )

        # Get training configuration
        training_config = get_training_config()
        print("shuffle in training_config:", training_config.get("shuffle", "DEFAULT(True for NumPy)"))

        print("\nTraining model...")
        history = model.fit(
            padded_train,
            categorical_train,
            validation_data=(padded_val, categorical_val),
            **training_config
        )

        # Save model
        model.save(model_path)
        print(f"\nModel saved to {model_path}")

        # Evaluate model
        print("\nEvaluating model...")
        test_loss, test_accuracy = model.evaluate(padded_test, categorical_test, verbose=0)
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        # Generate predictions
        test_predictions = model.predict(padded_test)
        predicted_labels = np.argmax(test_predictions, axis=1)

        # Plot results - commented out for repeated runs
        # plot_training_history(history)
        # plot_confusion_matrix(test_labels_encoded, predicted_labels,
        #                     label_encoder.classes_)

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(test_labels_encoded, predicted_labels,
                                 target_names=label_encoder.classes_,
                                 digits=4))

    except Exception as e:
        print("\nError during training:")
        print(str(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()