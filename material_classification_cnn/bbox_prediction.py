'''
Navid.Zarrabi@torontomu.ca
October 25, 2024
Goal: Testing a network using stored model for material identification
'''

import numpy as np
import pandas as pd
import os
import sys
import contextlib
from io import StringIO
import json
# Force TensorFlow to use CPU to avoid CUDA/cuDNN compatibility issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from preprocessing import process_signal, prepare_balanced_dataset
from config import get_training_config, preprocessing_parameters, predict_config, peak_extract_param
from peak_extract import find_prominent_maxima, get_dataset_params
import h5py
import os
import time


import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Set matplotlib backend - try to detect best option
import matplotlib
import os

def setup_matplotlib_backend():
    """Setup the best available matplotlib backend."""
    # Check if we have a DISPLAY environment variable (for X11)
    if os.environ.get('DISPLAY'):
        try:
            matplotlib.use('TkAgg')
            print("Using TkAgg backend - interactive display enabled")
            return True
        except:
            pass
    
    # Try Qt backend
    try:
        matplotlib.use('Qt5Agg')
        print("Using Qt5Agg backend - interactive display enabled") 
        return True
    except:
        pass
        
    # Fall back to non-interactive
    matplotlib.use('Agg')
    print("Using Agg backend - saving plots only (no interactive display)")
    return False

# Setup backend and check if interactive
interactive_available = setup_matplotlib_backend()
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter
import matplotlib.patches as patches
from collections import Counter



# Configuration
# Load preprocessing parameters
params = preprocessing_parameters()
SAMPLING_RATE = params["sampling_rate"]
LOWCUT_FREQ = params["lowcut_freq"]
HIGHCUT_FREQ = params["highcut_freq"]
SIGNAL_LENGTH = params["signal_length"]
SAMPLES_PER_CLASS = params["samples_per_class"]



def extract_neighborhood_signals(RF, coord, neighborhood_size=2):
    """Extract signals from a neighborhood around a coordinate for voting."""
    signals = []
    coords = []
    
    center_x, center_y = coord
    half_size = neighborhood_size // 2
    
    for dx in range(-half_size, half_size + 1):
        for dy in range(-half_size, half_size + 1):
            x = center_x + dx
            y = center_y + dy
            
            # Check bounds
            if 0 <= x < RF.shape[1] and 0 <= y < RF.shape[2]:
                signal = RF[:, x, y]
                signals.append(signal)
                coords.append((x, y))
    
    return signals, coords

def apply_voting_prediction(RF, coord, model, neighborhood_size=3):
    """Apply voting by extracting neighborhood signals and taking majority vote."""
    # Extract neighborhood signals
    neighborhood_signals, neighborhood_coords = extract_neighborhood_signals(
        RF, coord, neighborhood_size)
    
    if not neighborhood_signals:
        return None, None, 0
    
    # Process all neighborhood signals
    processed_signals = []
    for signal in neighborhood_signals:
        try:
            processed_signal = process_signal(signal)
            processed_signals.append(processed_signal)
        except Exception:
            # Skip invalid signals
            continue
    
    if not processed_signals:
        return None, None, 0
    
    # Prepare signals for prediction
    padded_signals = pad_sequences(processed_signals, maxlen=SIGNAL_LENGTH,
                                 dtype='float32', padding='post',
                                 truncating='post')
    padded_signals = np.expand_dims(padded_signals, axis=2)
    
    # Make predictions
    predictions = model.predict(padded_signals, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    
    # Apply majority voting
    vote_counts = Counter(predicted_classes)
    voted_class = vote_counts.most_common(1)[0][0]
    vote_strength = vote_counts[voted_class] / len(predicted_classes)
    
    # Calculate average confidence for the voted class
    voted_class_confidences = [conf for cls, conf in zip(predicted_classes, confidences) 
                              if cls == voted_class]
    avg_confidence = np.mean(voted_class_confidences) if voted_class_confidences else 0
    
    return voted_class, avg_confidence, vote_strength

def plot_results(max_amplitude_array, maxima_coords, decoded_labels, category=None, file_name=None, save_path=None):
    """Plot the results with material identification boxes and labels."""
   
    
    # Get box size from file-specific parameters
    params = get_dataset_params(category, file_name)
    box_size = params["box_size"]

    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    heatmap = sns.heatmap(max_amplitude_array, cmap='viridis', cbar=True, 
                         cbar_kws={'label': 'Amplitude'}, ax=ax)
    
    # Set colorbar label font size
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('Amplitude', fontsize=30)
    
    plt.title(f"Material Identification Results - {category}/{file_name}", fontsize=20)
    plt.xlabel("Y Index", fontsize=20)
    plt.ylabel("X Index", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Color mapping for different materials
    color_map = {
        'glass': 'cyan',
        'pe': 'yellow', 
        'steel': 'red',
        'pmma': 'magenta'
    }

    for i, (coord, label_) in enumerate(zip(maxima_coords, decoded_labels)):
        # Get color for this material
        edge_color = color_map.get(label_.lower(), 'white')
        
        # Add bounding box
        bbox = patches.Rectangle(
            (coord[1] - box_size / 2, coord[0] - box_size / 2),
            box_size, box_size,
            linewidth=2, edgecolor=edge_color, facecolor='none'
        )
        ax.add_patch(bbox)

        # Add label with background
        text_x = coord[1]
        text_y = coord[0] - box_size / 2 - 2
        plt.text(text_x, text_y, f"{label_}", color='white', fontsize=20,
                ha='center', va='bottom', weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))

    # legend_elements = [patches.Patch(color=color, label=material.capitalize()) 
    #                   for material, color in color_map.items()]
    # plt.legend(handles=legend_elements, loc='upper left', fontsize=12,
    #            bbox_to_anchor=(0.02, 0.98), frameon=True, fancybox=True, shadow=True)

    # Save the plot
    if save_path is None:
        os.makedirs("output", exist_ok=True)
        save_path = os.path.join("output", f"prediction_results_{category}_{file_name}.png")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Results visualization saved to: {save_path}")
    

    # Close the figure to free memory
    plt.close()

def predict_materials(mat_file_path, model_path, encoder_path, category=None, file_name=None):
    """Predict material types from a .mat file using a trained CNN model."""
    try:
        # Load the data
        with h5py.File(mat_file_path, 'r') as file:
            RF = np.array(file['RFdata']['RF']).transpose()

        # Compute max amplitude array
        max_amplitude_array = np.max(RF, axis=0)

        # Find prominent maxima using file-specific parameters (disable threshold analysis)
        maxima_coords, maxima_values = find_prominent_maxima(max_amplitude_array, category, file_name, save_threshold_analysis=False)

        # Load model and label encoder
        model = tf.keras.models.load_model(model_path)
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.load(encoder_path, allow_pickle=True)

        # Process signals at maxima locations
        maxima_signals = []
        signal_processing_times = []
        
        print(f"Processing {len(maxima_coords)} signals...")
        for i, coord in enumerate(maxima_coords):
            start_time = time.time()
            signal = RF[:, coord[0], coord[1]]
            processed_signal = process_signal(signal)
            maxima_signals.append(processed_signal)
            processing_time = time.time() - start_time
            signal_processing_times.append(processing_time)
            
            # Print progress every 50 signals
            if (i + 1) % 50 == 0 or (i + 1) == len(maxima_coords):
                print(f"  Processed {i + 1}/{len(maxima_coords)} signals...")

        # Prepare signals for prediction
        print("Preparing signals for prediction...")
        prep_start_time = time.time()
        padded_signals = pad_sequences(maxima_signals, maxlen=SIGNAL_LENGTH,
                                     dtype='float32', padding='post',
                                     truncating='post')
        padded_signals = np.expand_dims(padded_signals, axis=2)
        prep_time = time.time() - prep_start_time

        # Make predictions
        print("Running CNN predictions...")
        prediction_start_time = time.time()
        predictions = model.predict(padded_signals)
        predicted_labels = np.argmax(predictions, axis=1)
        decoded_labels = label_encoder.inverse_transform(predicted_labels)
        prediction_time = time.time() - prediction_start_time

        # Plot results with file-specific parameters
        os.makedirs("output/material_classification/dl", exist_ok=True)
        save_path = os.path.join("output", f"material_classification/dl/prediction_results_{category}_{file_name}.png")
        plot_results(max_amplitude_array, maxima_coords, decoded_labels, category, file_name, save_path)

        # Print predictions with confidence
        print("\nPrediction Results:")
        print("=" * 35)
        print(f"Found {len(decoded_labels)} microspheres")
        for i, (label, coord) in enumerate(zip(decoded_labels, maxima_coords)):
            confidence = np.max(predictions[i]) * 100
            print(f"Microsphere {i+1}: {label} at position ({coord[0]}, {coord[1]}) "
                  f"with {confidence:.1f}% confidence")

        # Count occurrences of each material type
        from collections import Counter
        label_counts = Counter(decoded_labels)
        
        print(f"\n{'='*35}")
        print("MATERIAL DETECTION SUMMARY")
        print(f"{'='*35}")
        print(f"Total microspheres detected: {len(decoded_labels)}")
        print("\nMaterial breakdown:")
        for material, count in sorted(label_counts.items()):
            percentage = (count / len(decoded_labels)) * 100
            print(f"  {material.upper()}: {count} microspheres ({percentage:.1f}%)")
        print(f"{'='*35}")

        # Print timing statistics
        print(f"\n{'='*50}")
        print("TIMING STATISTICS")
        print(f"{'='*50}")
        print(f"Total signals processed: {len(maxima_coords)}")
        print(f"Signal processing times:")
        print(f"  Min time per signal: {np.min(signal_processing_times)*1000:.2f} ms")
        print(f"  Max time per signal: {np.max(signal_processing_times)*1000:.2f} ms") 
        print(f"  Average time per signal: {np.mean(signal_processing_times)*1000:.2f} ms")
        print(f"  Total signal processing time: {np.sum(signal_processing_times):.3f} s")
        print(f"Data preparation time: {prep_time:.3f} s")
        print(f"CNN prediction time (batch): {prediction_time:.3f} s")
        print(f"Average CNN prediction per signal: {prediction_time/len(maxima_coords)*1000:.2f} ms")
        
        total_time = np.sum(signal_processing_times) + prep_time + prediction_time
        print(f"Total prediction pipeline time: {total_time:.3f} s")
        print(f"Throughput: {len(maxima_coords)/total_time:.1f} signals/second")
        print(f"{'='*50}")

        return decoded_labels, maxima_coords, predictions

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None, None, None

# Define available datasets by category
DATASETS = peak_extract_param()


def get_dataset_path(category, file_name, root_dir):
    """Get the full path for a dataset file."""
    try:
        if category not in DATASETS:
            raise ValueError(f"Category '{category}' not found in available datasets")

        if file_name not in DATASETS[category]['files']:
            raise ValueError(f"File '{file_name}' not found in category '{category}'")

        return os.path.join(root_dir, DATASETS[category]['files'][file_name]['path'])

    except Exception as e:
        print(f"Error getting dataset path: {str(e)}")
        return None

def list_available_datasets():
    """Print available datasets and their files with parameters."""
    print("\nAvailable Datasets:")
    print("==================")
    for category, data in DATASETS.items():
        print(f"\n{category}")
        print("-" * len(category))
        print(f"Description: {data['description']}")
        print("\nFiles:")
        for name, file_info in data['files'].items():
            print(f"\n  - {name}:")
            print(f"    Path: {file_info['path']}")
            print("    Parameters:")
            for param_name, param_value in file_info['params'].items():
                print(f"      {param_name}: {param_value}")

def main():
    """Example usage of the prediction function."""
    # File paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(root_dir)
    model_path = os.path.join(root_dir, "models/final_model.h5")
    encoder_path = os.path.join(parent_dir, "data/label_encoder_classes.npy")
    root_dir=os.path.join(parent_dir, "data")

    # List available datasets with their parameters
    list_available_datasets()

    # Example: predict materials in a steel sample
    category = "2024_10_25_PURE"
    file_name = "steel_1"

    print(f"\nProcessing {category}/{file_name}...")
    mat_file = get_dataset_path(category, file_name, root_dir)

    if mat_file:
        print("Processing file:", mat_file)
        # Pass both category and file_name to use file-specific parameters
        predict_materials(mat_file, model_path, encoder_path, category, file_name)

if __name__ == "__main__":
    main()