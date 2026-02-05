import numpy as np
from scipy.signal import hilbert
from config import preprocessing_parameters

# Load preprocessing parameters when needed
params = preprocessing_parameters()

def process_signal(signal):
    # 1) Remove DC offset
    signal = signal - np.mean(signal)

    # 2) Window to reduce spectral leakage
    window = np.hanning(len(signal))
    signal = signal * window

    # 3) Standardize (avoid division by zero)
    signal = signal / (np.std(signal) + 1e-8)

    # 4) FFT (real FFT is enough for real-valued signal)
    fft_result = np.fft.rfft(signal)

    # 5) Magnitude spectrum (real-valued)
    magnitude = np.abs(fft_result)

    # 6) Normalize magnitude to [0, 1]
    magnitude = magnitude / (np.max(magnitude) + 1e-8)

    return magnitude.astype(np.float32)


def prepare_balanced_dataset(signals, labels):
    """Create a balanced dataset with equal samples per class."""
    try:
        unique_labels = np.unique(labels)
        balanced_signals = []
        balanced_labels = []

        print(f"Found {len(unique_labels)} unique labels: {unique_labels}")
        print(f"Initial dataset size: {len(signals)} signals")

        # First, find the maximum length among all signals
        max_len = max(len(signal) for signal in signals)
        print(f"Maximum signal length: {max_len}")

        for label in unique_labels:
            indices = np.where(labels == label)[0]
            print(f"\nProcessing label: {label}")
            print(f"Found {len(indices)} samples")

            if len(indices) >= params["samples_per_class"]:
                selected_indices = np.random.choice(indices, params["samples_per_class"], replace=False)
                print(f"Selected {params['samples_per_class']} samples")
            else:
                print(f"Warning: Only {len(indices)} samples available for {label}")
                selected_indices = indices

            # Process and pad each selected signal
            for idx in selected_indices:
                if not isinstance(signals[idx], np.ndarray):
                    print(f"Warning: Invalid signal at index {idx}")
                    continue

                # Pad signal if necessary
                signal = signals[idx]
                if len(signal) < max_len:
                    padded_signal = np.pad(signal,
                                         (0, max_len - len(signal)),
                                         mode='constant',
                                         constant_values=0)
                else:
                    padded_signal = signal

                balanced_signals.append(padded_signal)
                balanced_labels.append(label)

        # Convert to numpy arrays
        balanced_signals = np.stack(balanced_signals)
        balanced_labels = np.array(balanced_labels)

        print(f"\nFinal balanced dataset size: {len(balanced_signals)} signals")
        print(f"Signal shape: {balanced_signals.shape}")

        return balanced_signals, balanced_labels

    except Exception as e:
        print(f"Error in prepare_balanced_dataset: {str(e)}")
        print("Detailed signal information:")
        print(f"Number of signals: {len(signals)}")
        print("Signal lengths:", [len(s) for s in signals[:5]], "... (first 5 shown)")
        print(f"Labels shape: {labels.shape}")
        raise