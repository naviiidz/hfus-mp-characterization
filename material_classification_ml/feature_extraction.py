import numpy as np
from scipy.stats import kurtosis, skew, entropy

# Function to extract time-domain features
def fe_time(signal):
    """
    Extract time-domain features from a signal.
    
    Parameters:
    - signal: np.ndarray
        The time-domain signal.
        
    Returns:
    - features: dict
        Dictionary containing time-domain features.
    """
    # Ensure signal is a numpy array
    signal = np.array(signal)
    
    # Maximal Magnitude
    max_magnitude = np.max(np.abs(signal))
    
    # Temporal Energy (sum of squares)
    temporal_energy = np.sum(signal**2)
    
    # Temporal Entropy
    # Normalize signal for probability distribution
    signal_abs = np.abs(signal)
    if np.sum(signal_abs) > 0:  # Avoid division by zero
        signal_norm = signal_abs / np.sum(signal_abs)
        # Remove zeros to avoid log(0)
        signal_norm = signal_norm[signal_norm > 0]
        temporal_entropy = -np.sum(signal_norm * np.log2(signal_norm))
    else:
        temporal_entropy = 0
    
    # Temporal Crest Factor (ratio of peak to RMS)
    rms = np.sqrt(np.mean(signal**2))
    temporal_crest_factor = max_magnitude / rms if rms > 0 else 0
    
    return {
        'max_magnitude': max_magnitude,
        'temporal_energy': temporal_energy,
        'temporal_entropy': temporal_entropy,
        'temporal_crest_factor': temporal_crest_factor
    }

# Function to extract frequency-domain features
def fe_freq(signal_freq):
    """
    Extract frequency-domain features from a signal.
    
    Parameters:
    - signal_freq: np.ndarray
        The frequency-domain signal (magnitude spectrum).
        
    Returns:
    - features: dict
        Dictionary containing frequency-domain features.
    """
    # Ensure signal is a numpy array
    signal_freq = np.array(signal_freq)
    n = len(signal_freq)
    
    # Normalized frequency axis (0 to 1)
    freqs = np.linspace(0, 1, n)
    
    # Basic statistics (already calculated in the original code)
    mean = np.mean(signal_freq)
    std = np.std(signal_freq)
    max_val = np.max(signal_freq)
    min_val = np.min(signal_freq)
    kurt = kurtosis(signal_freq)
    skewness = skew(signal_freq)
    
    # Spectral Energy
    spectral_energy = np.sum(signal_freq**2)
    
    # Spectral Entropy
    # Normalize signal for probability distribution
    if np.sum(signal_freq) > 0:  # Avoid division by zero
        signal_norm = signal_freq / np.sum(signal_freq)
        # Remove zeros to avoid log(0)
        signal_norm = signal_norm[signal_norm > 0]
        spectral_entropy = -np.sum(signal_norm * np.log2(signal_norm))
    else:
        spectral_entropy = 0
    
    # Spectral Crest Factor (ratio of peak to RMS)
    rms = np.sqrt(np.mean(signal_freq**2))
    spectral_crest_factor = max_val / rms if rms > 0 else 0
    
    # Spectral Centroid (weighted mean of frequencies)
    if np.sum(signal_freq) > 0:  # Avoid division by zero
        spectral_centroid = np.sum(freqs * signal_freq) / np.sum(signal_freq)
    else:
        spectral_centroid = 0
    
    # Spectral Spread (variance around centroid)
    if np.sum(signal_freq) > 0:  # Avoid division by zero
        spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * signal_freq) / np.sum(signal_freq))
    else:
        spectral_spread = 0
    
    # Bandwidth (frequency range containing most energy)
    # Using percentile approach for simplicity
    cumulative_energy = np.cumsum(signal_freq**2)
    normalized_energy = cumulative_energy / cumulative_energy[-1] if cumulative_energy[-1] > 0 else cumulative_energy
    bandwidth = np.sum(normalized_energy < 0.95) / n  # 95% of energy
    
    return {
        'mean': mean,
        'std': std,
        'max': max_val,
        'min': min_val,
        'kurtosis': kurt,
        'skewness': skewness,
        'bandwidth': bandwidth,
        'spectral_energy': spectral_energy,
        'spectral_entropy': spectral_entropy,
        'spectral_crest_factor': spectral_crest_factor,
        'spectral_centroid': spectral_centroid,
        'spectral_spread': spectral_spread
    }

def cal_corr(features):
    """
    Computes the correlation matrix of the given feature set.

    Parameters:
    - features: np.ndarray
        The dataset containing the features (after preprocessing).

    Returns:
    - corr_matrix: np.ndarray
        The computed correlation matrix.
    """
    # Compute the correlation matrix
    corr_matrix = np.corrcoef(features, rowvar=False)

    # Display the correlation matrix
    print("Feature Correlation Matrix:")
    print(corr_matrix)

    return corr_matrix

def show_corr(data, features, target_column=None, method='pearson'):
    """
    Computes and visualizes the correlation matrix of features.

    Parameters:
    - data: pd.DataFrame
        The dataset containing features and target.
    - features: list of str
        List of feature column names to include in the correlation matrix.
    - target_column: str, optional
        Name of the target column to include in the correlation analysis (default: None).
    - method: str
        Correlation method to use: 'pearson', 'spearman', or 'kendall' (default: 'pearson').

    Returns:
    - corr_matrix: pd.DataFrame
        The computed correlation matrix.
    """
    if target_column:
        # Include the target column in the correlation analysis
        features = features + [target_column]

    # Compute correlation matrix
    corr_matrix = data[features].corr(method=method)

    # Plot the correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
    plt.title(f'Feature Correlation ({method.title()} Method)')
    plt.show()

    return corr_matrix