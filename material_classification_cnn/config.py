import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping


def preprocessing_parameters():
    """Return preprocessing parameters."""
    SAMPLING_RATE = 40e6
    LOWCUT_FREQ = 1e6
    HIGHCUT_FREQ = 20e6
    SIGNAL_LENGTH = 3271
    SAMPLES_PER_CLASS = 1000

    return {
        "sampling_rate": SAMPLING_RATE,
        "lowcut_freq": LOWCUT_FREQ,
        "highcut_freq": HIGHCUT_FREQ,
        "signal_length": SIGNAL_LENGTH,
        "samples_per_class": SAMPLES_PER_CLASS,
    }



def visualization_settings():
    """Apply visualization settings."""
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.titleweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold'
    })



def get_training_config():
    """Return training configuration."""
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True
    )
    training_params = {
        "epochs": 100,
        "batch_size": 20,
        "callbacks": [early_stopping],
        "verbose": 1
    }
    return training_params


def predict_config():
    # Configuration
    return {
        "sampling_rate": 40e6,  # Hz
        "lowcut_freq": 1e6,     # Hz
        "highcut_freq": 20e6,   # Hz
        "window_size": 3,       # Default window size
        "alpha": 0.8,           # Default alpha
        "sigma": 3,             # Default sigma
        "box_size": 10          # Default box size
    }

def peak_extract_param():
    # Define available datasets by category
    DATASETS = {
        "2023_10_04_SAMPLES": {
            "description": "PMMA & PE pure samples without size variations",
            "files": {
                "pmma_50um": {
                    "path": "raw_data/20231004_pe_pmma/PMMA50um_40MHz_2200x2000x12_5umB.mat",
                    "params": {
                        "window_size": 5,
                        "alpha": 0.65,
                        "sigma": 1.5,
                        "box_size": 10
                    }
                },
                "pe_50um_1": {
                    "path": "raw_data/20231004_pe_pmma/PE50um_40MHz_1800x1400x12_5um.mat",
                    "params": {
                        "window_size": 5,
                        "alpha": 0.7,
                        "sigma": 1.8,
                        "box_size": 10
                    }
                },
                "pe_50um_2": {
                    "path": "raw_data/20231004_pe_pmma/PE50um_40MHz_2200x2000x12_5um.mat",
                    "params": {
                        "window_size": 5,
                        "alpha": 0.7,
                        "sigma": 1.8,
                        "box_size": 10
                    }
                }
            }
        },
        "2024_04_17_SAMPLES": {
            "description": "Glass, PE, Steel, PMMA pure samples with size variations",
            "files": {
                "glass_50um_small": {
                    "path": "raw_data/20240417_glass_pe_steel/40MHz_glass_300x300x10um.mat",
                    "params": {
                        "window_size": 5,
                        "alpha": 0.8,
                        "sigma": 2.0,
                        "box_size": 12
                    }
                },
                "glass_50um_large": {
                    "path": "raw_data/20240417_glass_pe_steel/40MHz_glass_1000x1000x20um.mat",
                    "params": {
                        "window_size": 5,
                        "alpha": 0.5,
                        "sigma": 3,
                        "box_size": 8
                    }
                },
                "pe_50um_single": {
                    "path": "raw_data/20240417_glass_pe_steel/40MHz_polye_300x300x10um.mat",
                    "params": {
                        "window_size": 20,
                        "alpha": 0.7,
                        "sigma": 1.8,
                        "box_size": 21
                    }
                },
                "pe_50um_multi": {
                    "path": "raw_data/20240417_glass_pe_steel/40MHz_polye_1200x1200x20um.mat",
                    "params": {
                        "window_size": 5,
                        "alpha": 0.5,
                        "sigma": 1.0,
                        "box_size": 6
                    }
                },
                "steel_50um_single": {
                    "path": "raw_data/20240417_glass_pe_steel/40MHz_steel_1000x1000x20um.mat",
                    "params": {
                        "window_size": 7,
                        "alpha": 0.85,
                        "sigma": 2.5,
                        "box_size": 14
                    }
                },
                "steel_50um_multi": {
                    "path": "raw_data/20240417_glass_pe_steel/40MHz_steel_1000x1000x20um.mat",
                    "params": {
                        "window_size": 6,
                        "alpha": 0.85,
                        "sigma": 2.5,
                        "box_size": 6
                    }
                },
                "pmma_20_80": {
                    "path": "raw_data/20250409_pmma/40MHz_PMMA20-80_3200x3000x10um.mat",
                    "params": {
                        "window_size": 7,
                        "alpha": 0.7,
                        "sigma": 2.0,
                        "box_size": 12
                    }
                }
            }
        },
        "2024_10_25_SAMPLES": {
            "description": "Glass, PE, and Steel pure samples with size variations",
            "files": {
                "glass": {
                    "path": "raw_data/20241025_glass_pe_steel/40MHz_glass_3200x3000x10um.mat",
                    "params": {
                        "window_size": 9,
                        "alpha": 0.8,
                        "sigma": 2.5,
                        "box_size": 16
                    }
                },
                "pe": {
                    "path": "raw_data/20241025_glass_pe_steel/40MHz_pe_3200x3000x10um.mat",
                    "params": {
                        "window_size": 7,
                        "alpha": 0.8,
                        "sigma": 1.0,
                        "box_size": 14
                    }
                },
                "steel_1": {
                    "path": "raw_data/20241025_glass_pe_steel/40MHz_steel_1800x1500x10um.mat",
                    "params": {
                        "window_size": 4,
                        "alpha": 0.85,
                        "sigma": 1.5,
                        "box_size": 10
                    }
                },
                "steel_2": {
                    "path": "raw_data/20241025_glass_pe_steel/40MHz_steel2_3200x3000x10um.mat",
                    "params": {
                        "window_size": 9,
                        "alpha": 0.85,
                        "sigma": 2.8,
                        "box_size": 16
                    }
                }
            }
        },
        "2024_11_15_MIXED": {
            "description": "Mixed samples with size variations",
            "files": {
                "mixed_area1": {
                    "path": "raw_data/20241115_Mixed/40MHz_mixedspheres_4200x4000x10um_10dB.mat",
                    "params": {
                        "window_size": 7,
                        "alpha": 0.8,
                        "sigma": 2.2,
                        "box_size": 14
                    }
                },
                "mixed_area2": {
                    "path": "raw_data/20241115_Mixed/40MHz_mixedspheres_4200x4000x10um_10dB_area2.mat",
                    "params": {
                        "window_size": 3,    # Reduced from 2 for more peaks
                        "alpha": 0.05,       # Reduced from 0.2 for lower threshold  
                        "sigma": 0.2,        # Reduced from 0.4 for less smoothing
                        "box_size": 7
                    }
                }
            }
        }
    }
    return DATASETS
