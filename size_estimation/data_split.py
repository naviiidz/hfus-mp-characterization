"""
Stratified size estimation data splitting

This script performs stratified splitting where each size bin maintains 
70-15-15 proportions across train/val/test splits.

Author: Assistant
Date: December 22, 2025
"""

import pandas as pd
import numpy as np
import random
import ast
from pathlib import Path
from collections import defaultdict

def clean_radius(radius_str):
    """Clean radius values by removing 'um', brackets, and converting to float."""
    if isinstance(radius_str, str):
        # Remove 'um', brackets, and any whitespace
        cleaned = radius_str.replace('um', '').replace('[', '').replace(']', '').strip()
        return float(cleaned)
    return float(radius_str)

def get_size_bin(radius, bin_ranges):
    """Get size bin for a given radius."""
    for i, (min_size, max_size) in enumerate(bin_ranges):
        if min_size <= radius <= max_size:
            return i
    return None  # Outside any bin

def load_data():
    """Load the complete dataset."""
    data_path = Path(__file__).parent.parent / "train_test_with_pmma.csv"
    df = pd.read_csv(data_path)
    
    # Clean radius values
    df['Radius'] = df['Radius'].apply(clean_radius)
    
    # Define size bins (from config)
    bin_ranges = [(15, 25), (40, 50), (65, 75), (290, 310)]
    
    # Add size bin column
    df['Size_Bin'] = df['Radius'].apply(lambda x: get_size_bin(x, bin_ranges))
    
    # Remove samples outside any bin
    df = df[df['Size_Bin'].notna()].copy()
    
    print(f"Total samples after filtering: {len(df)}")
    print(f"Size bin distribution:")
    for i, (min_size, max_size) in enumerate(bin_ranges):
        bin_count = len(df[df['Size_Bin'] == i])
        print(f"  Bin {i+1} ({min_size}-{max_size}μm): {bin_count} samples")
    
    return df

def add_particle_ids(df):
    """Add particle IDs based on coordinate continuity."""
    
    # Parse coordinates from string format
    def parse_coords(coord_str):
        try:
            coords = ast.literal_eval(coord_str)
            return coords[0], coords[1]  # x, y
        except:
            return None, None
    
    # Add X, Y columns
    df[['X', 'Y']] = df['coordinates'].apply(parse_coords).apply(pd.Series)
    
    # Remove rows with invalid coordinates
    df = df.dropna(subset=['X', 'Y']).reset_index(drop=True)
    
    df = df.sort_values(['Material Type', 'X', 'Y']).reset_index(drop=True)
    
    particle_ids = []
    current_particle_id = 0
    
    for material in df['Material Type'].unique():
        material_data = df[df['Material Type'] == material]
        
        if len(material_data) == 0:
            continue
            
        prev_x, prev_y = None, None
        
        for _, row in material_data.iterrows():
            x, y = row['X'], row['Y']
            
            # Start new particle if coordinates jump significantly
            if prev_x is None or abs(x - prev_x) > 1 or abs(y - prev_y) > 1:
                current_particle_id += 1
            
            particle_ids.append(current_particle_id)
            prev_x, prev_y = x, y
    
    # Add particle IDs back to the dataframe in the correct order
    df_with_ids = df.copy()
    df_with_ids['Particle_ID'] = particle_ids
    
    return df_with_ids

def stratified_split_by_size(df, bin_ranges, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Perform stratified splitting by size bins for all materials.
    
    Each size bin is split 70-15-15 independently to maintain size distribution.
    
    Args:
        df: Input dataframe
        bin_ranges: List of (min, max) tuples for size bins
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set  
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with materials as keys and (train, val, test) tuples as values
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Add size bins to dataframe
    df['Radius'] = df['Radius'].apply(clean_radius)  # Clean radius values first
    df['Size_Bin'] = df['Radius'].apply(lambda x: get_size_bin(x, bin_ranges))
    df = df[df['Size_Bin'].notna()].copy()
    
    # Add particle IDs
    df = add_particle_ids(df)
    
    results = {}
    
    # Process each material
    for material in sorted(df['Material Type'].unique()):
        material_data = df[df['Material Type'] == material].copy()
        
        if len(material_data) == 0:
            continue
        
        # Get unique particles for this material
        particles_by_bin = defaultdict(list)
        
        for particle_id in material_data['Particle_ID'].unique():
            particle_data = material_data[material_data['Particle_ID'] == particle_id]
            # Use the most common size bin for this particle (in case of mixed signals)
            size_bin = particle_data['Size_Bin'].mode()[0]
            particles_by_bin[size_bin].append(particle_id)
        
        print(f"\n{material} particles by size bin:")
        for size_bin, particles in particles_by_bin.items():
            print(f"  Size bin {int(size_bin)+1}: {len(particles)} particles")
        
        # Split each size bin independently
        train_particles = []
        val_particles = []
        test_particles = []
    
        for size_bin, particles in particles_by_bin.items():
            n_particles = len(particles)
            
            if n_particles < 3:  # Need at least 3 particles for train/val/test
                print(f"  Warning: Size bin {int(size_bin)+1} has only {n_particles} particles - adding all to train")
                train_particles.extend(particles)
                continue
            
            # Calculate split sizes
            n_train = max(1, int(n_particles * train_ratio))
            n_val = max(1, int(n_particles * val_ratio))
            n_test = n_particles - n_train - n_val  # Remainder goes to test
            
            if n_test < 1:  # Adjust if test set would be empty
                n_test = 1
                n_val = max(1, n_particles - n_train - n_test)
            
            print(f"  Size bin {int(size_bin)+1} split: Train={n_train}, Val={n_val}, Test={n_test}")
            
            # Shuffle particles and split
            shuffled_particles = particles.copy()
            random.shuffle(shuffled_particles)
            
            train_particles.extend(shuffled_particles[:n_train])
            val_particles.extend(shuffled_particles[n_train:n_train + n_val])
            test_particles.extend(shuffled_particles[n_train + n_val:])
        
        # Create split datasets
        train_data = material_data[material_data['Particle_ID'].isin(train_particles)]
        val_data = material_data[material_data['Particle_ID'].isin(val_particles)]
        test_data = material_data[material_data['Particle_ID'].isin(test_particles)]
        
        # Verify size distribution preservation
        print(f"\n{material} final size distribution:")
        for i in range(len(bin_ranges)):
            train_count = len(train_data[train_data['Size_Bin'] == i])
            val_count = len(val_data[val_data['Size_Bin'] == i])
            test_count = len(test_data[test_data['Size_Bin'] == i])
            total = train_count + val_count + test_count
            
            if total > 0:
                print(f"  Bin {i+1}: Train={train_count} ({train_count/total:.1%}), "
                      f"Val={val_count} ({val_count/total:.1%}), "
                      f"Test={test_count} ({test_count/total:.1%})")
        
        results[material] = (train_data, val_data, test_data)
    
    return results

def save_material_splits(splits_dict, output_dir):
    """Save the split data for all materials.
    
    Args:
        splits_dict: Dictionary with materials as keys and (train, val, test) tuples as values
        output_dir: Output directory path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Drop the helper columns before saving
    columns_to_drop = ['Size_Bin', 'Particle_ID']
    
    for material, (train_data, val_data, test_data) in splits_dict.items():
        train_clean = train_data.drop(columns=columns_to_drop, errors='ignore')
        val_clean = val_data.drop(columns=columns_to_drop, errors='ignore')
        test_clean = test_data.drop(columns=columns_to_drop, errors='ignore')
        
        # Save files
        train_path = output_dir / f"{material.lower()}_train.csv"
        val_path = output_dir / f"{material.lower()}_val.csv"
        test_path = output_dir / f"{material.lower()}_test.csv"
        
        train_clean.to_csv(train_path, index=False)
        val_clean.to_csv(val_path, index=False)
        test_clean.to_csv(test_path, index=False)
        
        print(f"Saved {material} splits:")
        print(f"  Train: {len(train_clean)} samples -> {train_path}")
        print(f"  Val:   {len(val_clean)} samples -> {val_path}")
        print(f"  Test:  {len(test_clean)} samples -> {test_path}")

def main():
    """Main function to perform stratified splitting."""
    print("=" * 80)
    print("STRATIFIED SIZE ESTIMATION DATA SPLITTING")
    print("=" * 80)
    
    # Load and prepare data
    print("\n1. Loading data...")
    df = load_data()
    
    # Define size bins
    bin_ranges = [(15, 25), (40, 50), (65, 75), (290, 310)]
    
    # Output directory
    output_dir = Path(__file__).parent.parent / "size_estimation_splits_stratified"
    
    print(f"\n2. Performing stratified splitting...")
    print(f"Output directory: {output_dir}")
    
    # Perform splitting for all materials
    print(f"\n{'='*60}")
    print(f"Processing all materials")
    print(f"{'='*60}")
    
    splits = stratified_split_by_size(df, bin_ranges, seed=50)
    
    if splits:
        save_material_splits(splits, output_dir)
    else:
        print("No data found for splitting")
    
    print(f"\n{'='*80}")
    print("STRATIFIED SPLITTING COMPLETE!")
    print(f"All splits saved to: {output_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()