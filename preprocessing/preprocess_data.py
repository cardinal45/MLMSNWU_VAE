#!/usr/bin/env python3
"""
Simple Data Preprocessor for VAE Materials Discovery

This script processes raw materials structure data to create feature-rich datasets
suitable for VAE training.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def extract_composition_from_formula(formula):
    """
    Extract elemental composition from a chemical formula.
    Returns a dictionary with element symbols as keys and fractions as values.
    """
    # Simple regex to extract elements and their counts
    pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
    matches = re.findall(pattern, formula)
    
    composition = {}
    total_atoms = 0
    
    for element, count in matches:
        count = float(count) if count else 1.0
        composition[element] = count
        total_atoms += count
    
    # Normalize to fractions
    if total_atoms > 0:
        for element in composition:
            composition[element] /= total_atoms
    
    return composition

def get_basic_elemental_properties():
    """
    Get basic elemental properties for feature engineering.
    """
    # Basic elemental properties for common elements
    elements = {
        'H': {'atomic_number': 1, 'electronegativity': 2.20, 'atomic_radius': 0.31},
        'Li': {'atomic_number': 3, 'electronegativity': 0.98, 'atomic_radius': 1.67},
        'C': {'atomic_number': 6, 'electronegativity': 2.55, 'atomic_radius': 0.67},
        'N': {'atomic_number': 7, 'electronegativity': 3.04, 'atomic_radius': 0.56},
        'O': {'atomic_number': 8, 'electronegativity': 3.44, 'atomic_radius': 0.48},
        'F': {'atomic_number': 9, 'electronegativity': 3.98, 'atomic_radius': 0.42},
        'Na': {'atomic_number': 11, 'electronegativity': 0.93, 'atomic_radius': 1.90},
        'Mg': {'atomic_number': 12, 'electronegativity': 1.31, 'atomic_radius': 1.45},
        'Al': {'atomic_number': 13, 'electronegativity': 1.61, 'atomic_radius': 1.18},
        'Si': {'atomic_number': 14, 'electronegativity': 1.90, 'atomic_radius': 1.11},
        'P': {'atomic_number': 15, 'electronegativity': 2.19, 'atomic_radius': 0.98},
        'S': {'atomic_number': 16, 'electronegativity': 2.58, 'atomic_radius': 0.88},
        'Cl': {'atomic_number': 17, 'electronegativity': 3.16, 'atomic_radius': 0.79},
        'K': {'atomic_number': 19, 'electronegativity': 0.82, 'atomic_radius': 2.43},
        'Ca': {'atomic_number': 20, 'electronegativity': 1.00, 'atomic_radius': 1.94},
        'Ti': {'atomic_number': 22, 'electronegativity': 1.54, 'atomic_radius': 1.76},
        'V': {'atomic_number': 23, 'electronegativity': 1.63, 'atomic_radius': 1.71},
        'Cr': {'atomic_number': 24, 'electronegativity': 1.66, 'atomic_radius': 1.66},
        'Mn': {'atomic_number': 25, 'electronegativity': 1.55, 'atomic_radius': 1.61},
        'Fe': {'atomic_number': 26, 'electronegativity': 1.83, 'atomic_radius': 1.56},
        'Co': {'atomic_number': 27, 'electronegativity': 1.88, 'atomic_radius': 1.52},
        'Ni': {'atomic_number': 28, 'electronegativity': 1.91, 'atomic_radius': 1.49},
        'Cu': {'atomic_number': 29, 'electronegativity': 1.90, 'atomic_radius': 1.45},
        'Zn': {'atomic_number': 30, 'electronegativity': 1.65, 'atomic_radius': 1.42},
        'Ga': {'atomic_number': 31, 'electronegativity': 1.81, 'atomic_radius': 1.36},
        'As': {'atomic_number': 33, 'electronegativity': 2.18, 'atomic_radius': 1.19},
        'Br': {'atomic_number': 35, 'electronegativity': 2.96, 'atomic_radius': 1.10},
        'Sr': {'atomic_number': 38, 'electronegativity': 0.95, 'atomic_radius': 2.19},
        'Nb': {'atomic_number': 41, 'electronegativity': 1.6, 'atomic_radius': 1.98},
        'Mo': {'atomic_number': 42, 'electronegativity': 2.16, 'atomic_radius': 1.90},
        'Ag': {'atomic_number': 47, 'electronegativity': 1.93, 'atomic_radius': 1.65},
        'Cd': {'atomic_number': 48, 'electronegativity': 1.69, 'atomic_radius': 1.61},
        'In': {'atomic_number': 49, 'electronegativity': 1.78, 'atomic_radius': 1.56},
        'Sn': {'atomic_number': 50, 'electronegativity': 1.96, 'atomic_radius': 1.45},
        'Sb': {'atomic_number': 51, 'electronegativity': 2.05, 'atomic_radius': 1.33},
        'Te': {'atomic_number': 52, 'electronegativity': 2.1, 'atomic_radius': 1.23},
        'I': {'atomic_number': 53, 'electronegativity': 2.66, 'atomic_radius': 1.15},
        'Ba': {'atomic_number': 56, 'electronegativity': 0.89, 'atomic_radius': 2.53},
        'La': {'atomic_number': 57, 'electronegativity': 1.1, 'atomic_radius': 2.40},
        'Ce': {'atomic_number': 58, 'electronegativity': 1.12, 'atomic_radius': 2.35},
        'Eu': {'atomic_number': 63, 'electronegativity': 1.2, 'atomic_radius': 2.42},
        'Gd': {'atomic_number': 64, 'electronegativity': 1.2, 'atomic_radius': 2.17},
        'Ta': {'atomic_number': 73, 'electronegativity': 1.5, 'atomic_radius': 2.00},
        'W': {'atomic_number': 74, 'electronegativity': 2.36, 'atomic_radius': 1.93},
        'Pt': {'atomic_number': 78, 'electronegativity': 2.28, 'atomic_radius': 1.77},
        'Au': {'atomic_number': 79, 'electronegativity': 2.54, 'atomic_radius': 1.74},
        'Pb': {'atomic_number': 82, 'electronegativity': 2.33, 'atomic_radius': 1.54},
        'Bi': {'atomic_number': 83, 'electronegativity': 2.02, 'atomic_radius': 1.43}
    }
    return elements

def extract_reduced_formula(structure_text):
    """Extract the reduced formula from structure text."""
    lines = structure_text.split('\n')
    for line in lines:
        if 'Reduced Formula:' in line:
            formula = line.split('Reduced Formula:')[1].strip()
            return formula
    return None

def create_features_from_composition(composition, elemental_props):
    """
    Create numerical features from elemental composition.
    """
    features = {}
    
    # Basic composition statistics
    features['n_elements'] = len(composition)
    
    if not composition:
        # Return zero features if no composition found
        return {f'feature_{i}': 0.0 for i in range(50)}
    
    # Weighted elemental properties
    props_to_calc = ['atomic_number', 'electronegativity', 'atomic_radius']
    
    for prop in props_to_calc:
        values = []
        weights = []
        for element, fraction in composition.items():
            if element in elemental_props:
                values.append(elemental_props[element][prop])
                weights.append(fraction)
        
        if values:
            # Weighted mean
            features[f'mean_{prop}'] = np.average(values, weights=weights)
            # Weighted standard deviation
            if len(values) > 1:
                features[f'std_{prop}'] = np.sqrt(np.average((np.array(values) - features[f'mean_{prop}'])**2, weights=weights))
            else:
                features[f'std_{prop}'] = 0.0
            # Range
            features[f'range_{prop}'] = max(values) - min(values) if len(values) > 1 else 0.0
        else:
            features[f'mean_{prop}'] = 0.0
            features[f'std_{prop}'] = 0.0
            features[f'range_{prop}'] = 0.0
    
    # Element-specific features (most common elements)
    common_elements = ['H', 'Li', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 
                      'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'As', 'Br', 'Sr', 'Nb', 'Mo']
    
    for element in common_elements:
        features[f'frac_{element}'] = composition.get(element, 0.0)
    
    return features

def process_formation_energy_data():
    """
    Process the raw formation energy data to create a VAE-ready dataset.
    """
    print("🔧 Processing formation energy data for VAE training...")
    
    # Load raw data
    data_dir = Path("../data")
    input_file = data_dir / "formation_energy.csv"
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return None
    
    # Read the CSV file
    print(f"📖 Reading data from {input_file}")
    df = pd.read_csv(input_file)
    print(f"✅ Loaded {len(df)} samples")
    
    # Check if already processed
    if 'feature_0' in df.columns and 'e_form_per_atom' in df.columns:
        print("✅ Data already appears to be processed")
        return df
    
    # Get elemental properties
    elemental_props = get_basic_elemental_properties()
    
    # Process each material
    processed_data = []
    failed_count = 0
    
    print("Processing materials...")
    for idx, row in df.iterrows():
        if idx % 1000 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(df)} samples...")
        
        try:
            # Extract reduced formula from structure
            structure_text = str(row['structure'])
            reduced_formula = extract_reduced_formula(structure_text)
            
            if not reduced_formula:
                failed_count += 1
                continue
            
            # Extract composition
            composition = extract_composition_from_formula(reduced_formula)
            
            if not composition:
                failed_count += 1
                continue
            
            # Create features
            features = create_features_from_composition(composition, elemental_props)
            
            # Add target (check column name)
            target_value = row.get('e_form', row.get('e_form_per_atom', None))
            if target_value is None:
                failed_count += 1
                continue
            
            # Create material data
            material_data = features.copy()
            material_data['e_form_per_atom'] = float(target_value)
            material_data['reduced_formula'] = reduced_formula
            
            processed_data.append(material_data)
            
        except Exception as e:
            failed_count += 1
            if failed_count <= 5:  # Only print first few errors
                print(f"⚠️ Failed to process sample {idx}: {e}")
    
    print(f"✅ Successfully processed {len(processed_data)} materials")
    print(f"⚠️ Failed to process {failed_count} materials")
    
    if not processed_data:
        print("❌ No materials were successfully processed")
        return None
    
    # Create DataFrame
    processed_df = pd.DataFrame(processed_data)
    
    # Save processed data
    output_file = data_dir / "formation_energy_processed.csv"
    processed_df.to_csv(output_file, index=False)
    print(f"💾 Saved processed data to {output_file}")
    
    # Print summary
    print(f"\n📊 Processed Dataset Summary:")
    print(f"   Samples: {len(processed_df)}")
    print(f"   Features: {len([col for col in processed_df.columns if not col.startswith(('e_form', 'reduced_formula', 'n_elements'))])}")
    print(f"   Formation energy range: {processed_df['e_form_per_atom'].min():.3f} to {processed_df['e_form_per_atom'].max():.3f} eV/atom")
    print(f"   Mean formation energy: {processed_df['e_form_per_atom'].mean():.3f} ± {processed_df['e_form_per_atom'].std():.3f} eV/atom")
    
    return processed_df

if __name__ == "__main__":
    print("🚀 Materials Data Preprocessor for VAE")
    print("="*50)
    
    # Process formation energy data
    processed_df = process_formation_energy_data()
    
    if processed_df is not None:
        print(f"\n✅ Data preprocessing complete!")
        print(f"📂 Processed data ready for VAE training")
    else:
        print(f"\n❌ Data preprocessing failed!")
