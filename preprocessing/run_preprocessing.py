#!/usr/bin/env python3
"""
Quick Data Preprocessing Runner

This script runs the data preprocessing to create a VAE-ready dataset.
"""

import sys
from pathlib import Path

# Add current directory to path to import preprocessing module
sys.path.append(str(Path(__file__).parent))

try:
    from preprocess_data import process_formation_energy_data
    
    print("🚀 Running Materials Data Preprocessing")
    print("="*50)
    
    # Run preprocessing
    result = process_formation_energy_data()
    
    if result is not None:
        print("\n✅ Preprocessing completed successfully!")
        print("📁 Processed data is ready for VAE training")
        print("\n💡 You can now run the VAE notebook with confidence!")
    else:
        print("\n❌ Preprocessing failed!")
        print("💡 Make sure the workshop data exists in 'workshop_data/workshop_ready/'")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure preprocess_data.py is in the same directory")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
