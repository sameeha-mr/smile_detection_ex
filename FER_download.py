import os
import pandas as pd
import numpy as np
import requests
import zipfile
from pathlib import Path

def download_fer2013_manual():
    """
    Manual download instructions for FER-2013
    """
    print("📋 FER-2013 DATASET DOWNLOAD GUIDE")
    print("="*60)
    print("1. Go to: https://www.kaggle.com/datasets/msambare/fer2013")
    print("2. Click 'Download' button (requires Kaggle account)")
    print("3. Extract fer2013.csv to: ./datasets/fer2013/")
    print("4. The file should be at: ./datasets/fer2013/fer2013.csv")
    print("="*60)
    
    # Create directory structure
    os.makedirs("./datasets/fer2013", exist_ok=True)
    
    # Check if already downloaded
    fer_path = "./datasets/fer2013/fer2013.csv"
    if os.path.exists(fer_path):
        print("✅ FER-2013 dataset found!")
        
        # Check dataset info
        df = pd.read_csv(fer_path)
        print(f"📊 Dataset info:")
        print(f"   Total images: {len(df)}")
        print(f"   Emotions: {df['emotion'].value_counts().to_dict()}")
        return True
    else:
        print("❌ FER-2013 dataset not found. Please download manually.")
        return False

def create_sample_fer_dataset():
    """
    Create a small sample FER dataset for testing
    """
    print("🔧 Creating sample FER dataset for testing...")
    
    # Create sample data matching FER-2013 structure
    sample_size = 1000
    emotions = np.random.randint(0, 7, sample_size)
    
    # Generate random pixel data (48x48 = 2304 pixels)
    pixels_data = []
    for i in range(sample_size):
        # Create realistic-looking face data
        pixels = np.random.randint(0, 255, 48*48)
        pixels_str = ' '.join(map(str, pixels))
        pixels_data.append(pixels_str)
    
    # Create DataFrame
    sample_df = pd.DataFrame({
        'emotion': emotions,
        'pixels': pixels_data,
        'Usage': ['Training'] * sample_size
    })
    
    # Save sample dataset
    os.makedirs("./datasets/fer2013", exist_ok=True)
    sample_df.to_csv("./datasets/fer2013/fer2013_sample.csv", index=False)
    
    print("✅ Sample FER dataset created!")
    print(f"   Sample size: {sample_size}")
    print(f"   Location: ./datasets/fer2013/fer2013_sample.csv")
    return True

if __name__ == "__main__":
    if not download_fer2013_manual():
        print("\n🔧 Creating sample dataset for testing...")
        create_sample_fer_dataset()