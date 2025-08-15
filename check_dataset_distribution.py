import os
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

def analyze_fer_dataset(data_path):
    """Analyze FER dataset distribution"""
    print("📊 ANALYZING FER DATASET DISTRIBUTION")
    print("="*50)
    print(f"📁 Dataset path: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ Dataset path not found: {data_path}")
        
        # Try alternative paths
        alternative_paths = [
            "E:/Academic/Y4S1/CMIS 4+26 Research Project/smile_detection_ex/FER/fer2013",
            "E:/Academic/Y4S1/CMIS 4+26 Research Project/smile_detection_ex/fer2013",
            "E:/Academic/Y4S1/CMIS 4+26 Research Project/smile_detection_ex/FER",
            "./FER/fer2013/train",
            "./fer2013/train",
            "./FER",
            "./fer2013"
        ]
        
        print("\n🔍 Checking alternative paths:")
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                print(f"   ✅ Found: {alt_path}")
                # Check if it has emotion folders
                subfolders = [f for f in os.listdir(alt_path) if os.path.isdir(os.path.join(alt_path, f))]
                if any(emotion in subfolders for emotion in ['happy', 'sad', 'angry', 'neutral']):
                    print(f"   🎯 This looks like the FER dataset!")
                    data_path = alt_path
                    break
            else:
                print(f"   ❌ Not found: {alt_path}")
        
        if not os.path.exists(data_path):
            print("\n💡 Please check your FER dataset location!")
            return None
    
    print(f"\n📂 Analyzing dataset at: {data_path}")
    
    # Expected emotion folders
    emotion_folders = {
        'angry': 0,
        'disgust': 1, 
        'fear': 2,
        'happy': 3,
        'sad': 4,
        'surprise': 5,
        'neutral': 6
    }
    
    emotion_counts = {}
    total_images = 0
    found_emotions = []
    
    # Check each emotion folder
    for emotion_name, emotion_id in emotion_folders.items():
        emotion_path = os.path.join(data_path, emotion_name)
        
        if os.path.exists(emotion_path):
            # Count image files
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
            image_count = 0
            
            for ext in image_extensions:
                image_count += len(list(Path(emotion_path).glob(ext)))
                image_count += len(list(Path(emotion_path).glob(ext.upper())))
            
            emotion_counts[emotion_name] = image_count
            total_images += image_count
            found_emotions.append(emotion_name)
            
            print(f"   {emotion_name}: {image_count:,} images")
            
            # Show some sample files
            sample_files = list(Path(emotion_path).glob('*'))[:3]
            if sample_files:
                print(f"      Sample files: {[f.name for f in sample_files]}")
        else:
            print(f"   {emotion_name}: ❌ Folder not found")
            emotion_counts[emotion_name] = 0
    
    print(f"\n📈 DATASET SUMMARY:")
    print(f"   Total images: {total_images:,}")
    print(f"   Emotions found: {len(found_emotions)}/7")
    print(f"   Found emotions: {found_emotions}")
    
    if total_images == 0:
        print("❌ No images found in any emotion folder!")
        print("\n💡 Possible issues:")
        print("   1. Wrong dataset path")
        print("   2. Different folder structure")
        print("   3. Images in different format")
        
        # Check actual folder structure
        print(f"\n🔍 Actual folder structure in {data_path}:")
        try:
            items = os.listdir(data_path)
            for item in items[:10]:  # Show first 10 items
                item_path = os.path.join(data_path, item)
                if os.path.isdir(item_path):
                    file_count = len([f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))])
                    print(f"   📁 {item}/ ({file_count} files)")
                else:
                    print(f"   📄 {item}")
        except Exception as e:
            print(f"   Error reading directory: {e}")
        
        return None
    
    # Calculate statistics
    if emotion_counts:
        values = list(emotion_counts.values())
        max_count = max(values)
        min_count = min([v for v in values if v > 0])  # Exclude 0s
        avg_count = total_images / len([v for v in values if v > 0])
        
        print(f"\n📊 STATISTICS:")
        print(f"   Maximum: {max_count:,} images")
        print(f"   Minimum: {min_count:,} images") 
        print(f"   Average: {avg_count:,.0f} images")
        print(f"   Imbalance ratio: {max_count/min_count:.1f}:1")
        
        # Check balance
        if max_count / min_count > 3:
            print("   ⚠️ WARNING: Significant class imbalance!")
        elif max_count / min_count > 2:
            print("   🔶 CAUTION: Moderate class imbalance")
        else:
            print("   ✅ GOOD: Well balanced dataset")
    
    # Training recommendations
    print(f"\n💡 TRAINING RECOMMENDATIONS:")
    
    if total_images < 1000:
        print("   ❌ Very small dataset - consider data augmentation")
        max_per_emotion = min(100, total_images // 7)
    elif total_images < 5000:
        print("   🔶 Small dataset - use data augmentation")
        max_per_emotion = min(500, total_images // 7)
    elif total_images < 20000:
        print("   ✅ Good dataset size")
        max_per_emotion = min(1000, total_images // 7)
    else:
        print("   🎉 Large dataset - excellent for training!")
        max_per_emotion = min(2000, total_images // 7)
    
    print(f"   Suggested max_per_emotion: {max_per_emotion}")
    
    # Show percentages
    print(f"\n📊 EMOTION DISTRIBUTION:")
    for emotion_name, count in emotion_counts.items():
        if total_images > 0:
            percentage = (count / total_images) * 100
            bar = "█" * int(percentage / 2)  # Visual bar
            print(f"   {emotion_name:8}: {count:5,} ({percentage:5.1f}%) {bar}")
    
    return emotion_counts, total_images

def plot_emotion_distribution(emotion_counts):
    """Create visualization of emotion distribution"""
    try:
        # Filter out emotions with 0 images
        filtered_counts = {k: v for k, v in emotion_counts.items() if v > 0}
        
        if not filtered_counts:
            print("No data to plot")
            return
        
        emotions = list(filtered_counts.keys())
        counts = list(filtered_counts.values())
        
        # Create bar plot
        plt.figure(figsize=(12, 8))
        
        # Main bar plot
        plt.subplot(2, 1, 1)
        bars = plt.bar(emotions, counts, color=['red', 'brown', 'orange', 'green', 'blue', 'purple', 'gray'])
        plt.title('FER Dataset - Images per Emotion', fontsize=16, fontweight='bold')
        plt.xlabel('Emotions')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01, 
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        
        # Pie chart
        plt.subplot(2, 1, 2)
        plt.pie(counts, labels=emotions, autopct='%1.1f%%', startangle=90)
        plt.title('FER Dataset - Emotion Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('fer_dataset_analysis.png', dpi=150, bbox_inches='tight')
        print("📊 Distribution plot saved as: fer_dataset_analysis.png")
        plt.show()
        
    except Exception as e:
        print(f"Could not create plot: {e}")

def check_sample_images(data_path, emotion_counts):
    """Check sample images from each emotion"""
    print(f"\n🖼️ SAMPLE IMAGE CHECK:")
    
    for emotion_name, count in emotion_counts.items():
        if count > 0:
            emotion_path = os.path.join(data_path, emotion_name)
            
            # Get first few image files
            sample_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                sample_files.extend(list(Path(emotion_path).glob(ext))[:2])
            
            if sample_files:
                print(f"\n   {emotion_name.upper()} samples:")
                for sample_file in sample_files[:2]:
                    try:
                        # Try to get file size
                        file_size = sample_file.stat().st_size
                        print(f"      {sample_file.name} ({file_size:,} bytes)")
                    except Exception as e:
                        print(f"      {sample_file.name} (error: {e})")

def main():
    """Main analysis function"""
    print("🔍 FER DATASET DISTRIBUTION CHECKER")
    print("="*60)
    
    # Primary dataset path
    data_path = "E:/Academic/Y4S1/CMIS 4+26 Research Project/smile_detection_ex/FER/fer2013/train"
    
    # Analyze dataset
    result = analyze_fer_dataset(data_path)
    
    if result:
        emotion_counts, total_images = result
        
        # Create visualization
        plot_emotion_distribution(emotion_counts)
        
        # Check sample images
        check_sample_images(data_path, emotion_counts)
        
        # Update train_multi_emotion.py recommendation
        non_zero_counts = [v for v in emotion_counts.values() if v > 0]
        if non_zero_counts:
            min_count = min(non_zero_counts)
            recommended_max = min(min_count, 1000)  # Don't exceed smallest class
            
            print(f"\n🔧 UPDATE YOUR TRAIN_MULTI_EMOTION.PY:")
            print(f"   Change line: max_per_emotion={recommended_max}")
            print(f"   This ensures balanced training across all emotions")
            
            if total_images >= 10000:
                print(f"   ✅ You have sufficient data for good training!")
            elif total_images >= 5000:
                print(f"   🔶 Moderate dataset - use data augmentation")
            else:
                print(f"   ❌ Small dataset - heavy augmentation recommended")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()