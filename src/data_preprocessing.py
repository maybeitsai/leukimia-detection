"""
Data preprocessing utilities for blood cell classification.
This module contains functions for segmentation, data preparation, and augmentation.
"""

import os
import cv2
import numpy as np
import pandas as pd
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy import ndimage as ndi
from skimage import morphology
import time
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_temp_folders(base_path="temp_data"):
    """Create temporary folders for processed data."""
    folders = {
        "prepared_data": ["benign", "PreB", "ProB", "EarlyPreB"],
        "prepared_test": ["benign", "PreB", "ProB", "EarlyPreB"]
    }
    
    for parent, subfolders in folders.items():
        for sub in subfolders:
            path = os.path.join(base_path, parent, sub)
            os.makedirs(path, exist_ok=True)
    
    print("All temporary folders created successfully!")


def segment_blood_cell(image):
    """
    Segment blood cell from background using K-means clustering and morphological operations.
    
    Args:
        image: Input RGB image
        
    Returns:
        Segmented image with background removed
    """
    # Convert RGB to LAB color space
    i_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(i_lab)
    
    # Reshape for K-means clustering
    i2 = a.reshape(a.shape[0] * a.shape[1], 1)
    
    # Apply K-means clustering
    km = KMeans(n_clusters=7, random_state=42).fit(i2)
    p2s = km.cluster_centers_[km.labels_]
    ic = p2s.reshape(a.shape[0], a.shape[1])
    ic = ic.astype(np.uint8)
    
    # Binary thresholding
    r, t = cv2.threshold(ic, 141, 255, cv2.THRESH_BINARY)
    
    # Morphological operations
    fh = ndi.binary_fill_holes(t)
    m1 = morphology.remove_small_objects(fh, 200)
    m2 = morphology.remove_small_holes(m1, 250)
    m2 = m2.astype(np.uint8)
    
    # Apply mask to original image
    out = cv2.bitwise_and(image, image, mask=m2)
    
    return out


def prepare_data_splits(data_dir, train_ratio=0.90, random_seed=88):
    """
    Split data into training and testing sets.
    
    Args:
        data_dir: Directory containing the blood cell data
        train_ratio: Ratio of training data (default: 0.90)
        random_seed: Random seed for reproducibility
        
    Returns:
        train_list, test_list: Lists of file paths for training and testing
    """
    data_list = sorted(list(paths.list_images(data_dir)))
    
    random.seed(random_seed)
    random.shuffle(data_list)
    
    train_list, test_list = train_test_split(
        data_list, 
        train_size=train_ratio, 
        shuffle=True, 
        random_state=random_seed
    )
    
    print(f'Number of training samples: {len(train_list)}')
    print(f'Number of testing samples: {len(test_list)}')
    
    return train_list, test_list


def process_and_save_test_data(test_list, output_dir="temp_data/prepared_test"):
    """
    Process and save test data without augmentation.
    
    Args:
        test_list: List of test image paths
        output_dir: Directory to save processed test images
    """
    print("Processing test data...")
    
    for p, img_path in enumerate(test_list):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        
        # Extract label from path
        label = os.path.basename(os.path.dirname(img_path))
        
        # Map labels to folder names
        if label == "Benign":
            save_path = os.path.join(output_dir, "benign", f"{label}{p}.png")
        elif label == "[Malignant] Pre-B":
            save_path = os.path.join(output_dir, "PreB", f"{label}{p}.png")
        elif label == "[Malignant] Pro-B":
            save_path = os.path.join(output_dir, "ProB", f"{label}{p}.png")
        elif label == "[Malignant] early Pre-B":
            save_path = os.path.join(output_dir, "EarlyPreB", f"{label}{p}.png")
        
        cv2.imwrite(save_path, img)
    
    print(f"Test data processing completed! Processed {len(test_list)} images.")


def process_and_save_train_data(train_list, output_dir="temp_data/prepared_data", 
                               enable_preprocessing=True, apply_clahe=False):
    """
    Process and save training data with NORMAL image preprocessing (NO SEGMENTATION).
    Each original sample is resized and optionally enhanced, then saved as is.
    
    Args:
        train_list: List of training image paths
        output_dir: Directory to save processed training images
        enable_preprocessing: Whether to apply basic image enhancements
        apply_clahe: Whether to apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    """
    print("=" * 60)
    print("🔄 NORMAL IMAGE PREPROCESSING MODE (NO SEGMENTATION)")
    print("=" * 60)
    print(f"Processing {len(train_list)} training images...")
    if enable_preprocessing:
        print("✅ Basic image enhancements: ENABLED")
    if apply_clahe:
        print("✅ CLAHE enhancement: ENABLED")
    print("❌ K-means segmentation: DISABLED")
    print("-" * 60)
    
    tic = time.perf_counter()
    
    # Create CLAHE object if needed
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    for p, img_path in enumerate(train_list):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        
        # Apply optional preprocessing
        if enable_preprocessing:
            # Convert to RGB for processing
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Apply CLAHE if enabled
            if apply_clahe:
                # Apply CLAHE to each channel
                img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
                img_lab[:,:,0] = clahe.apply(img_lab[:,:,0])
                img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
            
            # Optional: Add slight denoising
            img_rgb = cv2.bilateralFilter(img_rgb, 9, 75, 75)
            
            # Convert back to BGR for saving
            img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # Extract label from path
        label = os.path.basename(os.path.dirname(img_path))
        
        # Save processed image
        if label == "Benign":
            save_path = os.path.join(output_dir, "benign", f"{label}_{p:05d}.png")
        elif label == "[Malignant] Pre-B":
            save_path = os.path.join(output_dir, "PreB", f"PreB_{p:05d}.png")
        elif label == "[Malignant] Pro-B":
            save_path = os.path.join(output_dir, "ProB", f"ProB_{p:05d}.png")
        elif label == "[Malignant] early Pre-B":
            save_path = os.path.join(output_dir, "EarlyPreB", f"EarlyPreB_{p:05d}.png")
        
        cv2.imwrite(save_path, img)
        
        # Progress indicator
        if (p + 1) % 100 == 0:
            print(f"Processed {p + 1}/{len(train_list)} images...")
    
    toc = time.perf_counter()
    print("=" * 60)
    print("✅ NORMAL PREPROCESSING COMPLETED!")
    print(f"⏱️  Time taken: {((toc - tic)/60):.2f} minutes")
    print(f"📊 Processed: {len(train_list)} original images (1:1 ratio)")
    print(f"💾 Saved to: {output_dir}")
    print("=" * 60)


def choose_preprocessing_method(train_list, output_dir="temp_data/prepared_data", 
                              use_segmentation=False, enable_enhancements=True, apply_clahe=False):
    """
    Choose between normal preprocessing or segmentation-based preprocessing.
    
    RECOMMENDED: Use normal preprocessing (use_segmentation=False) for better performance.
    
    Args:
        train_list: List of training image paths
        output_dir: Directory to save processed training images
        use_segmentation: If True, applies K-means segmentation (more complex, slower)
                         If False, uses normal image processing (recommended)
        enable_enhancements: Whether to apply basic image enhancements (normal mode only)
        apply_clahe: Whether to apply CLAHE enhancement (normal mode only)
    """
    if use_segmentation:
        print("⚠️ WARNING: Segmentation mode selected - this is complex and slow!")
        print("📝 RECOMMENDATION: Use normal preprocessing instead (use_segmentation=False)")
        # Use the legacy segmentation function (if it exists in the codebase)
        process_and_save_train_data_with_segmentation(train_list, output_dir)
    else:
        print("✅ RECOMMENDED: Using normal image preprocessing")
        process_and_save_train_data(train_list, output_dir, enable_enhancements, apply_clahe)


def process_and_save_train_data_with_segmentation(train_list, output_dir="temp_data/prepared_data"):
    """
    LEGACY FUNCTION: Process and save training data with segmentation augmentation.
    Each original sample creates two samples: original and segmented.
    
    NOTE: This function is kept for backward compatibility but is not recommended.
    Use process_and_save_train_data() for normal preprocessing instead.
    
    Args:
        train_list: List of training image paths
        output_dir: Directory to save processed training images
    """
    print("Processing training data with segmentation... (LEGACY MODE)")
    tic = time.perf_counter()
    
    p = 0
    for img_path in train_list:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        
        # Extract label from path
        label = os.path.basename(os.path.dirname(img_path))
        
        # Save original image
        if label == "Benign":
            save_path = os.path.join(output_dir, "benign", f"{label}{p}.png")
        elif label == "[Malignant] Pre-B":
            save_path = os.path.join(output_dir, "PreB", f"{label}{p}.png")
        elif label == "[Malignant] Pro-B":
            save_path = os.path.join(output_dir, "ProB", f"{label}{p}.png")
        elif label == "[Malignant] early Pre-B":
            save_path = os.path.join(output_dir, "EarlyPreB", f"{label}{p}.png")
        
        cv2.imwrite(save_path, img)
        p += 1
        
        # Process segmented version
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        segmented = segment_blood_cell(img_rgb)
        segmented_bgr = cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR)
        
        # Save segmented image
        if label == "Benign":
            save_path = os.path.join(output_dir, "benign", f"{label}{p}.png")
        elif label == "[Malignant] Pre-B":
            save_path = os.path.join(output_dir, "PreB", f"{label}{p}.png")
        elif label == "[Malignant] Pro-B":
            save_path = os.path.join(output_dir, "ProB", f"{label}{p}.png")
        elif label == "[Malignant] early Pre-B":
            save_path = os.path.join(output_dir, "EarlyPreB", f"{label}{p}.png")
        
        cv2.imwrite(save_path, segmented_bgr)
        p += 1
    
    toc = time.perf_counter()
    print(f"Training data processing completed in {((toc - tic)/60):.2f} minutes")
    print(f"Processed {len(train_list)} samples into {p} augmented samples")


def create_dataframes(train_dir="temp_data/prepared_data", test_dir="temp_data/prepared_test", random_seed=88):
    """
    Create pandas DataFrames from processed data directories.
    
    Args:
        train_dir: Directory containing processed training data
        test_dir: Directory containing processed test data
        random_seed: Random seed for shuffling
        
    Returns:
        train_df, test_df: DataFrames with filenames and labels
    """
    # Create test DataFrame
    test_filenames = sorted(list(paths.list_images(test_dir)))
    random.shuffle(test_filenames)
    test_labels = [os.path.basename(os.path.dirname(path)) for path in test_filenames]
    
    test_df = pd.DataFrame({
        'filenames': test_filenames,
        'labels': test_labels
    })
    
    # Create train DataFrame
    train_filenames = sorted(list(paths.list_images(train_dir)))
    random.shuffle(train_filenames)
    train_labels = [os.path.basename(os.path.dirname(path)) for path in train_filenames]
    
    train_df = pd.DataFrame({
        'filenames': train_filenames,
        'labels': train_labels
    })
    
    print("DataFrames created successfully!")
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print("\nClass distribution in training data:")
    print(train_df['labels'].value_counts())
    
    return train_df, test_df


def create_data_generators(train_df, valid_df, test_df, batch_size=32, img_size=(224, 224)):
    """
    Create data generators for training, validation, and testing.
    
    Args:
        train_df: Training DataFrame
        valid_df: Validation DataFrame
        test_df: Test DataFrame
        batch_size: Batch size for generators
        img_size: Image size tuple
        
    Returns:
        train_gen, valid_gen, test_gen: Data generators
    """
    # Training generator with augmentation
    train_gen_config = ImageDataGenerator(
        rescale=1./255,
        vertical_flip=True,
        horizontal_flip=True,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    
    # Validation and test generators without augmentation
    val_test_gen_config = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_gen_config.flow_from_dataframe(
        train_df,
        x_col='filenames',
        y_col='labels',
        target_size=img_size,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True,
        batch_size=batch_size,
        seed=88
    )
    
    valid_gen = val_test_gen_config.flow_from_dataframe(
        valid_df,
        x_col='filenames',
        y_col='labels',
        target_size=img_size,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True,
        batch_size=batch_size,
        seed=88
    )
    
    test_gen = val_test_gen_config.flow_from_dataframe(
        test_df,
        x_col='filenames',
        y_col='labels',
        target_size=img_size,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False,
        batch_size=len(test_df),  # Load all test data at once
        seed=88
    )
    
    return train_gen, valid_gen, test_gen


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def example_normal_preprocessing():
    """
    Example of how to use NORMAL preprocessing (RECOMMENDED).
    This is the standard approach without segmentation complexity.
    """
    print("\n" + "="*60)
    print("📚 USAGE EXAMPLE: NORMAL PREPROCESSING (RECOMMENDED)")
    print("="*60)
    print("""
# For normal image preprocessing (RECOMMENDED):
from src.data_preprocessing import choose_preprocessing_method, prepare_directories

# Setup directories
prepare_directories()

# Get your training image paths
train_list = get_train_list()  # Your function to get image paths

# Method 1: Simple normal preprocessing
process_and_save_train_data(train_list)

# Method 2: Normal preprocessing with enhancements
process_and_save_train_data(
    train_list, 
    enable_preprocessing=True,  # Apply basic enhancements
    apply_clahe=True           # Apply contrast enhancement
)

# Method 3: Using the wrapper function (EASIEST)
choose_preprocessing_method(
    train_list,
    use_segmentation=False,    # Normal preprocessing (RECOMMENDED)
    enable_enhancements=True,  # Apply image enhancements
    apply_clahe=True          # Apply contrast enhancement
)
    """)
    print("="*60)


def example_segmentation_preprocessing():
    """
    Example of segmentation preprocessing (NOT RECOMMENDED for beginners).
    """
    print("\n" + "="*60)
    print("⚠️  USAGE EXAMPLE: SEGMENTATION PREPROCESSING (ADVANCED)")
    print("="*60)
    print("""
# For segmentation-based preprocessing (ADVANCED - NOT RECOMMENDED):
from src.data_preprocessing import choose_preprocessing_method

# Setup directories
prepare_directories()

# Get your training image paths
train_list = get_train_list()  # Your function to get image paths

# Segmentation preprocessing (COMPLEX, SLOW)
choose_preprocessing_method(
    train_list,
    use_segmentation=True     # WARNING: Complex and time-consuming
)

# ⚠️ NOTE: Segmentation is much slower and more complex
# ⚠️ RECOMMENDATION: Use normal preprocessing instead
    """)
    print("="*60)


if __name__ == "__main__":
    print("🩸 Blood Cell Classification - Data Preprocessing Module")
    print("Choose your preprocessing method:")
    print("1. Normal preprocessing (RECOMMENDED)")
    print("2. Segmentation preprocessing (ADVANCED)")
    
    example_normal_preprocessing()
    example_segmentation_preprocessing()