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


def process_and_save_train_data(train_list, output_dir="temp_data/prepared_data"):
    """
    Process and save training data with segmentation augmentation.
    Each original sample creates two samples: original and segmented.
    
    Args:
        train_list: List of training image paths
        output_dir: Directory to save processed training images
    """
    print("Processing training data with segmentation...")
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