import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.cluster import KMeans
from skimage import morphology
from scipy import ndimage as ndi

def explain_lab_color_space():
    """
    Explain LAB color space and why we use the 'a' channel for blood cell segmentation.
    """
    print("LAB Color Space Explanation:")
    print("=" * 50)
    print("L Channel: Lightness (0=black, 100=white) - GRAYSCALE")
    print("A Channel: Green-Red axis (-128=green, +127=red) - NOT GRAYSCALE!")
    print("B Channel: Blue-Yellow axis (-128=blue, +127=yellow) - NOT GRAYSCALE!")
    print()
    print("Why A Channel for Blood Cells?")
    print("- Blood cells are reddish (positive A values)")
    print("- Background is often greenish/neutral (lower A values)")
    print("- A channel provides best contrast between cell and background")
    print("=" * 50)

def visualize_lab_channels_detailed(image):
    """
    Show detailed LAB channel analysis with histograms and color maps.
    """
    # Convert to LAB
    i_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(i_lab)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Detailed LAB Color Analysis', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original RGB Image')
    axes[0, 0].axis('off')
    
    # L channel (grayscale)
    axes[0, 1].imshow(l, cmap='gray')
    axes[0, 1].set_title('L Channel (Lightness)\nGrayscale: 0-100')
    axes[0, 1].axis('off')
    
    # L histogram
    axes[0, 2].hist(l.flatten(), bins=50, alpha=0.7, color='gray')
    axes[0, 2].set_title('L Channel Histogram')
    axes[0, 2].set_xlabel('Lightness Value')
    
    # A channel (green-red axis)
    axes[1, 0].imshow(a, cmap='RdYlGn_r')  # Red-Green colormap
    axes[1, 0].set_title('A Channel (Green-Red)\nGreen(-) to Red(+)')
    axes[1, 0].axis('off')
    
    # A channel grayscale view
    axes[1, 1].imshow(a, cmap='gray')
    axes[1, 1].set_title('A Channel (Grayscale View)\nfor K-means Processing')
    axes[1, 1].axis('off')
    
    # A histogram
    axes[1, 2].hist(a.flatten(), bins=50, alpha=0.7, color='red')
    axes[1, 2].set_title('A Channel Histogram')
    axes[1, 2].set_xlabel('Green(-) ← → Red(+)')
    axes[1, 2].axvline(x=np.mean(a), color='blue', linestyle='--', label='Mean')
    axes[1, 2].legend()
    
    # B channel (blue-yellow axis)
    axes[2, 0].imshow(b, cmap='RdYlBu_r')  # Blue-Yellow colormap
    axes[2, 0].set_title('B Channel (Blue-Yellow)\nBlue(-) to Yellow(+)')
    axes[2, 0].axis('off')
    
    # B channel grayscale
    axes[2, 1].imshow(b, cmap='gray')
    axes[2, 1].set_title('B Channel (Grayscale View)')
    axes[2, 1].axis('off')
    
    # B histogram
    axes[2, 2].hist(b.flatten(), bins=50, alpha=0.7, color='blue')
    axes[2, 2].set_title('B Channel Histogram')
    axes[2, 2].set_xlabel('Blue(-) ← → Yellow(+)')
    
    plt.tight_layout()
    plt.show()
    
    print(f"A Channel Stats: Min={a.min()}, Max={a.max()}, Mean={a.mean():.1f}")
    print(f"B Channel Stats: Min={b.min()}, Max={b.max()}, Mean={b.mean():.1f}")
    print(f"L Channel Stats: Min={l.min()}, Max={l.max()}, Mean={l.mean():.1f}")

def find_optimal_clusters(image, max_clusters=10):
    """
    Find optimal number of clusters using elbow method and silhouette score.
    """
    from sklearn.metrics import silhouette_score
    
    # Convert to LAB and get A channel
    i_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(i_lab)
    i2 = a.reshape(-1, 1)
    
    # Sample data if too large (for performance)
    if len(i2) > 10000:
        indices = np.random.choice(len(i2), 10000, replace=False)
        i2_sample = i2[indices]
    else:
        i2_sample = i2
    
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_clusters + 1)
    
    print("Testing different cluster numbers...")
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(i2_sample)
        inertias.append(kmeans.inertia_)
        
        if len(i2_sample) < 10000:  # Only calculate silhouette for smaller datasets
            score = silhouette_score(i2_sample, kmeans.labels_)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(0)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Elbow method
    ax1.plot(K_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia (Within-cluster sum of squares)')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True)
    
    # Silhouette score
    if any(s > 0 for s in silhouette_scores):
        ax2.plot(K_range, silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True)
    else:
        ax2.text(0.5, 0.5, 'Silhouette analysis\nskipped for large dataset', 
                ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.show()
    
    # Recommend optimal k
    if silhouette_scores and max(silhouette_scores) > 0:
        optimal_k = K_range[np.argmax(silhouette_scores)]
        print(f"Recommended k based on silhouette score: {optimal_k}")
    else:
        print("Recommendation: k=7 (commonly works well for blood cells)")
    
    return K_range, inertias, silhouette_scores

def find_optimal_threshold(image, min_thresh=100, max_thresh=200):
    """
    Find optimal threshold using Otsu's method and testing multiple values.
    """
    # Get K-means result first
    i_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(i_lab)
    i2 = a.reshape(-1, 1)
    
    km = KMeans(n_clusters=7, random_state=42).fit(i2)
    p2s = km.cluster_centers_[km.labels_]
    ic = p2s.reshape(a.shape[0], a.shape[1])
    ic = ic.astype(np.uint8)
    
    # Try Otsu's automatic threshold
    otsu_thresh, _ = cv2.threshold(ic, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Test range of thresholds
    thresholds = range(min_thresh, max_thresh, 5)
    results = []
    
    print("Testing different threshold values...")
    for thresh in thresholds:
        _, binary = cv2.threshold(ic, thresh, 255, cv2.THRESH_BINARY)
        
        # Calculate metrics
        white_pixels = np.sum(binary == 255)
        total_pixels = binary.size
        white_ratio = white_pixels / total_pixels
        
        # Simple quality metric (adjust based on your needs)
        quality_score = white_ratio if 0.1 < white_ratio < 0.6 else 0
        
        results.append({
            'threshold': thresh,
            'white_ratio': white_ratio,
            'quality_score': quality_score
        })
    
    # Plot results
    threshs = [r['threshold'] for r in results]
    ratios = [r['white_ratio'] for r in results]
    scores = [r['quality_score'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(threshs, ratios, 'b-o', label='White pixel ratio')
    ax1.axvline(x=otsu_thresh, color='r', linestyle='--', label=f'Otsu threshold: {otsu_thresh}')
    ax1.axvline(x=141, color='g', linestyle='--', label='Current threshold: 141')
    ax1.set_xlabel('Threshold Value')
    ax1.set_ylabel('White Pixel Ratio')
    ax1.set_title('Threshold vs White Pixel Ratio')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(threshs, scores, 'r-o')
    ax2.set_xlabel('Threshold Value')
    ax2.set_ylabel('Quality Score')
    ax2.set_title('Threshold Quality Analysis')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Recommendations
    best_threshold = max(results, key=lambda x: x['quality_score'])['threshold']
    print(f"Otsu's automatic threshold: {otsu_thresh}")
    print(f"Best threshold based on quality score: {best_threshold}")
    print(f"Current threshold: 141")
    
    return results, otsu_thresh, best_threshold

def test_morphological_parameters(image):
    """
    Test different parameters for morphological operations.
    """
    # Get binary mask first (using current method)
    i_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(i_lab)
    i2 = a.reshape(-1, 1)
    km = KMeans(n_clusters=7, random_state=42).fit(i2)
    p2s = km.cluster_centers_[km.labels_]
    ic = p2s.reshape(a.shape[0], a.shape[1]).astype(np.uint8)
    _, binary = cv2.threshold(ic, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Fill holes first
    filled = ndi.binary_fill_holes(binary)
    
    # Test different parameters
    small_obj_sizes = [50, 100, 200, 300, 500]
    small_hole_sizes = [100, 200, 250, 400, 600]
    
    fig, axes = plt.subplots(len(small_obj_sizes), len(small_hole_sizes), 
                            figsize=(20, 16))
    fig.suptitle('Morphological Parameters Testing\n(Rows: remove_small_objects, Cols: remove_small_holes)', 
                fontsize=14)
    
    for i, obj_size in enumerate(small_obj_sizes):
        for j, hole_size in enumerate(small_hole_sizes):
            # Apply morphological operations
            m1 = morphology.remove_small_objects(filled, obj_size)
            m2 = morphology.remove_small_holes(m1, hole_size)
            
            # Apply to original image
            result = cv2.bitwise_and(image, image, mask=m2.astype(np.uint8))
            
            axes[i, j].imshow(result)
            axes[i, j].set_title(f'obj:{obj_size}, hole:{hole_size}', fontsize=8)
            axes[i, j].axis('off')
            
            # Highlight current parameters
            if obj_size == 200 and hole_size == 250:
                for spine in axes[i, j].spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
    
    plt.tight_layout()
    plt.show()
    
    print("Current parameters are highlighted in RED")
    print("Choose parameters that:")
    print("- Remove noise but keep the main cell")
    print("- Don't over-segment the cell")
    print("- Preserve cell structure")

def comprehensive_parameter_analysis(image):
    """
    Comprehensive analysis of all segmentation parameters.
    """
    print("🔬 COMPREHENSIVE SEGMENTATION PARAMETER ANALYSIS")
    print("=" * 60)
    
    # 1. LAB Color Analysis
    print("1. Analyzing LAB color channels...")
    visualize_lab_channels_detailed(image)
    
    # 2. Cluster Analysis
    print("\n2. Finding optimal number of clusters...")
    find_optimal_clusters(image)
    
    # 3. Threshold Analysis
    print("\n3. Finding optimal threshold...")
    find_optimal_threshold(image)
    
    # 4. Morphological Parameters
    print("\n4. Testing morphological parameters...")
    test_morphological_parameters(image)
    
    print("\n" + "=" * 60)
    print("PARAMETER EXPLANATIONS:")
    print("=" * 60)
    print("""
    WHY THESE VALUES:
    
    🎯 K-means clusters = 7:
    - Blood cells typically have 3-4 main color regions
    - Background has 2-3 color variations
    - Extra clusters help separate subtle differences
    - 7 is empirically found to work well for blood cells
    
    🎯 Binary threshold (Otsu's method):
    - Automatically determines optimal threshold value
    - Based on histogram analysis to minimize intra-class variance
    - Separates reddish cells from greenish background optimally
    - No manual assumption needed - theoretically grounded
    
    🎯 Remove small objects (min_size = 200):
    - Removes noise and artifacts smaller than 200 pixels
    - Typical blood cell is >500 pixels at 224x224 resolution
    - Adjust based on your image resolution and cell size
    
    🎯 Remove small holes (area = 250):
    - Fills gaps inside cells (nucleus regions, artifacts)
    - 250 pixels ≈ small internal structures
    - Prevents over-fragmentation of cells
    
    🔧 HOW TO OPTIMIZE:
    1. Use find_optimal_clusters() for your dataset
    2. Use find_optimal_threshold() or Otsu's method
    3. Adjust morphological parameters based on cell size
    4. Test on multiple representative images
    """)
    print("=" * 60)

def visualize_segmentation_steps(image, save_path=None):
    """
    Visualize each step of the blood cell segmentation process.
    
    Args:
        image: Input RGB image
        save_path: Optional path to save the visualization
        
    Returns:
        Dictionary containing all intermediate results
    """
    # Store results for return
    results = {}
    
    # Step 1: Original image
    results['original'] = image.copy()
    
    # Step 2: Convert RGB to LAB color space
    i_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(i_lab)
    results['lab_l'] = l
    results['lab_a'] = a
    results['lab_b'] = b
    
    # Step 3: Reshape for K-means clustering
    i2 = a.reshape(a.shape[0] * a.shape[1], 1)
    
    # Step 4: Apply K-means clustering
    km = KMeans(n_clusters=7, random_state=42).fit(i2)
    p2s = km.cluster_centers_[km.labels_]
    ic = p2s.reshape(a.shape[0], a.shape[1])
    ic = ic.astype(np.uint8)
    results['kmeans_result'] = ic
    
    # Step 5: Binary thresholding using Otsu's method
    otsu_thresh, t = cv2.threshold(ic, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results['binary_threshold'] = t
    results['otsu_threshold'] = otsu_thresh
    
    # Step 6: Fill holes
    fh = ndi.binary_fill_holes(t)
    results['filled_holes'] = fh.astype(np.uint8) * 255
    
    # Step 7: Remove small objects
    m1 = morphology.remove_small_objects(fh, 200)
    results['removed_small_objects'] = m1.astype(np.uint8) * 255
    
    # Step 8: Remove small holes
    m2 = morphology.remove_small_holes(m1, 250)
    results['final_mask'] = m2.astype(np.uint8) * 255
    
    # Step 9: Apply mask to original image
    out = cv2.bitwise_and(image, image, mask=m2.astype(np.uint8))
    results['segmented_result'] = out
    
    # Create visualization
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Blood Cell Segmentation - Step by Step Process', fontsize=16, fontweight='bold')
    
    # Plot each step
    axes[0, 0].imshow(results['original'])
    axes[0, 0].set_title('1. Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(results['lab_l'])
    axes[0, 1].set_title('2. LAB - L Channel')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(results['lab_a'])
    axes[0, 2].set_title('3. LAB - A Channel')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(results['lab_b'])
    axes[0, 3].set_title('4. LAB - B Channel')
    axes[0, 3].axis('off')
    
    axes[1, 0].imshow(results['kmeans_result'])
    axes[1, 0].set_title('5. K-means Clustering\n(7 clusters on A channel)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(results['binary_threshold'])
    axes[1, 1].set_title(f'6. Binary Threshold\n(Otsu = {results["otsu_threshold"]})')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(results['filled_holes'])
    axes[1, 2].set_title('7. Fill Holes')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(results['removed_small_objects'])
    axes[1, 3].set_title('8. Remove Small Objects\n(min_size = 200)')
    axes[1, 3].axis('off')
    
    axes[2, 0].imshow(results['final_mask'])
    axes[2, 0].set_title('9. Remove Small Holes\n(area_threshold = 250)')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(results['segmented_result'])
    axes[2, 1].set_title('10. Final Segmented Result')
    axes[2, 1].axis('off')
    
    # Show mask overlay on original
    overlay = results['original'].copy()
    mask_colored = np.zeros_like(overlay)
    mask_colored[:, :, 1] = results['final_mask']  # Green channel
    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
    axes[2, 2].imshow(overlay)
    axes[2, 2].set_title('11. Mask Overlay\n(Green = Detected Cell)')
    axes[2, 2].axis('off')
    
    # Show comparison
    comparison = np.hstack([results['original'], results['segmented_result']])
    axes[2, 3].imshow(comparison)
    axes[2, 3].set_title('12. Before vs After')
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()
    
    # Print Otsu threshold information
    print(f"\n🎯 Otsu Threshold Analysis:")
    print(f"   Optimal threshold value: {results['otsu_threshold']}")
    print(f"   This value was automatically calculated based on histogram analysis")
    print(f"   to minimize intra-class variance between foreground and background.")
    
    return results

# Updated segment_blood_cell function with optional visualization
def segment_blood_cell(image, visualize=False, save_visualization=None):
    """
    Segment blood cell from background using K-means clustering and morphological operations.
    
    Args:
        image: Input RGB image
        visualize: If True, shows step-by-step visualization
        save_visualization: Path to save visualization (optional)
        
    Returns:
        Segmented image with background removed
    """
    if visualize:
        results = visualize_segmentation_steps(image, save_visualization)
        return results['segmented_result']
    
    # Original segmentation code (unchanged)
    # Convert RGB to LAB color space
    i_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(i_lab)
    
    # Reshape for K-means clustering
    i2 = a.reshape(a.shape[0] * a.shape[1], 1)
    
    # Apply K-means clustering
    km = KMeans(n_clusters=17, random_state=42).fit(i2)
    p2s = km.cluster_centers_[km.labels_]
    ic = p2s.reshape(a.shape[0], a.shape[1])
    ic = ic.astype(np.uint8)
    
    # Binary thresholding using Otsu's method
    otsu_thresh, t = cv2.threshold(ic, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations
    fh = ndi.binary_fill_holes(t)
    m1 = morphology.remove_small_objects(fh, 200)
    m2 = morphology.remove_small_holes(m1, 250)
    m2 = m2.astype(np.uint8)
    
    # Apply mask to original image
    out = cv2.bitwise_and(image, image, mask=m2)
    
    return out

# Example usage function
def demo_segmentation_visualization():
    """
    Demo function to test the segmentation visualization.
    """
    print("Blood Cell Segmentation Visualization Demo")
    print("=" * 50)
    
    # Example usage:
    # Load your image
    # image_path = "path/to/your/blood_cell_image.jpg"
    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    
    # Method 1: Segmentation with visualization
    # segmented = segment_blood_cell(image, visualize=True)
    
    # Method 2: Segmentation with saved visualization
    # segmented = segment_blood_cell(image, visualize=True, 
    #                              save_visualization="segmentation_steps.png")
    
    # Method 3: Just get step-by-step results
    # results = visualize_segmentation_steps(image)
    
    print("\nTo use the visualization:")
    print("1. Load your blood cell image")
    print("2. Call segment_blood_cell(image, visualize=True)")
    print("3. Or call visualize_segmentation_steps(image) directly")