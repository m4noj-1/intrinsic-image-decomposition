# INSTRUCTIONS:
# 1. Upload a file named "img.jpg" to Colab (use the folder icon on left)
# 2. Run this entire code
# 3. Done!
# ===========================================================================

# Install dependencies
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy", "matplotlib", "pillow"])

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

# Core Algorithm
class IntrinsicImageDecomposer:
    def __init__(self, sigma_low=40):
        self.sigma_low = sigma_low
    
    def _gaussian_filter_2d(self, img, sigma):
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        x = np.arange(kernel_size) - kernel_size // 2
        kernel_1d = np.exp(-x**2 / (2 * sigma**2))
        kernel_1d /= kernel_1d.sum()
        
        result = np.zeros_like(img, dtype=np.float64)
        temp = np.zeros_like(img, dtype=np.float64)
        
        pad = kernel_size // 2
        padded = np.pad(img, ((0, 0), (pad, pad)), mode='reflect')
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                temp[i, j] = np.sum(padded[i, j:j+kernel_size] * kernel_1d)
        
        padded = np.pad(temp, ((pad, pad), (0, 0)), mode='reflect')
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                result[i, j] = np.sum(padded[i:i+kernel_size, j] * kernel_1d)
        
        return result
    
    def decompose_grayscale(self, image):
        img_norm = image.astype(np.float64)
        img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-10)
        
        epsilon = 1e-10
        log_img = np.log(img_norm + epsilon)
        low_freq = self._gaussian_filter_2d(log_img, self.sigma_low)
        high_freq = log_img - low_freq
        
        illumination = np.exp(low_freq)
        reflectance = np.exp(high_freq)
        
        illumination = (illumination - illumination.min()) / (illumination.max() - illumination.min() + 1e-10)
        reflectance = (reflectance - reflectance.min()) / (reflectance.max() - reflectance.min() + 1e-10)
        
        return reflectance, illumination, log_img, low_freq, high_freq
    
    def decompose_color(self, image):
        h, w, c = image.shape
        img_norm = image.astype(np.float64)
        
        # Don't normalize globally - keep original color intensities
        img_norm = img_norm / 255.0 if img_norm.max() > 1.0 else img_norm
        img_norm = np.clip(img_norm, 0.01, 1.0)  # Avoid log issues
        
        # Work in log domain
        log_channels = np.log(img_norm)
        
        # Extract ONLY the illumination variation (the gradient)
        illumination_channels = np.zeros_like(log_channels)
        for ch in range(3):
            illumination_channels[:, :, ch] = self._gaussian_filter_2d(log_channels[:, :, ch], self.sigma_low)
        
        # Use joint illumination
        joint_illumination_log = np.mean(illumination_channels, axis=2)
        
        # Calculate the mean illumination level to preserve base brightness
        global_mean_log = np.mean(log_channels)
        illumination_mean_log = np.mean(joint_illumination_log)
        
        # Corrected reflectance: Remove illumination gradient but ADD BACK the global mean
        reflectance_channels = np.zeros_like(img_norm)
        for ch in range(3):
            # This keeps the base color and only removes shadows/highlights
            corrected_log = log_channels[:, :, ch] - joint_illumination_log + illumination_mean_log
            reflectance_channels[:, :, ch] = np.exp(corrected_log)
        
        # Clip to valid range
        reflectance_color = np.clip(reflectance_channels, 0, 1)
        illumination_color = np.exp(joint_illumination_log - illumination_mean_log)
        illumination_color = np.clip(illumination_color, 0, 1)
        
        return reflectance_color, illumination_color

# ============================================================================
# MAIN PROCESSING - LOADS img.jpg
# ============================================================================

print("\n" + "="*80)
print("PROCESSING img.jpg")
print("="*80 + "\n")

try:
    # Load image
    print("📂 Loading img.jpg...")
    img = Image.open('img.jpg')
    img_array = np.array(img)
    
    # Normalize
    if img_array.dtype == np.uint8:
        img_array = img_array.astype(np.float64) / 255.0
    
    print(f"✅ Image loaded: {img_array.shape}")
    
    # Check if color or grayscale
    is_color = len(img_array.shape) == 3 and img_array.shape[2] >= 3
    
    # Initialize decomposer (change sigma here if needed)
    SIGMA = 40  # CHANGE THIS VALUE: 20-30 (local light), 40-50 (default), 60-80 (gradual)
    decomposer = IntrinsicImageDecomposer(sigma_low=SIGMA)
    
    print(f"⚙  Using sigma = {SIGMA}")
    
    if is_color:
        print("\n🎨 Processing COLOR image...")
        
        # Handle RGBA -> RGB
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        
        start_time = time.time()
        reflectance, illumination = decomposer.decompose_color(img_array)
        elapsed = time.time() - start_time
        
        print(f"✅ Completed in {elapsed:.2f} seconds\n")
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(img_array)
        axes[0].set_title('Original Image', fontsize=16, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(reflectance)
        axes[1].set_title('Reflectance (Corrected)', fontsize=16, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(np.stack([illumination]*3, axis=2), cmap='hot')
        axes[2].set_title('Illumination Field', fontsize=16, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('result_color.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Save individual outputs
        Image.fromarray((reflectance * 255).astype(np.uint8)).save('reflectance_color.png')
        Image.fromarray((np.stack([illumination]*3, axis=2) * 255).astype(np.uint8)).save('illumination_color.png')
        
        print("💾 Saved files:")
        print("   - result_color.png (visualization)")
        print("   - reflectance_color.png (corrected image)")
        print("   - illumination_color.png (lighting field)")
        
    else:
        print("\n⚫ Processing GRAYSCALE image...")
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            img_gray = np.mean(img_array, axis=2)
        else:
            img_gray = img_array
        
        start_time = time.time()
        reflectance, illumination, log_img, low_freq, high_freq = decomposer.decompose_grayscale(img_gray)
        elapsed = time.time() - start_time
        
        print(f"✅ Completed in {elapsed:.2f} seconds\n")
        
        # Visualize - 6 panel view
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        axes[0, 0].imshow(img_gray, cmap='gray')
        axes[0, 0].set_title('Original Image I(x,y)', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(reflectance, cmap='gray')
        axes[0, 1].set_title('Reflectance R̂(x,y)', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(illumination, cmap='hot')
        axes[0, 2].set_title('Illumination L̂(x,y)', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(log_img, cmap='gray')
        axes[1, 0].set_title('Log Domain: log(I)', fontsize=12)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(low_freq, cmap='gray')
        axes[1, 1].set_title('Low Frequency: log(L)', fontsize=12)
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(high_freq, cmap='gray')
        axes[1, 2].set_title('High Frequency: log(R)', fontsize=12)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('result_gray.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Save individual outputs
        Image.fromarray((reflectance * 255).astype(np.uint8)).save('reflectance_gray.png')
        Image.fromarray((illumination * 255).astype(np.uint8)).save('illumination_gray.png')
        
        print("💾 Saved files:")
        print("   - result_gray.png (6-panel visualization)")
        print("   - reflectance_gray.png (corrected image)")
        print("   - illumination_gray.png (lighting field)")
    
    print("\n" + "="*80)
    print("✅ PROCESSING COMPLETE!")
    print("="*80)
    print("\n💡 TIP: Download files from the folder panel on the left (📁)")
    
except FileNotFoundError:
    print("❌ ERROR: img.jpg not found!")
    print("\n📝 TO FIX:")
    print("   1. Click the folder icon (📁) on the left sidebar")
    print("   2. Click the upload button (📤)")
    print("   3. Upload your image and name it 'img.jpg'")
    print("   4. Run this code again")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    print("\n📝 Make sure your image file is named exactly 'img.jpg'")

# ============================================================================
# BONUS: Want to use a different filename? Change it here:
# ============================================================================
# Just replace 'img.jpg' with your filename like 'photo.jpg' or 'test.png'
# in the line: img = Image.open('img.jpg')

# ============================================================================
