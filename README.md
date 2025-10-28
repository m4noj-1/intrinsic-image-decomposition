A Python implementation designed for Google Colab that decomposes images into intrinsic components (reflectance and illumination) using frequency-domain analysis. This project solves the classic computer vision problem of recovering true surface texture under non-uniform lighting conditions.


📋 Problem Statement
Given a single photograph of a textured surface under uneven illumination, reconstruct the true texture pattern as if it were uniformly lit, without any prior knowledge of the lighting source or direction.
Image Formation Model
The observed image follows the equation:
I(x,y) = R(x,y) × L(x,y)
Where:
I(x,y) = Observed image intensity
R(x,y) = Reflectance (intrinsic texture pattern) ← Goal
L(x,y) = Illumination (smooth, slowly varying function)


🎯 Features

✅ Google Colab Ready – Zero setup, runs in browser
✅ Grayscale & Color Image Support – Works with both image types
✅ Custom Gaussian Filter – No built-in filters used (as per requirements)
✅ Log-Domain Decomposition – Mathematically sound frequency separation
✅ Multi-Channel Correction – Joint illumination estimation for color images
✅ Real-time Visualization – Side-by-side comparison of results
✅ Auto-Installation – Dependencies install automatically

🚀 Quick Start (Google Colab)
Open Google Colab: colab.research.google.com
Upload the notebook or copy-paste the code into a new notebook
Upload your image:

Click the folder icon (📁) in the left sidebar
Click upload button (📤)
Upload your image and name it img.jpg


Run all cells (Runtime → Run all)
Download results from the files panel.

📸 Example Results
The algorithm successfully separates texture from lighting in various conditions:
### Example 1: Fabric Texture Under Desk Lamp
<img width="1564" height="710" alt="image" src="https://github.com/user-attachments/assets/c090f667-a218-48c9-9254-2cf622efb681" />

### Example 2: Image of a Paper Under Desk Lamp
<img width="1667" height="624" alt="image" src="https://github.com/user-attachments/assets/ffdc9253-ad52-4b06-b738-a09985eba255" />




