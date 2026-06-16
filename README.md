# Fetal Ultrasound (USG) Image Segmentation

This project is a Computer Vision implementation for the automatic segmentation and measurement of fetal dimensions in ultrasonography (USG) images. The system is built entirely using traditional image processing techniques without relying on Machine Learning or Deep Learning models, making it highly lightweight and fast to execute.

The algorithm is specifically designed to tackle common challenges in medical imaging, such as speckle noise, acoustic shadows, and blurred tissue boundaries between the amniotic fluid, uterine wall, and the fetal body.

##  Key Features
* **Interactive Polygonal ROI:** Users can freely select the target area (the fetus) using mouse clicks (Freehand Polygon). This feature effectively isolates the fetus from the uterine wall, which often interferes with the thresholding process.
* **Robust Denoising & Enhancement:** Utilizes Wavelet Denoising (PyWavelets) to reduce speckle noise and CLAHE (Contrast Limited Adaptive Histogram Equalization) to clarify fetal tissue boundaries.
* **Smart Otsu Thresholding:** Applies Otsu-based color separation with calibrated tolerance thresholds to preserve dim fetal body parts (such as the leg area).
* **"Sandwich" Morphology Pipeline:** Employs a custom large-scale morphological technique (Closing -> Opening -> Closing) to reconnect disconnected limbs, erode outer noise, and solidify the overall fetal shape.
* **Automated Measurement:** Automatically calculates and extracts fetal measurement metrics in pixels (Area, Width, Height, and Diagonal) using Bounding Rect and the largest contour extraction.

##  Technologies & Libraries
* **Python 3.x**
* **OpenCV** (`cv2`) - Morphological execution, thresholding, and interactive GUI.
* **NumPy** - Matrix computation and pixel array manipulation.
* **Matplotlib** - Visualization of the segmentation pipeline.
* **PyWavelets** (`pywt`) - Wavelet transformation for medical image denoising.

##  How to Run the Project

1. **Install Dependencies**
   Make sure you are in your Virtual Environment (optional but recommended), then run:
   ```bash
   pip install opencv-python numpy matplotlib PyWavelets
2. **Directory Structure**
    Ensure your project folder follows this exact structure before running the program:
    ```bash
    📁 root_directory/
    ├── main.py
    ├── enhancement.py
    ├── segmentation.py
    ├── measurement.py
    ├── 📁 dataset/        <-- Place your USG image / dataset
    └── 📁 output/
        ├── 📁 enhancement/
        └── 📁 segmentation/
3. **Execution**
   ```bash
   python main.py