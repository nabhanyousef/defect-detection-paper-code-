Casting Defect Detection with Lightweight CNN and PhysAugNet-Inspired Preprocessing
This project demonstrates a lightweight yet powerful image classification pipeline for automated defect detection in metal castings, using a combination of contrast enhancement, noise suppression, geometric augmentation, and a simple CNN classifier. The framework is ideal for industrial-grade real-time inspection systems and optimized for edge devices.

📂 Dataset Structure
Organize your dataset with one folder per defect type. Each folder should contain .jpg images of castings:

Copy
Edit
defect_dataset/
├── Porosity/
│   ├── img1.jpg
│   └── ...
├── Crack/
├── Slag/
└── Flash/
Each image should be in grayscale (or will be converted internally).

🧠 Project Features
CLAHE (Adaptive Histogram Equalization) for illumination normalization

Non-local means filtering for artifact suppression

Geometric augmentation using random affine transforms

Phong-shading simulation for specular highlights

Simple CNN model trained on dynamically preprocessed patches

Achieves high accuracy with minimal memory and compute load

🛠️ Requirements
Install the following Python packages (recommended: Python 3.8+):

bash
Copy
Edit
pip install torch torchvision opencv-python numpy
🚀 How to Run the Training
Place your dataset in the format shown above.

Update the dataset path in the training function:

python
Copy
Edit
# Inside the script
train_model("/path/to/your/defect_dataset")
Run the script:

bash
Copy
Edit
python defect_detection_train.py
During training, you'll see per-epoch loss and validation accuracy printed to the console.

📊 Output
Training loss and validation accuracy

Can be easily extended to save models or visualize predictions

Real-time capable: works well on Jetson devices and CPU-only machines

🧪 Future Enhancements
Export trained model for real-time inference

Add segmentation (e.g., U-Net or DeepLabV3)

Integrate GUI for uploading and classifying new images

Deploy on edge hardware like NVIDIA Jetson or Raspberry Pi

👨‍🔬 Author
Developed by Nabhan Yousef as part of a research initiative on intelligent quality control systems in smart manufacturing environments.
