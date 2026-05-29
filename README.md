# Post-Storm Damage Assessment AI

A deep learning application that automatically detects and classifies hurricane and post-storm damage from aerial imagery using ResNet50.

## Overview

This project uses a trained ResNet50 convolutional neural network to analyze high-resolution aerial drone and satellite images to identify areas damaged by hurricanes and severe weather. The model classifies image patches as either **damaged** or **undamaged**, providing a rapid assessment tool for disaster response and damage evaluation.

## Features

- **AI-Powered Detection**: Uses transfer learning with ResNet50 for accurate damage classification
- **User-Friendly GUI**: Tkinter-based interface for easy image upload and analysis
- **GPU Acceleration**: Automatically detects and utilizes CUDA for faster processing
- **Flexible Image Support**: Accepts multiple image formats (JPG, JPEG, PNG, TIF, TIFF, WEBP)
- **Real-time Results**: Instant damage assessment with visual feedback (Red for damaged, Green for undamaged)
- **Pre-trained Model**: Includes a pre-trained model weights file for immediate predictions

## Project Structure

```
Post-storm-damage/
├── README.md                          # Project documentation
├── train.py                          # Training script with transfer learning
├── predict.py                        # Simple prediction script
├── predict_v2.py                     # Enhanced GUI-based prediction tool
├── check_files.py                    # Utility to verify repository structure
├── hurricane_damage_resnet50.pth     # Pre-trained model weights (94MB)
├── data/                             # Dataset directory
│   ├── train/                        # Training images (subdirectories: damaged/, undamaged/)
│   └── val/                          # Validation images (subdirectories: damaged/, undamaged/)
└── screenshots/                      # Application screenshots
```

## Technical Stack

- **Framework**: PyTorch with torchvision
- **Model Architecture**: ResNet50 (pre-trained on ImageNet)
- **GUI Framework**: Tkinter
- **Image Processing**: PIL (Pillow)
- **Hardware Support**: CUDA (GPU) with fallback to CPU

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/gnilwobnnej/Post-storm-damage.git
   cd Post-storm-damage
   ```

2. **Install dependencies**:
   ```bash
   pip install torch torchvision pillow
   ```

3. **Verify model file exists**:
   ```bash
   python check_files.py
   ```

## Usage

### Option 1: GUI-Based Analysis (Recommended)

Run the interactive prediction tool:

```bash
python predict_v2.py
```

**Steps**:
1. Click "Upload & Analyze Patch"
2. Select an aerial image from your computer
3. The model will analyze the image and display results
4. Red text = "DAMAGED", Green text = "UNDAMAGED"

### Option 2: Command-line Prediction

For batch processing or integration:

```bash
python predict.py
```

### Option 3: Train Your Own Model

To train with your own dataset:

```bash
python train.py
```

**Dataset structure required**:
```
data/
├── train/
│   ├── damaged/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── undamaged/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── val/
    ├── damaged/
    └── undamaged/
```

## Model Details

### Architecture
- **Base Model**: ResNet50 (pre-trained on ImageNet)
- **Custom Output Layer**: 2-class classifier (damaged/undamaged)
- **Input Size**: 224×224 pixels
- **Output**: Binary classification with confidence scores

### Training Configuration
- **Batch Size**: 32
- **Epochs**: 10
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Function**: Cross Entropy Loss
- **Data Augmentation**: Random crops, horizontal flips, and 15° rotations

### Image Processing Pipeline
1. Resize to 256×256 pixels
2. Center crop to 224×224 pixels
3. Convert to tensor and normalize
4. ImageNet normalization (mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])

## Application Screenshots

### Main Interface
![Hurricane Damage Detector - Main Screen](Screenshot%202026-05-28%20205036.png)

### Analysis in Progress
![Analysis Interface](Screenshot%202026-05-24%20094149.png)

### Results Display
![Results Output](Screenshot%202026-05-28%20205259.png)

## How It Works

### Prediction Pipeline

1. **Image Upload**: User selects an aerial image via file dialog
2. **Preprocessing**: Image is resized and normalized to model specifications
3. **Model Inference**: Image passes through ResNet50 for feature extraction
4. **Classification**: Final layer outputs confidence scores for each class
5. **Result Display**: Prediction rendered with color-coded visual feedback

### Key Code Components

**predict_v2.py** - Contains:
- `predict_damage()`: Core AI inference function with GPU/CPU support
- `select_and_analyze_image()`: GUI event handler for image selection and analysis
- Tkinter UI initialization and layout specification

**train.py** - Implements:
- Transfer learning setup with frozen base layers
- Data augmentation for better generalization
- Training loop with validation metrics
- Model checkpoint saving

## Performance

- **Input Processing**: Real-time image loading and preprocessing
- **Inference Speed**: < 1 second on GPU, 2-5 seconds on CPU
- **Memory Usage**: ~400MB (GPU with model weights)
- **Supported Image Sizes**: Up to 8K resolution (automatically resized)

## Requirements

- Python 3.8+
- PyTorch 1.9+
- torchvision 0.10+
- Pillow 8.0+
- tkinter (usually included with Python)
- CUDA 11.0+ (optional, for GPU acceleration)

## Future Enhancements

- Batch processing for multiple images
- API endpoint for integration with disaster management systems
- Multi-class classification (damage severity levels)
- Real-time video stream analysis
- Model optimization and quantization for edge deployment
- Web interface using Flask/Django

## License

This project is open source and available for academic and commercial use.

## Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and create pull requests.

## Contact

For questions or collaboration inquiries, please reach out to the repository owner.

---

**Note**: This AI model is a demonstration tool for hurricane damage assessment. For production disaster response systems, validation with real damage data and expert review is recommended before deployment.
