Stunting Detection Using CNN: 
A deep learning model for detecting stunting in children from grayscale images using Convolutional Neural Networks (CNN).

Overview: 
This project implements a binary classification model to identify stunting cases from medical images. The model achieves approximately 89% accuracy on test data with strong performance in detecting both normal and stunting cases.

Model Architecture
The CNN architecture consists of:
- 3 Convolutional blocks (32, 64, 128 filters)
- Batch Normalization after each Conv2D layer
- MaxPooling layers for dimension reduction
- Flatten layer followed by Dense layers
- Dropout (0.3) for regularization
- Sigmoid activation for binary classification

**Input:** 224×224 grayscale images  
**Output:** Binary classification (Normal vs Stunting)

Performance Metrics
Based on test set evaluation:
| Accuracy  | 88.89% |
| Precision | 87.63% |
| Recall    | 90.43% |
| F1-Score  | 89.01% |

Confusion Matrix
|                 | Predicted Normal | Predicted Stunting |
| Actual Normal   |        83        |         12         |
| Actual Stunting |         9        |         85         |

- **True Positives (TP):** 85 - Correctly detected stunting
- **True Negatives (TN):** 83 - Correctly detected normal
- **False Positives (FP):** 12 - Normal flagged as stunting
- **False Negatives (FN):** 9 - Stunting cases missed

## Project Structure
```
├── Main.py                    # Main training pipeline
├── ConfigHyperParameter.py    # Hyperparameters configuration
├── ModelArchitect.py          # CNN model definition
├── ModelTraining.py           # Training logic with callbacks
├── Data_Loader.py             # Dataset loading and augmentation
├── PreProcess_IMG.py          # Image preprocessing utilities
├── Evaluation.py              # Model evaluation metrics
├── Prediction.py              # Single image prediction
├── SavingModel.py             # Model saving/loading utilities
├── Plotting.py                # Training history visualization
├── ConfusionMatrix.py         # Confusion matrix visualization
├── class_indices.json         # Class label mapping
└── dataimg/                   # Dataset directory
    ├── train/
    ├── valid/
    └── test/
```

Requirements
```
tensorflow>=2.10.0
numpy
matplotlib
scikit-learn
pillow
```
Install dependencies:
```bash
pip install tensorflow numpy matplotlib scikit-learn pillow
```

Dataset Structure
Organize your dataset in the following structure:
```
dataimg/
├── train/
│   ├── normal/
│   │   ├── image1.jpg
│   │   └── ...
│   └── stunting/
│       ├── image1.jpg
│       └── ...
├── valid/
│   ├── normal/
│   └── stunting/
└── test/
    ├── normal/
    └── stunting/
```
Usage
Training the Model

Run the complete training pipeline:

```bash
python Main.py
```

This will:
1. Preprocess all images (resize to 224×224, convert to grayscale)
2. Load training, validation, and test datasets
3. Define and compile the CNN model
4. Train with data augmentation
5. Evaluate on test set
6. Save the trained model as `stunting_model.keras`
7. Generate training history plots and confusion matrix

Making Predictions
Predict on a single image:
```bash
python Prediction.py
```
Then enter the image path when prompted.

Output example:
```
============================================================
STUNTING DETECTION RESULT
============================================================
Image: test_image.jpg
Prediction: STUNTING
Confidence: 92.45%
Raw Score: 0.9245
WARNING: Stunting detected!
============================================================
```

Hyperparameters
Default configuration in `ConfigHyperParameter.py`:
```python
config = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50,
    'input_shape': (224, 224, 1),
    'target_size': (224, 224),
    'dropout_rate': 0.3
}
```
Data Augmentation
Training data is augmented with:
- Rotation (±20°)
- Brightness adjustment (0.8-1.2)
- Width/height shifts (±20%)
- Shear transformation (20%)
- Zoom (±20%)
- Horizontal flip
- Rescaling (0-1 normalization)
Validation and test sets are only rescaled without augmentation.

Training Callbacks
- **EarlyStopping:** Stops training if validation performance doesn't improve for 5 epochs
- **ReduceLROnPlateau:** Reduces learning rate by 50% if no improvement for 3 epochs
- **ModelCheckpoint:** Saves best model based on validation performance

Output Files
After training, the following files are generated:
- `stunting_model.keras` - Trained model
- `best_model.h5` - Best model checkpoint
- `class_indices.json` - Class label mapping
- `training_history.png` - Accuracy and loss plots
- `model_structure.png` - Model architecture diagram
- `confusion_matrix.png` - Confusion matrix visualization

Model Loading
Load a saved model:
```python
from SavingModel import loadModel

model = loadModel('stunting_model.keras')
```
Results Interpretation
The model shows:
- **High recall (90.43%):** Successfully detects 90% of stunting cases, minimizing missed diagnoses
- **Good precision (87.63%):** Reliable when predicting stunting with low false alarm rate
- **Balanced performance:** Strong F1-score indicates good balance between precision and recall

Notes
- All images are converted to grayscale for consistent processing
- Images are automatically resized to 224×224 pixels
- The model uses binary crossentropy loss for two-class classification
- Threshold for classification: 0.5 (probability > 0.5 → Stunting)

Future Improvements
Potential enhancements:
- Implement cross-validation for more robust evaluation
- Add more data augmentation techniques
- Experiment with transfer learning (VGG16, ResNet)
- Add grad-CAM visualization for model interpretability
- Deploy as web API for real-time predictions

Authors : Johannes Aaron Framon
