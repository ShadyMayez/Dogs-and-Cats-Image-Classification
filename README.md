# Dogs and Cats Image Classification (CNN)

## Project Overview and Purpose
This project implements a Deep Learning model to distinguish between images of dogs and cats. It serves as a comprehensive example of building, training, and regularizing a Convolutional Neural Network (CNN) to achieve high accuracy on binary image classification tasks.

## Key Technologies and Libraries
- **Framework**: TensorFlow / Keras
- **Image Processing**: OpenCV, Matplotlib, PIL
- **Data Handling**: NumPy, Pandas
- **Regularization**: L2 Regularization, Dropout, Batch Normalization

## Model Architecture and Methodology
### Architecture
The model is a `Sequential` CNN consisting of:
- **Feature Extraction**: Multiple `Conv2D` layers with increasing filters (32, 64, 128) paired with `MaxPooling2D` to capture hierarchical patterns.
- **Stability**: `BatchNormalization` is used after convolutional layers to speed up training and provide stability.
- **Regularization**: `L2 Regularization` and `Dropout` (0.5) are implemented to prevent the model from overfitting on the training data.
- **Classification**: A `Flatten` layer followed by a `Dense` layer and a final output neuron with a `Sigmoid` activation.



### Workflow
1. **Data Augmentation**: Used `ImageDataGenerator` to perform real-time data augmentation (rotation, shifting, shearing, and zooming) to improve model generalization.
2. **Callbacks**: Implemented `EarlyStopping` (monitoring validation loss) and `ModelCheckpoint` to save only the best iteration of the model.
3. **Training**: Compiled using the `Adam` optimizer and `binary_crossentropy` loss.

## Results and Insights
- **Performance**: The model shows a steady decrease in loss and an increase in accuracy across epochs, demonstrating effective learning.
- **Evaluation**: The notebook includes a visualization of training vs. validation accuracy/loss curves to diagnose model fit.
- **Predictions**: A custom prediction script is included to test the model on individual images.

## How to Run
1. **Dataset**: This project is designed for the Kaggle "Dogs vs. Cats" competition dataset. Ensure your data is organized into `train` and `test` directories.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
