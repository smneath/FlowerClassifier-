# TensorFlow Flower Classification

## Overview

Welcome to the TensorFlow Flower Classification project! This repository presents a comprehensive flower image classification system using TensorFlow and Keras. The project's primary objective is to achieve accurate classification of flower images into five distinct classes: daisy, dandelion, roses, sunflowers, and tulips. This README provides an in-depth overview of the project structure, technologies used, and step-by-step instructions for usage.

## Project Structure

### Data Preprocessing

The project begins by loading and preprocessing the flower dataset located at `E:\Project\Machine Learing\paid\flowers`. The dataset is structured into five categories, each representing a specific type of flower. The preprocessing pipeline involves the following steps:

1. **Data Loading:** Images are loaded using OpenCV from the specified directory.
2. **Labeling:** Each image is associated with a label corresponding to its flower category.
3. **Image Transformation:** Images are resized to a standard 224x224 pixel size.
4. **Data Augmentation:** TensorFlow's ImageDataGenerator is applied for data augmentation, including rotation, width shift, height shift, shear, zoom, and horizontal flip.

### Model Architecture

The core of the project lies in the CNN architecture defined in `main.py`. The model architecture is structured as follows:

1. **Input Layer:** Accepts input images with dimensions 224x224x3 (RGB).
2. **Convolutional Layers:** Multiple convolutional layers with varying filter sizes and activation functions to extract hierarchical features.
3. **Max-Pooling Layers:** Applied to reduce spatial dimensions and retain important features.
4. **Flatten Layer:** Converts the 3D output to a 1D vector for input to fully connected layers.
5. **Fully Connected Layers:** Dense layers with ReLU activation.
6. **Output Layer:** Dense layer with softmax activation for multi-class classification.

### Training

The model is trained using the Adam optimizer with a learning rate of 1e-4 and sparse categorical crossentropy loss. Key training procedures include:

1. **Data Splitting:** The dataset is split into training and testing sets using `train_test_split` from scikit-learn.
2. **Feature Scaling:** Image pixel values are scaled to the range [0, 1].
3. **Learning Rate Schedule:** Adaptive learning rate scheduling using `ReduceLROnPlateau` callback.
4. **Early Stopping:** Implemented to halt training if validation loss plateaus.

### Evaluation

The trained model is evaluated on a separate test set, and performance metrics are computed, including accuracy and loss. Additionally, a set of test images is visualized with both actual and predicted labels for qualitative assessment.

### Results Visualization

Matplotlib is employed for visualizing the training history. Two primary plots are generated:

1. **Accuracy Plot:** Displays training and validation accuracy over epochs.
2. **Loss Plot:** Illustrates training and validation loss over epochs.

## Technology Stack

The following technologies constitute the technology stack for this project:

- **TensorFlow:** An open-source machine learning framework for building and training deep learning models.
- **Keras:** A high-level neural networks API running on top of TensorFlow, simplifying model development.
- **OpenCV:** A computer vision library for image processing tasks, used for loading and manipulating images.
- **NumPy:** A fundamental package for scientific computing with Python, employed for array operations.
- **Matplotlib:** A comprehensive library for creating static, animated, and interactive visualizations in Python, used for plotting training history and test images.
- **scikit-learn:** A machine learning library for data splitting and other utility functions.

## Usage

1. **Clone the Repository:**
   
   ```bash
   git clone https://github.com/your-username/tensorflow-flower-classification.git
3. Install Dependencies:
   
   ```bash
   pip install -r requirements.txt
   
5. Set Up Your Flower Dataset:
   
  Ensure you have a folder named dataset in the main directory.
  Organize your flower images into subdirectories for each category within the dataset/flowers folder. For example:
  
      dataset
      ├── flowers
      │   ├── daisy
      │   ├── dandelion
      │   ├── roses
      │   ├── sunflowers
      │   └── tulips

  Update the categories list in the main.py file with the names of your flower categories.

4. Run the Jupyter Notebook:
  Ensure you have Jupyter Notebook installed:

    ```bash
    pip install jupyter
    
  Now, start the Jupyter Notebook server:
     ```
      jupyter notebook
    ```

  Navigate to the tensorflow-flower-classification.ipynb notebook in the main directory and run the cells sequentially. This notebook provides an interactive environment for exploring the      code, visualizing results, and experimenting with the flower classification model.

5. Run the Main Script:
  Alternatively, you can run the main script directly:
     ```
      python mtf_Flowers Classification.py
    ```
  This will initiate the training process, save the trained model as 'mymodel.h5', and display evaluation results

## Future Enhancements
  This project serves as a foundation for flower classification, and potential future enhancements include:

  - Fine-tuning hyperparameters for improved model performance.
  - corporating additional data sources or transfer learning techniques.
  - Deployment of the model for real-time flower classification applications.


## Contributing
  Contributions to this project are encouraged! If you encounter issues, have suggestions, or want to contribute enhancements, please open an issue or submit a pull request.

## License
  This project is licensed under the [MIT License](LICENSE).


