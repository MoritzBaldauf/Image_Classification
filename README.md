# Image Classification Project

This project implements a Convolutional Neural Network (CNN) for image classification using PyTorch. It includes functionality for dataset preparation, model training, and evaluation.

## Project Structure

- `main.py`: The entry point of the project, orchestrating the entire workflow.
- `architecture.py`: Defines the CNN model architecture.
- `dataset.py`: Contains the custom dataset class for loading and preprocessing images.
- `train.py`: Implements the training loop and model evaluation.
- `utils.py`: Provides utility functions for dataset splitting and model evaluation.

## Features

- Custom CNN architecture for image classification
- Dataset splitting into training and validation sets
- Image augmentation for improved model performance
- Training with early stopping and learning rate scheduling
- Model evaluation and performance metrics

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- Pillow
- pandas
- scikit-learn
- tqdm

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/MoritzBaldauf/image-classification-project.git
   cd image-classification-project
   ```

2. Install the required packages:
   ```
   pip install torch torchvision numpy Pillow pandas scikit-learn tqdm
   ```

## Usage

1. Prepare your dataset:
   - Place your images in the `Images` folder.
   - Ensure you have a CSV file with image names and labels in the `Images` folder or its parent directory.

2. Run the main script:
   ```
   python main.py
   ```

   This will:
   - Split the dataset into training and validation sets
   - Train the model
   - Evaluate the model's performance

3. The trained model will be saved as `model.pth` in the project directory.

## Customization

- Adjust hyperparameters in `main.py` (e.g., learning rate, number of epochs).
- Modify the model architecture in `architecture.py`.
- Customize data augmentation techniques in `main.py`.

## Results

After training, the script will output:
- Training and validation loss/accuracy for each epoch
- Final validation loss and accuracy
- 
