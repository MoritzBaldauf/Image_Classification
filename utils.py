import torch
from tqdm import tqdm
import os
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd

def evaluate_model(model, data_loader, criterion, device):
    """
    :param model: Trained model
    :param data_loader: Dataloader for the Evaluation set
    :param criterion: Loss function to use for the model
    :param device: Device used for training (CPU/GPU)
    :return:
    """
    model.eval() # Set model into evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # Dont compute gradients

        # Loop through evaluation data loader
        pbar = tqdm(data_loader, desc="Evaluating", leave=False)
        for inputs, labels, _, _ in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs) # Test model with evaluation data
            loss = criterion(outputs, labels) # Loss between prediction and actual labels

            total_loss += loss.item() # Sum up loss

            _, predicted = outputs.max(1) #get prediciton for samples
            total += labels.size(0) # Track number of samples processed
            correct += predicted.eq(labels).sum().item() # Count correctly labeled samples

            # Update progress bar
            pbar.set_postfix({'loss': f"{total_loss / total:.4f}", 'acc': f"{100. * correct / total:.2f}%"})

    # Calc metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total

    return avg_loss, accuracy



def split_dataset(source_dir, train_dir, val_dir, split_ratio=0.2):
    """
    :param source_dir: Original Image directory, all images are stored there
    :param train_dir: Directory for Images used form model training
    :param val_dir: Directory for Images used to evaluate the model. These images are not used for the model training
    :param split_ratio: Ratio between training and valuation images. Standard ration 80/20
    :return:
    """
    # Read the labels.csv file
    labels_df = pd.read_csv(os.path.join(source_dir, 'labels.csv'), delimiter=';')

    # Split the data
    train_df, val_df = train_test_split(labels_df, test_size=split_ratio, stratify=labels_df['label'], random_state=42)

    # Set up folders if they dont exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Saving images into training folder
    for _, row in train_df.iterrows():
        shutil.copy(os.path.join(source_dir, row['name']), os.path.join(train_dir, row['name']))

    # Saving images into valuation folder
    for _, row in val_df.iterrows():
        shutil.copy(os.path.join(source_dir, row['name']), os.path.join(val_dir, row['name']))

    # Create a copy of labels.csv for the train and valuation folders
    train_df.to_csv(os.path.join(train_dir, 'labels.csv'), index=False, sep=';')
    val_df.to_csv(os.path.join(val_dir, 'labels.csv'), index=False, sep=';')

    print(f"Total images: {len(labels_df)}")
    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
