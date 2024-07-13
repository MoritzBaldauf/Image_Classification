import torch
from tqdm import tqdm
import os
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating", leave=False)
        for inputs, labels, _, _ in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({'loss': f"{total_loss / total:.4f}", 'acc': f"{100. * correct / total:.2f}%"})

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total

    return avg_loss, accuracy



def split_dataset(source_dir, train_dir, val_dir, split_ratio=0.2):
    # Read the labels.csv file
    labels_df = pd.read_csv(os.path.join(source_dir, 'labels.csv'), delimiter=';')

    # Split the data, stratifying by the class labels
    train_df, val_df = train_test_split(labels_df, test_size=split_ratio, stratify=labels_df['label'], random_state=42)

    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Copy files to respective directories
    for _, row in train_df.iterrows():
        shutil.copy(os.path.join(source_dir, row['name']), os.path.join(train_dir, row['name']))

    for _, row in val_df.iterrows():
        shutil.copy(os.path.join(source_dir, row['name']), os.path.join(val_dir, row['name']))

    # Copy labels.csv to both train and val directories
    train_df.to_csv(os.path.join(train_dir, 'labels.csv'), index=False, sep=';')
    val_df.to_csv(os.path.join(val_dir, 'labels.csv'), index=False, sep=';')

    print(f"Total images: {len(labels_df)}")
    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")


from collections import Counter


def check_class_distribution(dataset):
    labels = [item[1] for item in dataset]  # Assuming the label is the second item returned by dataset.__getitem__
    class_distribution = Counter(labels)

    print("Class Distribution:")
    for class_id, count in class_distribution.items():
        print(f"Class {class_id}: {count} ({count / len(labels) * 100:.2f}%)")

    return class_distribution