from architecture import MyCNN
from dataset import ImagesDataset
from train import train_model
from utils import evaluate_model
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import split_dataset, evaluate_model, check_class_distribution
import os


# Main function structure is inspired by the main functions in a5_ex1.py and a5_ex2.py
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Set random seed for reproducibility (inspired by a5_ex1.py)
    torch.manual_seed(1234)

    # Define directories
    source_dir = "Images"
    train_dir = "Images/training"
    val_dir = "Images/valuation"

    # Split the dataset
    split_dataset(source_dir, train_dir, val_dir, split_ratio=0.2)

    # Data augmentation is inspired by a6_ex1.py
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
    ])

    # Dataset loading is similar to a3_ex1.py
    train_dataset = ImagesDataset(train_dir, width=100, height=100)
    val_dataset = ImagesDataset(val_dir, width=100, height=100)

    # DataLoader creation is similar to a3_ex2.py
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Check class distribution
    """
    print("Training set class distribution:")
    train_distribution = check_class_distribution(train_dataset)
    print("\nValidation set class distribution:")
    val_distribution = check_class_distribution(val_dataset)"""

    model = MyCNN()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    # Training process is inspired by a5_ex1.py and a5_ex2.py
    train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, device=device)

    # Model loading and final evaluation is inspired by a5_ex2.py
    model.load_state_dict(torch.load("model.pth"))

    criterion = torch.nn.CrossEntropyLoss()
    val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
    print(f"Final Validation Loss: {val_loss:.4f}, Final Validation Accuracy: {val_accuracy:.4f}")


if __name__ == "__main__":
    main()