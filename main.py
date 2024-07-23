import torch
from architecture import MyCNN
from dataset import ImagesDataset
from train import train_model
from utils import evaluate_model
from utils import split_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

# Main function structure is inspired by the main functions in a5_ex1.py and a5_ex2.py
def main():
    # Find device for model training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU available
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Set random seed for reproducibility
    torch.manual_seed(1234)

    # Split Image dataset into training and valuation and save them in separate folders
    # Define directories
    source_dir = "Images"
    train_dir = "Images/training"
    val_dir = "Images/valuation"

    # Split the dataset (see utils.py for function documentation)
    split_dataset(source_dir, train_dir, val_dir, split_ratio=0.2)

    # Performing simple Image augmentation to improve model performance (similar to a6_ex1.py)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(100, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.Normalize((0.5,), (0.5,))
    ])

    val_transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Loading Images
    train_dataset = ImagesDataset(train_dir, width=100, height=100, transform=train_transform)
    val_dataset = ImagesDataset(val_dir, width=100, height=100, transform=val_transform)

    # DataLoader creation is similar to a3_ex2.py
    # Batch size, and number of workers is set as in best practice
    # Shuffle = true for the training data, to mitigate biases and improve generalization
    # Shuffle = false for evaluation, because we want to improve consistency
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    model = MyCNN()  # Create an instance of the model defined in architecture.py
    model.to(device)  # Move model to the selected device (CPU or GPU)

    # Model training (similar to a5_ex1.py and a5_ex2.py)
    train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, device=device)

    # Model loading and final evaluation is inspired by a5_ex2.py
    model.load_state_dict(torch.load("model.pth", map_location=device))

    criterion = torch.nn.CrossEntropyLoss()
    val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
    print(f"Final Validation Loss: {val_loss:.4f}, Final Validation Accuracy: {val_accuracy:.4f}")

if __name__ == "__main__":
    main()