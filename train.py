import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from architecture import MyCNN
from dataset import ImagesDataset
from utils import evaluate_model
from tqdm import tqdm
import time


# train_model function is inspired by the training loop in a5_ex1.py and a5_ex2.py
def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val_accuracy = 0.0
    patience = 10
    early_stop_counter = 0

    print(f"Training on device: {device}")
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(val_loader.dataset)}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Initial learning rate: {learning_rate}")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        start_time = time.time()

        # Training loop with progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", leave=False)
        for inputs, labels, _, _ in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            # Update progress bar
            train_pbar.set_postfix(
                {'loss': f"{train_loss / train_total:.4f}", 'acc': f"{100. * train_correct / train_total:.2f}%"})

        # Validation loop is similar to the one in a5_ex2.py
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        epoch_time = time.time() - start_time

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss / train_total:.4f}, Train Accuracy: {100. * train_correct / train_total:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {100. * val_accuracy:.2f}%")
        print(f"Epoch Time: {epoch_time:.2f} seconds")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "model.pth")
            print(f"\nNew best model saved with validation accuracy: {100. * best_val_accuracy:.2f}%")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

        print("-" * 60)

    print("Training finished!")
    print(f"Best validation accuracy: {100. * best_val_accuracy:.2f}%")


# Main function structure is inspired by the main functions in a5_ex1.py and a5_ex2.py
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data augmentation is inspired by a6_ex1.py
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

    train_dataset = ImagesDataset("path/to/train/images", width=100, height=100, transform=train_transform)
    val_dataset = ImagesDataset("path/to/val/images", width=100, height=100, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = MyCNN().to(device)

    train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, device=device)


if __name__ == "__main__":
    main()