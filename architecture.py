import torch
import torch.nn as nn

class MyCNN(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), # Normalizing outputs
            nn.ReLU(),# Activation function
            nn.MaxPool2d(kernel_size=2, stride=2), # Reduce dimensions

            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Classification
        self.classifier = nn.Sequential(
            nn.Linear(128 * 12 * 12, 512), # Flattern
            nn.ReLU(),
            nn.Dropout(0.5), # Prevent overfitting by deactivating 50% of neurons -> model generalizes better
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    # Forward method structure (addapted from a4_ex1.py)
    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        x = self.features(input_images)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = MyCNN()