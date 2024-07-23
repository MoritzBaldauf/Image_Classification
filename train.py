import torch
from utils import evaluate_model
from tqdm import tqdm
import time


# function is addapted from a5_ex1.py and a5_ex2.py
def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    criterion = torch.nn.CrossEntropyLoss() # Lossfunction
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) # Optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5) #Scheduler, reduces learning rate if model is not getting better

    best_accuracy = 0.0 # Track best model accuracy
    patience = 10 # How many iterations are performed without improvement bevor early stopping
    early_stop_cnt = 0 # Track iterations without improvement

    print(f"\nNumber of epochs: {num_epochs}")
    print(f"Initial learning rate: {learning_rate}")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        start_time = time.time() # Measure time of one epoch

        # Training loop with progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", leave=False)
        for inputs, labels, _, _ in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device) # move imput tensor to CPU or GPU

            # Training step
            optimizer.zero_grad() # reset gradients
            outputs = model(inputs) # call model forward method
            loss = criterion(outputs, labels) # calc loss
            loss.backward() # backpropagation
            optimizer.step() # updated model parameters

            train_loss += loss.item() # calculate loss of current batch
                                        # loss.item() gives loss tensors value
            _, predicted = outputs.max(1) # gets predicted labels
            train_total += labels.size(0) # counts samples processed
            # compare predicted labels with actual
            train_correct += predicted.eq(labels).sum().item() # sum -> count of correct predictions; item -> gets correct count as number

            # Update progress bar
            train_pbar.set_postfix(
                {'loss': f"{train_loss / train_total:.4f}", 'acc': f"{100. * train_correct / train_total:.2f}%"})

        # Validation loop
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)

        scheduler.step(val_loss) # updated scheduler with the calcualted validation loss

        epoch_time = time.time() - start_time # epoch runtime

        # Information on epoch printed
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss / train_total:.4f}, Train Accuracy: {100. * train_correct / train_total:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {100. * val_accuracy:.2f}%")
        print(f"Epoch Time: {epoch_time:.2f} seconds")

        # If the newly trained model is better than the saved one we save it instead
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), "model.pth")
            print(f"\nNew best model saved with validation accuracy: {100. * best_accuracy:.2f}%")
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        # Stop training if earlystopping limit is reached
        if early_stop_cnt >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

        print("-" * 60) # Visual seperator for better readability in the terminal

    print("Training finished!")
    print(f"Best validation accuracy: {100. * best_accuracy:.2f}%")
