import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_data_loaders
from model import get_pretrained_model

def train_model(epochs=5, batch_size=64, lr=0.001, save_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # FIX: Dynamically calculate absolute path to project root
    current_script_dir = os.path.dirname(os.path.abspath(__file__)) # src/
    project_root = os.path.dirname(current_script_dir)             # vision-classifier-project/
    
    if save_path is None:
        save_path = os.path.join(project_root, 'models', 'best_mobilenet.pt')
    
    print(f" -> Model checkpoints will save strictly to: {save_path}")

    # 1. Load Data
    print("Loading data partitions...")
    train_loader, val_loader, _ = get_data_loaders(batch_size=batch_size)

    # 2. Initialize Model
    print("Initializing MobileNetV2...")
    model = get_pretrained_model(num_classes=10).to(device)

    # 3. Define Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 4. Core Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        train_acc = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)

        # 5. Validation Phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f"--- Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% ---")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"==> New best model saved with Val Accuracy: {val_acc:.2f}%")

if __name__ == '__main__':
    train_model(epochs=3, batch_size=64, lr=0.0005)