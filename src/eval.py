import torch
from sklearn.metrics import classification_report, confusion_matrix
from dataset import get_data_loaders
from model import get_pretrained_model

def evaluate_model(model_path='../models/best_mobilenet.pt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    # 1. Load test data loader
    _, _, test_loader = get_data_loaders(batch_size=64)

    # 2. Re-instantiate model architecture and load saved weights
    model = get_pretrained_model(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device if device.type=='cpu' else None))
    model.eval()

    all_preds = []
    all_labels = []

    print("Running final model evaluation over test dataset...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 3. Compute Professional Machine Learning Metrics
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("\n================ DETAILED CLASSIFICATION REPORT ================")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    print("======================= CONFUSION MATRIX =======================")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == '__main__':
    evaluate_model()