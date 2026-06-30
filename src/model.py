import torch
import torch.nn as nn
import torchvision.models as models

def get_pretrained_model(num_classes=10):
    """
    Loads a pretrained MobileNetV2 architecture and modifies the 
    final classification head to match the target number of classes.
    """
    # Download MobileNetV2 with modern default weights
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    
    # Locate the internal classifier block features
    # MobileNetV2 classifier structure: Dropout(p=0.2), Linear(in_features=1280, out_features=1000)
    in_features = model.classifier[1].in_features
    
    # Replace the final linear layer with our 10-class target output
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model