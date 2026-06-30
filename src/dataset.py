# import torch
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import random_split, DataLoader

# def get_data_loaders(data_dir='../data', batch_size=64):
#     """
#     Prepares and returns train, validation, and test DataLoaders.
#     Resizes CIFAR-10 images to 224x224 to match pretrained backbone requirements.
#     """
#     # Pretrained models require specific ImageNet channel normalization stats
#     transform_pipeline = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     # Load local datasets (download=False because you already extracted it)
#     train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform_pipeline)
#     test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform_pipeline)

#     # Deterministic validation split (10% of training data)
#     train_size = int(0.9 * len(train_set))
#     val_size = len(train_set) - train_size
#     train_subset, val_subset = random_split(
#         train_set, [train_size, val_size], 
#         generator=torch.Generator().manual_seed(42)
#     )

#     # Build efficient DataLoaders
#     train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
#     val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
#     test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

#     return train_loader, val_loader, test_loader

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader

def get_data_loaders(data_dir=None, batch_size=64):
    """
    Prepares and returns train, validation, and test DataLoaders.
    Automatically resolves absolute paths to prevent terminal directory errors.
    """
    # If no directory is provided, find the 'data' folder relative to this file
    if data_dir is None:
        current_script_dir = os.path.dirname(os.path.abspath(__file__)) # src/
        project_root = os.path.dirname(current_script_dir)             # vision-classifier-project/
        data_dir = os.path.join(project_root, 'data')                  # vision-classifier-project/data
    
    print(f" -> Accessing dataset at: {data_dir}")

    # Pretrained models require specific ImageNet channel normalization stats
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load local datasets
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform_pipeline)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform_pipeline)

    # Deterministic validation split (10% of training data)
    train_size = int(0.9 * len(train_set))
    val_size = len(train_set) - train_size
    train_subset, val_subset = random_split(
        train_set, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    # Build efficient DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader