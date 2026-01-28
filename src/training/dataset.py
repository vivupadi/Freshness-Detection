# dataset.py
"""
Dataset for fruit freshness classification
"""

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from sklearn.model_selection import train_test_split


class FruitDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        self.fruit_classes = ['apples', 'banana', 'bittergroud', 'capsicum', 'cucumber', 
                          'okra', 'oranges', 'potato', 'tomato']
        
        # Load images
        for folder_name in os.listdir(root_dir):
            folder_lower = folder_name.lower()
            freshness = None
            fruit_type = None
            
            # Parse folder name: freshapples, rottenapples, etc.
            if folder_lower.startswith('fresh'):
                freshness = 'fresh'
                fruit_type = folder_lower[5:]  # Remove 'fresh' prefix
            elif folder_lower.startswith('rotten'):
                freshness = 'rotten'
                fruit_type = folder_lower[6:]  # Remove 'rotten' prefix
            else:
                continue
            
            if fruit_type not in self.fruit_classes:
                continue
            if freshness not in ['fresh', 'rotten']:
                continue
            
            fruit_label = self.fruit_classes.index(fruit_type)
            freshness_label = 0 if freshness == 'fresh' else 1
            
            folder_path = os.path.join(root_dir, folder_name)
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(folder_path, img_name)
                    self.samples.append((img_path, fruit_label, freshness_label))
        
        print(f"Loaded {len(self.samples)} images from {root_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, fruit_label, freshness_label = self.samples[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, fruit_label, freshness_label


def create_dataloaders(data_dir, batch_size=32, num_workers=4, test_split=0.1, val_split=0.1):
    """
    Create train and validation dataloaders
    
    Args:
        data_dir: Path to dataset root directory
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        test_split: Proportion of data for test set
        val_split: Proportion of training data for validation
    
    Returns:
        train_loader, val_loader
    """
    # Define transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load full dataset
    full_dataset = FruitDataset(data_dir, transform=train_transforms)
    
    # Split indices
    indices = list(range(len(full_dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=test_split, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_split, random_state=42)
    
    # Create subsets
    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    
    # Apply validation transforms to validation set
    val_dataset_obj = FruitDataset(data_dir, transform=val_transforms)
    val_dataset = Subset(val_dataset_obj, val_idx)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader