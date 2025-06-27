import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any

# Model training script for emotion recognition
# Import the model architecture from our visualization script
from visualize_activations import EmotionCNN

class EmotionDataset(Dataset):
    """Dataset for emotion recognition from face images"""
    
    def __init__(self, data_dir: str, transform: Optional[Any] = None) -> None:
        """
        Args:
            data_dir (string): Directory with emotion folders
            transform (callable, optional): Transform to be applied on images
        """
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        
        self.image_paths = []
        self.labels = []
        
        # Collect all image paths and their corresponding labels
        for i, emotion in enumerate(self.classes):
            emotion_dir = os.path.join(data_dir, emotion)
            for img_name in os.listdir(emotion_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(emotion_dir, img_name))
                    self.labels.append(i)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image and apply transform
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(f"Could not open image {img_path}: {e}")
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

def train_model(
    train_dir: str,
    val_dir: Optional[str] = None,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    save_dir: str = 'model',
    img_size: Tuple[int, int] = (224, 224),
    device: Optional[str] = None
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Train an emotion recognition model
    
    Args:
        train_dir: Directory containing training images by emotion class
        val_dir: Directory containing validation images (if None, uses train_dir)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        save_dir: Directory to save model weights
        img_size: Size to resize input images
        device: Device to use for training (auto-detected if None)
    
    Returns:
        model: Trained model
        history: Training history
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories for model saving
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = EmotionDataset(train_dir, transform=train_transform)
    
    if val_dir is None:
        val_dir = train_dir
    val_dataset = EmotionDataset(val_dir, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = EmotionCNN(num_classes=len(train_dataset.classes)).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        
        # Update learning rate and manually print status
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"Learning rate changed from {old_lr} to {new_lr}")
        
        # Print statistics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'emotion_model_best.pth'))
            print("Saved best model")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'history': history
        }, os.path.join(save_dir, 'emotion_model_checkpoint.pth'))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'emotion_model.pth'))
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()
    
    return model, history

def main() -> None:
    """Example usage for training the model."""
    # Example usage
    train_dir = 'train'  # Changed from 'Face Data/train'
    val_dir = 'test'     # Changed from 'Face Data/test'
    
    # Train model
    model, history = train_model(
        train_dir=train_dir,
        val_dir=val_dir,
        epochs=20,
        batch_size=16, # Reduced batch size for memory efficiency 32 -> 16
        learning_rate=0.001,
        save_dir='model'
    )
    
    print("Training complete!")

if __name__ == '__main__':
    main() 