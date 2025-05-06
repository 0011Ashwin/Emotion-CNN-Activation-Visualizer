import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import OrderedDict

class EmotionCNN(nn.Module):
    """
    CNN model for emotion detection. 
    This is a sample model architecture that you can replace with your own pre-trained model.
    """
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        # Use a pre-trained ResNet as the base
        self.base_model = resnet18(pretrained=True)
        
        # Replace the final fully connected layer
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)
        
        # Store activations for visualization
        self.activations = OrderedDict()
        
        # Register hooks to capture activations
        self.hooks = []
        
        # Register hooks for the layers we want to visualize
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Conv2d) and 'layer' in name:
                self.hooks.append(
                    module.register_forward_hook(self._get_activation(name))
                )
    
    def _get_activation(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def forward(self, x):
        return self.base_model(x)
    
    def get_activations(self):
        return self.activations

class ActivationMapVisualizer:
    """Class to handle the visualization of CNN activation maps"""
    
    def __init__(self, model_path=None, img_size=(224, 224), device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.img_size = img_size
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Initialize model
        self.model = EmotionCNN(num_classes=len(self.emotion_labels)).to(device)
        
        # Load pre-trained weights if provided
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model loaded from {model_path}")
        else:
            print("No model loaded. Using the model with pre-trained ImageNet weights.")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Define image transformation
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, img_path):
        """Preprocess an image for the model"""
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        return img_tensor, img
    
    def get_activation_maps(self, img_tensor):
        """Get activation maps from the model for a given image tensor"""
        with torch.no_grad():
            pred = self.model(img_tensor)
            predicted_class = torch.argmax(pred, dim=1).item()
            
        return self.model.get_activations(), predicted_class
    
    def visualize_activation_maps(self, img_path, output_dir=None, num_filters=16, figsize=(15, 10)):
        """
        Visualize activation maps for a given image
        
        Args:
            img_path: Path to the input image
            output_dir: Directory to save visualization images (if None, just display)
            num_filters: Number of filters to visualize for each layer
            figsize: Size of the figure for matplotlib
        """
        # Preprocess image
        img_tensor, original_img = self.preprocess_image(img_path)
        
        # Get activation maps
        activations, predicted_class = self.get_activation_maps(img_tensor)
        
        # Original image as numpy array for overlaying
        img_np = np.array(original_img)
        
        # Create a directory for output if specified
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Plot and save activation maps
        for layer_name, activation in activations.items():
            # Only process convolutional layers
            if 'conv' not in layer_name and 'downsample.0' not in layer_name:
                continue
            
            # Get activations for the current layer
            act = activation.squeeze(0)
            
            # Limit the number of filters to visualize
            num_filters_to_show = min(num_filters, act.size(0))
            
            # Create figure
            fig, axs = plt.subplots(2, num_filters_to_show, figsize=figsize)
            fig.suptitle(f'Layer: {layer_name} | Predicted: {self.emotion_labels[predicted_class]}', fontsize=16)
            
            for i in range(num_filters_to_show):
                # Get activation map for the current filter
                act_map = act[i].cpu().numpy()
                
                # Normalize activation map to [0, 1]
                act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min() + 1e-8)
                
                # Resize activation map to match original image size
                act_map_resized = cv2.resize(act_map, (original_img.width, original_img.height))
                
                # Display activation map
                axs[0, i].imshow(act_map, cmap='jet')
                axs[0, i].set_title(f'Filter {i}')
                axs[0, i].axis('off')
                
                # Create heatmap overlay on original image
                heatmap = cv2.applyColorMap(np.uint8(255 * act_map_resized), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                # Overlay heatmap on original image
                overlay = heatmap * 0.5 + img_np * 0.5
                overlay = overlay.astype(np.uint8)
                
                # Display overlay
                axs[1, i].imshow(overlay)
                axs[1, i].set_title(f'Overlay {i}')
                axs[1, i].axis('off')
            
            plt.tight_layout()
            
            # Save or display figure
            if output_dir:
                layer_name_safe = layer_name.replace('.', '_')
                plt.savefig(os.path.join(output_dir, f'activation_map_{layer_name_safe}.png'))
            else:
                plt.show()
                
            plt.close(fig)
        
        return predicted_class
    
    def visualize_class_activation_map(self, img_path, output_dir=None):
        """
        Generate a Class Activation Map (CAM) to show which regions 
        of the image are most important for the classification
        """
        # This is a simplified CAM implementation
        # For a full Grad-CAM implementation, more hooks and gradient tracking would be needed
        
        # Preprocess image
        img_tensor, original_img = self.preprocess_image(img_path)
        
        # Get activation maps
        activations, predicted_class = self.get_activation_maps(img_tensor)
        
        # Get the last convolutional layer activation
        last_conv_layer = None
        for name in reversed(list(activations.keys())):
            if 'conv' in name or 'downsample.0' in name:
                last_conv_layer = name
                break
        
        if last_conv_layer is None:
            print("Could not find a convolutional layer")
            return
        
        last_conv_act = activations[last_conv_layer].squeeze(0)
        
        # Average the activations across channels to create a rough CAM
        # This is a simplified approach; Grad-CAM would use gradients for weighting
        cam = torch.mean(last_conv_act, dim=0).cpu().numpy()
        
        # Normalize CAM
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Resize CAM to match original image size
        cam_resized = cv2.resize(cam, (original_img.width, original_img.height))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on original image
        img_np = np.array(original_img)
        overlay = heatmap * 0.5 + img_np * 0.5
        overlay = overlay.astype(np.uint8)
        
        # Create figure
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Display original image, CAM, and overlay
        axs[0].imshow(original_img)
        axs[0].set_title('Original Image')
        axs[0].axis('off')
        
        axs[1].imshow(cam, cmap='jet')
        axs[1].set_title('Class Activation Map')
        axs[1].axis('off')
        
        axs[2].imshow(overlay)
        axs[2].set_title(f'Overlay - Predicted: {self.emotion_labels[predicted_class]}')
        axs[2].axis('off')
        
        plt.tight_layout()
        
        # Save or display figure
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(os.path.join(output_dir, 'class_activation_map.png'))
        else:
            plt.show()
            
        plt.close(fig)
        
        return predicted_class, overlay

def main():
    # Example usage
    model_path = 'model/emotion_model.pth'  # Replace with your model path
    
    # Use a sample image from train directory
    try:
        # Look for a sample in the happy folder
        emotion_dir = 'train/happy'
        if os.path.exists(emotion_dir):
            img_files = [f for f in os.listdir(emotion_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if img_files:
                img_path = os.path.join(emotion_dir, img_files[0])
            else:
                # Try to find any image in any emotion folder
                for emotion in ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']:
                    emotion_dir = os.path.join('train', emotion)
                    if os.path.exists(emotion_dir):
                        img_files = [f for f in os.listdir(emotion_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                        if img_files:
                            img_path = os.path.join(emotion_dir, img_files[0])
                            break
        else:
            img_path = 'data/sample_face.jpg'  # Fallback to default path
    except:
        img_path = 'data/sample_face.jpg'  # Fallback to default path
    
    output_dir = 'outputs'
    
    visualizer = ActivationMapVisualizer(model_path=model_path)
    
    # Visualize activation maps
    visualizer.visualize_activation_maps(img_path, output_dir)
    
    # Visualize class activation map
    visualizer.visualize_class_activation_map(img_path, output_dir)

if __name__ == '__main__':
    main() 