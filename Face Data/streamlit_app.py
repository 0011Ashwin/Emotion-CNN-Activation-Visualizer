import streamlit as st
import os
import tempfile
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from visualize_activations import ActivationMapVisualizer

# Main streamlit ui web-app
st.set_page_config(page_title="Emotion CNN Activation Visualizer", layout="wide")

def main():
    st.title("Emotion CNN Activation Maps Visualizer")
    
    # Sidebar for model path
    st.sidebar.header("Model Settings")
    model_path = st.sidebar.text_input(
        "Path to model weights (leave empty to use ImageNet weights)",
        value="model/emotion_model.pth"
    )
    
    # Sidebar for visualization settings
    st.sidebar.header("Visualization Settings")
    num_filters = st.sidebar.slider("Number of filters to visualize", 4, 32, 16, 4)
    
    # File uploader for image
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            img_path = tmp_file.name
        
        # Initialize the visualizer
        visualizer = ActivationMapVisualizer(model_path=model_path if os.path.exists(model_path) else None)
        
        # Generate class activation map
        with col2:
            st.subheader("Class Activation Map")
            with st.spinner('Generating class activation map...'):
                predicted_class, overlay = visualizer.visualize_class_activation_map(img_path)
                st.image(overlay, caption=f"Predicted: {visualizer.emotion_labels[predicted_class]}", use_column_width=True)
        
        # Generate activation maps for different layers
        st.header("Layer Activation Maps")
        with st.spinner('Generating activation maps for each layer...'):
            # Preprocess image
            img_tensor, original_img = visualizer.preprocess_image(img_path)
            
            # Get activation maps
            activations, predicted_class = visualizer.get_activation_maps(img_tensor)
            
            # Original image as numpy array for overlaying
            img_np = np.array(original_img)
            
            # Display activation maps for each layer
            for layer_name, activation in activations.items():
                # Only process convolutional layers
                if 'conv' not in layer_name and 'downsample.0' not in layer_name:
                    continue
                
                # Get activations for the current layer
                act = activation.squeeze(0)
                
                # Limit the number of filters to visualize
                num_filters_to_show = min(num_filters, act.size(0))
                
                # Create expander for each layer
                with st.expander(f"Layer: {layer_name}"):
                    # Create columns for filters
                    cols = st.columns(4)
                    
                    for i in range(num_filters_to_show):
                        col_idx = i % 4
                        
                        # Get activation map for the current filter
                        act_map = act[i].cpu().numpy()
                        
                        # Normalize activation map to [0, 1]
                        act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min() + 1e-8)
                        
                        # Resize activation map to match original image size
                        import cv2
                        act_map_resized = cv2.resize(act_map, (original_img.width, original_img.height))
                        
                        # Create heatmap overlay on original image
                        heatmap = cv2.applyColorMap(np.uint8(255 * act_map_resized), cv2.COLORMAP_JET)
                        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                        
                        # Overlay heatmap on original image
                        overlay = heatmap * 0.5 + img_np * 0.5
                        overlay = overlay.astype(np.uint8)
                        
                        # Display in the appropriate column
                        with cols[col_idx]:
                            st.image(overlay, caption=f"Filter {i}", use_column_width=True)
        
        # Clean up the temporary file
        os.unlink(img_path)
    
    else:
        st.info("Please upload an image to visualize activation maps")

if __name__ == "__main__":
    main() 