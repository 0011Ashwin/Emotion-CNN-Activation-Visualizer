import streamlit as st
import os
import tempfile
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
from visualize_activations import ActivationMapVisualizer

# Main streamlit ui web-app
st.set_page_config(page_title="Emotion CNN Activation Visualizer", layout="wide")

def get_layer_names(visualizer: ActivationMapVisualizer) -> List[str]:
    # Dummy forward pass to get layer names
    dummy_img = np.zeros((visualizer.img_size[0], visualizer.img_size[1], 3), dtype=np.uint8)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    try:
        Image.fromarray(dummy_img).save(tmp_file.name)
        tmp_file.close()  # Ensure file is closed before deletion
        img_tensor, _ = visualizer.preprocess_image(tmp_file.name)
        activations, _ = visualizer.get_activation_maps(img_tensor)
    finally:
        os.unlink(tmp_file.name)
    return [name for name in activations.keys() if 'conv' in name or 'downsample.0' in name]

def main() -> None:
    st.title("Emotion CNN Activation Maps Visualizer")
    st.sidebar.header("Model Settings")
    model_path = st.sidebar.text_input(
        "Path to model weights (leave empty to use ImageNet weights)",
        value="model/emotion_model.pth"
    )
    st.sidebar.header("Visualization Settings")
    num_filters = st.sidebar.slider("Number of filters to visualize", 4, 32, 16, 4)
    demo_mode = st.sidebar.button("Demo Mode: Use Sample Image")
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "jpeg", "png"])
    sample_img_path = os.path.join("train", "happy", "Training_99988263.jpg")
    use_demo = demo_mode or (not uploaded_file and os.path.exists(sample_img_path))
    if uploaded_file or use_demo:
        if use_demo:
            image = Image.open(sample_img_path)
            img_path = sample_img_path
        else:
            image = Image.open(uploaded_file)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                img_path = tmp_file.name
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="Demo Image" if use_demo else "Uploaded Image", use_column_width=True)
        visualizer = ActivationMapVisualizer(model_path=model_path if os.path.exists(model_path) else None)
        layer_names = get_layer_names(visualizer)
        selected_layers = st.sidebar.multiselect(
            "Select layers to visualize", layer_names, default=layer_names
        )
        with col2:
            st.subheader("Class Activation Map")
            try:
                with st.spinner('Generating class activation map...'):
                    predicted_class, overlay = visualizer.visualize_class_activation_map(img_path)
                    st.image(overlay, caption=f"Predicted: {visualizer.emotion_labels[predicted_class]}", use_column_width=True)
                    st.download_button(
                        label="Download Overlay Image",
                        data=Image.fromarray(overlay).tobytes(),
                        file_name="class_activation_overlay.png",
                        mime="image/png"
                    )
            except Exception as e:
                st.error(f"Error generating class activation map: {e}")
        st.header("Layer Activation Maps")
        try:
            with st.spinner('Generating activation maps for each layer...'):
                img_tensor, original_img = visualizer.preprocess_image(img_path)
                activations, predicted_class = visualizer.get_activation_maps(img_tensor)
                img_np = np.array(original_img)
                for layer_name, activation in activations.items():
                    if layer_name not in selected_layers:
                        continue
                    if 'conv' not in layer_name and 'downsample.0' not in layer_name:
                        continue
                    act = activation.squeeze(0)
                    num_filters_to_show = min(num_filters, act.size(0))
                    with st.expander(f"Layer: {layer_name}"):
                        cols = st.columns(4)
                        for i in range(num_filters_to_show):
                            col_idx = i % 4
                            act_map = act[i].cpu().numpy()
                            act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min() + 1e-8)
                            import cv2
                            act_map_resized = cv2.resize(act_map, (original_img.width, original_img.height))
                            heatmap = cv2.applyColorMap(np.uint8(255 * act_map_resized), cv2.COLORMAP_JET)
                            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                            overlay = heatmap * 0.5 + img_np * 0.5
                            overlay = overlay.astype(np.uint8)
                            with cols[col_idx]:
                                st.image(overlay, caption=f"Filter {i}", use_column_width=True)
        except Exception as e:
            st.error(f"Error generating activation maps: {e}")
        if not use_demo and uploaded_file:
            os.unlink(img_path)
    else:
        st.info("Please upload an image or use Demo Mode to visualize activation maps")

if __name__ == "__main__":
    main() 