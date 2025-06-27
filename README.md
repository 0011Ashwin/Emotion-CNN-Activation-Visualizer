# Emotion CNN Activation Maps Visualizer

This application visualizes the activation maps of a CNN model for emotion detection, allowing you to see which parts of an image are important for the classification decision.

## 🚀 What's New (2025 Upgrade)

- **Code Refactoring & Optimization:** Improved readability, modularity, and error handling across all main scripts. Type hints and docstrings throughout.
- **UI/UX Enhancements:** Modern Streamlit interface with sidebar controls, layer selection, CAM/Grad-CAM toggle, and download options.
- **Grad-CAM Support:** Advanced, interpretable visualizations for model explainability.
- **Demo Mode:** Instantly try the app with a sample image—no upload required.
- **Type Hints & Docstrings:** Improved maintainability and developer experience.

## Features

- Upload your own face images
- Visualize Class Activation Maps (CAM) and Grad-CAM
- Explore activation maps for different CNN layers
- Adjust visualization settings (number of filters, layers)
- Use custom trained models
- Download overlay images
- Demo mode for instant testing

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

### Using the Application

1. The app will open in your browser (typically at http://localhost:8501)
2. **Model Settings:** Change the path to your model in the sidebar (or leave empty to use ImageNet weights)
3. **Visualization Settings:**
   - Select the number of filters to visualize
   - Choose which layers to display
   - Toggle between CAM and Grad-CAM
   - Use Demo Mode to instantly load a sample image
4. **Upload Image:** Upload a face image using the uploader, or use Demo Mode
5. **Explore Results:**
   - View the class activation map (CAM or Grad-CAM)
   - Download the overlay image
   - Expand layer sections to see filter-wise activation overlays

## Model

The default model is a ResNet18 architecture trained for emotion classification with 7 classes:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

You can use your own trained model by specifying the path in the sidebar.

## Requirements

- Python 3.7+
- PyTorch
- Streamlit
- OpenCV
- Other dependencies listed in requirements.txt

## Advanced Features

- **Grad-CAM:** Select "Grad-CAM" in the sidebar to generate more interpretable heatmaps.
- **Demo Mode:** Click the "Demo Mode" button in the sidebar to instantly try the app with a sample image from the dataset.
- **Download Overlays:** Download the generated overlay images for further analysis or sharing.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements or new features.

## License

MIT License 