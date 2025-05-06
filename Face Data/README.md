# Emotion CNN Activation Maps Visualizer

This application visualizes the activation maps of a CNN model for emotion detection, allowing you to see which parts of an image are important for the classification decision.

## Features

- Upload your own face images
- Visualize Class Activation Maps (CAM)
- Explore activation maps for different CNN layers
- Adjust visualization settings
- Use custom trained models

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
2. You can change the path to your model in the sidebar (or leave empty to use ImageNet weights)
3. Upload a face image using the uploader
4. Adjust the number of filters to visualize using the slider in the sidebar
5. Explore the activation maps for different layers

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