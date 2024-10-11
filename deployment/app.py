from fastai.vision.all import *
import gradio as gr
import requests
from PIL import Image
import base64
import io
from pathlib import Path, PosixPath
import torch
import pickle

# Label list
labels = (
    'Blueberry',
    'Cranberry',
    'Currants',
    'Elderberry',
    'Goji berry or Barberry',
    'Goldenberry',
    'Gooseberry',
    'Mulberry',
    'Raspberry',
    'Strawberry'
)

# Load the model
model = load_learner('berry-recogniser.pkl')

# Define the prediction function
def recognize_image(image_data):
    try:
        # Ensure image_data is a base64-encoded string
        if isinstance(image_data, str):
            image_data = image_data.split(",")[1]  # Remove the base64 header
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        elif isinstance(image_data, Image.Image):
            # If the image is already a PIL image, use it directly
            image = image_data
        else:
            raise ValueError("Invalid image data format. Expected base64-encoded string or PIL Image.")

        # Make predictions using the model
        pred, idx, probs = model.predict(image)

        # Return predictions in the desired format
        return {labels[i]: float(probs[i]) for i in range(len(labels))}
    
    except Exception as e:
        return {"error": str(e)}

# Set up Gradio interface
image = gr.Image(type="pil")
label = gr.Label(num_top_classes=10)

# Update example paths
examples = [
    "test_image_01.jpg",
    "test_image_02.jpg",
    "test_image_03.jpg",
    "test_image_04.jpg",
    "test_image_05.jpg"
]

# Create Gradio interface
iface = gr.Interface(fn=recognize_image, inputs=image, outputs=label, examples=examples)
iface.launch()






