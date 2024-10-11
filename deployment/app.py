from fastai.vision.all import *
import gradio as gr
import requests
from PIL import Image
import base64
import io

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
model = load_learner("berry-recogniser-v2.pkl")

# Define the prediction function
def recognize_image(image_data):
    try:
        # Ensure the image_data is a base64-encoded string
        if isinstance(image_data, str):
            # Extract the base64 string from the data URL format
            image_data = image_data.split(",")[1]
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        elif isinstance(image_data, Image.Image):
            image = image_data
        else:
            raise ValueError("Invalid image data format. Expected base64-encoded string or PIL Image.")
        
        # Convert the image to RGB if it isn't already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Make predictions using the model
        pred, idx, probs = model.predict(image)

        # Return the label of the top prediction and its probability
        top_label = labels[idx]
        return {"label": top_label, "probability": float(probs[idx])}
    
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





