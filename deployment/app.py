from fastai.vision.all import *
import gradio as gr
import base64
import io
from PIL import Image

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
        # Ensure image_data is a base64-encoded string or PIL Image
        if isinstance(image_data, str):
            # Split the base64 string and decode
            try:
                image_data = image_data.split(",")[1]  # Remove the base64 header
                image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGB")
            except Exception as e:
                return {"error": f"Base64 decoding error: {str(e)}"}
        elif isinstance(image_data, Image.Image):
            image = image_data.convert("RGB")  # Ensure it's in RGB format
        else:
            return {"error": "Invalid image data format. Expected base64-encoded string or PIL Image."}

        # Make predictions using the model
        pred, idx, probs = model.predict(image)

        # Return predictions in the desired format
        return {labels[i]: float(probs[i]) for i in range(len(labels))}
    
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

# Set up Gradio interface
image = gr.Image(type="pil")
label = gr.Label(num_top_classes=10)

# Example image paths
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








