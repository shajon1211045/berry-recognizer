from fastai.vision.all import *
import gradio as gr
import base64
import io
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from starlette.responses import JSONResponse

# Initialize FastAPI
app = FastAPI()

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

# Prediction function
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")

        # Make predictions using the model
        pred, idx, probs = model.predict(image)

        # Format the prediction result
        result = {labels[i]: float(probs[i]) for i in range(len(labels))}

        return JSONResponse(content={"prediction": result})
    
    except Exception as e:
        return JSONResponse(content={"error": f"Prediction error: {str(e)}"}, status_code=500)

# Gradio for local testing (optional)
iface = gr.Interface(fn=predict_image, inputs="image", outputs="label", examples=["test_image_01.jpg", "test_image_02.jpg"])
iface.launch(server_name="0.0.0.0", server_port=7860)









