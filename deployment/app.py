from fastai.vision.all import *
import gradio as gr

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
model = load_learner("D:\\berry_Type_recognizer\\models\\berry-recognizer-v2.pkl")

# Define the prediction function
def recognize_image(image):
    pred, idx, probs = model.predict(image)
    return dict(zip(labels, map(float, probs)))

# Set up Gradio interface (removed shape and added 'type' for image)
image = gr.Image(type="pil")  # 'pil' refers to PIL image format, which is compatible with fastai
label = gr.Label(num_top_classes=10)

examples = [
    "D:\\berry_Type_recognizer\\test images\\test_image_01.jpg",
    "D:\\berry_Type_recognizer\\test images\\test_image_02.jpg",
    "D:\\berry_Type_recognizer\\test images\\test_image_03.jpg",
    "D:\\berry_Type_recognizer\\test images\\test_image_04.jpg",
    "D:\\berry_Type_recognizer\\test images\\test_image_05.jpg"
]

iface = gr.Interface(fn=recognize_image, inputs=image, outputs=label, examples=examples)
iface.launch(share=True)


