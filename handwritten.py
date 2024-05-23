from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import gradio as gr

# App title
title = "Welcome to your first handwritten recognition app!"

# You can load any model from Hugging Face
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')

# Prediction function for handwriting
def predict(image_url, img_upload):

    # Fetch the image from URL or handwritten canvas or the uploaded image
    if image_url:
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    else:
        image = img_upload.convert("RGB")

    # Predict the image using the Microsoft/TROCR-large-handwritten model loaded earlier
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

# Gradio interface
interface = gr.Interface(fn=predict,
                         inputs=["text",
                                 gr.Image(type="pil", label="Upload image")],
                         outputs="text",
                         title=title)

try:
    interface.launch(server_name="localhost", server_port=8080)
except KeyboardInterrupt:
    print("Keyboard interrupt received, shutting down the server.")
