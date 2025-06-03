from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import onnxruntime as ort
from io import BytesIO
import torchvision.transforms as transforms
from labels import label_map


app = FastAPI()

# Load the ONNX model
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image bytes
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).numpy()

    # Run inference
    outputs = session.run([output_name], {input_name: input_tensor})
    predicted_class = int(np.argmax(outputs[0]))

    return JSONResponse(content={
    "predicted_class_index": predicted_class,
    "predicted_label": label_map.get(predicted_class, "unknown")})
