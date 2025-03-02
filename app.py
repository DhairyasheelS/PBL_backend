from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from io import BytesIO

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict it to ["http://localhost:4200"] for security
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load the trained model
model = load_model("braintumor.h5")

# Labels for classification
labels = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

# Function to preprocess the uploaded image
def preprocess_image(image_bytes):
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (150, 150))
    img_array = img.reshape(1, 150, 150, 3)
    return img_array

# API Endpoint to classify an image
@app.post("/predict/")
async def predict_tumor(file: UploadFile = File(...)):
    image_bytes = await file.read()
    processed_image = preprocess_image(image_bytes)
    
    predictions = model.predict(processed_image)
    predicted_class = predictions.argmax()
    predicted_label = labels[predicted_class]

    return {"predicted_class": predicted_label}

# Run with: uvicorn main:app --reload
