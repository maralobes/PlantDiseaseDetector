from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import numpy as np
import tensorflow as tf

app = FastAPI()

MODEL_PATH = 'model/modelo_DEP.keras'
model = tf.keras.models.load_model(MODEL_PATH)

class_names = {
    0: "Apple Black rot",
    1: "Apple Cedar apple rust",
    2: "Apple Healthy",
    3: "Apple Apple scab",
    4: "Pepper bell Bacterial spot",
    5: "Pepper bell Healthy",
    6: "Cherry (including sour) Powdery mildew",
    7: "Cherry (including sour) Healthy"
}

def preprocess_image(image: Image.Image):
    image = image.resize((128, 128))  # Asegura que las dimensiones sean correctas
    image_array = np.array(image)
    print(f"Image array shape: {image_array.shape}")  # Verifica las dimensiones
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.post("/plantdisease/")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Leer la imagen subida por el usuario
        image = Image.open(io.BytesIO(await file.read()))
        image = image.convert('RGB')  # Asegura que esté en formato RGB
        
        # Mostrar el tipo y tamaño de la imagen
        print(f"Tipo de imagen: {image.format}, Tamaño de imagen: {image.size}")

        # Preprocesar la imagen para el modelo
        preprocessed_image = preprocess_image(image)
        
        # Hacer la predicción
        predictions = model.predict(preprocessed_image)
        
        # Mostrar las predicciones crudas
        print(f"Predicciones crudas: {predictions}")

        # Obtener la clase con la mayor probabilidad
        predicted_class = np.argmax(predictions, axis=1)[0]
        class_label = class_names.get(predicted_class, "Unknown")
        
        return {"prediction": class_label}
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}
