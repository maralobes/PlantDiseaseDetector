# Importamos dependencias
import io
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import tensorflow as tf
import joblib

app = FastAPI()

# Cargar el LabelEncoder guardado en la carpeta 'models'
label_encoder = joblib.load('models/label_encoder.pkl')

# Mostrar las clases almacenadas en el LabelEncoder
print("Clases del LabelEncoder:")
print(label_encoder.classes_)

# Ruta del modelo
MODEL_PATH = 'models/modelo_DEP.keras'
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image: Image.Image):
    image = image.resize((128, 128))  # Ajusta al tamaño que espera el modelo
    image_array = np.array(image) / 255.0  # Normaliza los valores de los píxeles
    image_array = np.expand_dims(image_array, axis=0)  # Añade una dimensión para la predicción
    return image_array

@app.post("/plantdisease/")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Leer la imagen subida
        image = Image.open(io.BytesIO(await file.read()))
        image = image.convert('RGB')  # Asegura que la imagen está en formato RGB
        
        # Preprocesar la imagen
        preprocessed_image = preprocess_image(image)
        
        # Hacer la predicción
        predictions = model.predict(preprocessed_image)
        print(f"Predicciones crudas: {predictions}")

        # Obtener la clase con la mayor probabilidad
        predicted_class = np.argmax(predictions, axis=1)[0]
        print(f"Clase predicha (número): {predicted_class}")

        # Usar el LabelEncoder para convertir la clase numérica a etiqueta
        class_label = label_encoder.inverse_transform([predicted_class])[0]
        print(f"Clase predicha (nombre): {class_label}")

        return {"prediction": class_label}
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}
