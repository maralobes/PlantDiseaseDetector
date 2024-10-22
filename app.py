# Importamos dependencias
import io
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# Ruta del modelo
MODEL_PATH = 'model/modelo_DEP.keras'
model = tf.keras.models.load_model(MODEL_PATH)

# Lista de clases con las que entrenaste el modelo
etiquetas = ["Apple Black rot", "Apple Cedar apple rust", "Apple Healthy", "Apple Apple scab",
             "Pepper bell Bacterial spot", "Pepper bell Healthy",
             "Cherry (including sour) Powdery mildew", "Cherry (including sour) Healthy"]

# Crear el LabelEncoder y ajustarlo con las clases
label_encoder = LabelEncoder()
label_encoder.fit(etiquetas)

# Diccionario de clases generado por LabelEncoder
class_names = {i: label for i, label in enumerate(label_encoder.classes_)}

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
        image = image.convert('RGB')  # Asegúrate de que la imagen está en formato RGB
        
        # Preprocesar la imagen
        preprocessed_image = preprocess_image(image)
        
        # Hacer la predicción
        predictions = model.predict(preprocessed_image)
        print(f"Predicciones crudas: {predictions}")

        # Obtener la clase con la mayor probabilidad
        predicted_class = np.argmax(predictions, axis=1)[0]
        print(f"Clase predicha (número): {predicted_class}")

        # Mapeo de la clase a un nombre de enfermedad
        class_label = class_names.get(predicted_class, "Unknown")
        print(f"Clase predicha (nombre): {class_label}")

        return {"prediction": class_label}
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}
