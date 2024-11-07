# Importamos dependencias
import io
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import json
import logging
import tensorflow as tf
import joblib

app = FastAPI()

# Cargar el LabelEncoder guardado en la carpeta 'models'
label_encoder = joblib.load('models/label_encoder.pkl')

# Mostrar las clases almacenadas en el LabelEncoder
print("Clases del LabelEncoder:")
print(label_encoder.classes_)

# Definir las clases 
class_names = [ 
    "Cherry (including sour) Powdery mildew", 
    "Cherry (including_sour) healthy", 
    "Pepper bell Bacterial spot", 
    "Pepper bell healthy", 
    "Strawberry Leaf scorch", 
    "Strawberry healthy" 
    ] 
print(class_names)

# Cargar el diccionario desde el archivo JSON 
with open('disease_info.json', 'r') as f: 
    disease_info = json.load(f)
    
# Ruta del modelo
MODEL_PATH = 'models/modelo_DEP94.keras'
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
        class_predicted_number = int(np.argmax(predictions, axis=1)[0])
        print(f"Clase predicha (número): {class_predicted_number}")

        # Usar el LabelEncoder para convertir la clase numérica a etiqueta
        class_predicted_name = label_encoder.inverse_transform([class_predicted_number])[0]
        print(f"Clase predicha (nombre): {class_predicted_name}")

        # Obtener la descripción y tratamiento de disease_info usando el número de clase 
        class_predicted_number_str = str(class_predicted_number) 
        if class_predicted_number_str in disease_info: 
            disease_data = disease_info[class_predicted_number_str] 
            description = disease_data['description'] 
            treatment = disease_data['treatment'] 
            return { 
                    "prediction": class_predicted_number,   
                    "description": description, 
                    "treatment": treatment 
            } 
        else: 
            return {"error": "Información de enfermedad no disponible."}
        
    except Exception as e: 
        print(f"Error: {str(e)}") 
        return {"error": str(e)} 