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
MODEL_PATH = 'models/modelo_DEP 92%.keras'
model = tf.keras.models.load_model(MODEL_PATH)

# Diccionario de enfermedades  
disease_info = {  
    "apple apple scab": {  
        "description": {  
            "en": "Apple scab is a common disease of plants in the rose family (Rosaceae) caused by the fungus Venturia inaequalis. Symptoms include dark, irregular lesions on leaves and fruits.",  
            "es": "El tizón del manzano es una enfermedad común de las plantas de la familia Rosaceae causada por el hongo Venturia inaequalis. Los síntomas incluyen lesiones oscuras e irregulares en las hojas y frutos."  
        },  
        "treatment": {  
            "en": "Preventive practices include sanitation and resistance breeding, along with targeted fungicide or biocontrol treatments.",  
            "es": "Las prácticas preventivas incluyen la desinfección y el mejoramiento de resistencia, junto con tratamientos fungicidas o de biocontrol específicos."  
        }  
    },  
    "apple black rot": {  
        "description": {  
            "en": "Black rot is caused by the fungus Diplodia seriata, affecting dead and living tissue.",  
            "es": "La podredumbre negra es causada por el hongo Diplodia seriata, que afecta el tejido muerto y vivo."  
        },  
        "treatment": {  
            "en": "Use Mancozeb or Ziram fungicides before infection to prevent spore germination.",  
            "es": "Utilizar fungicidas Mancozeb o Ziram antes de la infección para prevenir la germinación de esporas."  
        }  
    },  
    "apple cedar apple rust": {  
        "description": {  
            "en": "Cedar apple rust requires plants from two families to complete its life cycle.",  
            "es": "El óxido del manzano requiere plantas de dos familias diferentes para completar su ciclo de vida."  
        },  
        "treatment": {  
            "en": "Apply fungicides with Myclobutanil before symptoms appear.",  
            "es": "Aplicar fungicidas con Myclobutanil antes de que aparezcan los síntomas."  
        }  
    },  
    "cherry (including sour) powdery mildew": {  
        "description": {  
            "en": "Powdery mildew on cherry is caused by Podosphaera clandestina. It renders affected fruits unmarketable due to white fungal growth.",  
            "es": "El moho polvoriento en las cerezas es causado por Podosphaera clandestina. Hace que las frutas afectadas no sean comercializables debido al crecimiento fúngico blanco."  
        },  
        "treatment": {  
            "en": "A solution of 2-3 tablespoons of apple cider vinegar with a gallon of water can control powdery mildew.",  
            "es": "Una solución de 2-3 cucharadas de vinagre de sidra de manzana con un galón de agua puede controlar el moho polvoriento."  
        }  
    },  
    "pepper bell bacterial spot": {  
        "description": {  
            "en": "Bacterial spot is a devastating disease of pepper and tomato, causing significant yield reduction.",  
            "es": "La mancha bacteriana es una enfermedad devastadora del pimiento y el tomate, causando una reducción significativa en el rendimiento."  
        },  
        "treatment": {  
            "en": "Seed treatment with hot water can reduce bacterial populations; ensure proper temperature to avoid affecting germination.",  
            "es": "El tratamiento de semillas con agua caliente puede reducir las poblaciones bacterianas; asegúrese de la temperatura adecuada para evitar afectar la germinación."  
        }  
    },  
    "healthy": {  
        "description": {  
            "en": "The plant is healthy and free from diseases.",  
            "es": "La planta está saludable y libre de enfermedades."  
        },  
        "treatment": {  
            "en": "Maintain good agricultural practices, proper watering, and pest management to ensure plant health.",  
            "es": "Mantenga buenas prácticas agrícolas, un riego adecuado y el manejo de plagas para asegurar la salud de la planta."  
        }  
    }  
}  
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
