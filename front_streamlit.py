# Frontend (FE)
import io
import requests
import streamlit as st
from PIL import Image

# Definición de la URL de la API
API_URL = "http://localhost:8031/plantdisease/"

# Streamlit interface, # Configuración de la página
st.title('Análisis de enfermedades en plantas')
st.markdown('Sube una imagen de tu planta enferma: \n')

# Cargar la imagen
image_load = st.file_uploader("Sube tu imagen aquí:", type=["jpg", "jpeg", "png"])  # Permitir solo imágenes

if st.button('Analizar imagen'):
    if image_load:
        # Convertir la imagen cargada en un formato adecuado para enviar a la API
        image = Image.open(image_load)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Preparar los datos para enviar a la API (imagen en formato bytes)
        files = {'file': ('image.png', img_byte_arr, 'image/png')}
        
        # Hacer la solicitud POST a la API
        response = requests.post(API_URL, files=files)
        
        if response.status_code == 200:
            # Procesar los resultados devueltos por la API
            results = response.json()
            st.write("Predicción de enfermedad:", results.get("prediction"))
        else:
            st.error("Error en la respuesta de la API")
    else:
        st.warning("Por favor, sube una imagen para analizar.")
