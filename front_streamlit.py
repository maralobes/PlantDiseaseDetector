# Frontend (FE)
import io
import requests
import streamlit as st
from PIL import Image
import json
import time

# Definición de la URL de la API
API_URL = "http://localhost:8031/plantdisease/"

# Cargar el diccionario del archivo json 
with open('disease_info.json', 'r', encoding='utf-8') as json_file: 
    disease_info = json.load(json_file)

# Configuración de la página
st.set_page_config(page_title="Análisis de Enfermedades en Plantas", layout="wide") 
st.title('🔍 Análisis de Enfermedades en Plantas') 
st.markdown('Sube una imagen de una hoja de tu planta enferma con un fondo claro y homogéneo para que sea analizada:')

# Cargar la imagen 
image_load = st.file_uploader("📥Arrastra y suelta tu imagen aquí o haz clic para cargar (Formatos: JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"])

if image_load is not None:
    # Mostrar la imagen cargada
    st.image(image_load, caption="Imagen subida", use_container_width=True)

if st.button('Analizar imagen'):
    if image_load:
        # Mostrar la imagen cargada 
        image = Image.open(image_load).convert('RGB') # Asegurar de que está en formato RGB 
        
        # Convertir la imagen a bytes 
        img_byte_arr = io.BytesIO() 
        image.save(img_byte_arr, format='PNG') 
        img_byte_arr = img_byte_arr.getvalue()

        # Preparar los datos para enviar a la API (imagen en formato bytes)
        files = {'file': ('image.png', img_byte_arr, 'image/png')}
        
        # Hacer la solicitud POST a la API
        with st.spinner("Analizando la imagen..."):
            time.sleep(2)  # Espera de 2 segundos para simular procesamiento
            try:
                response = requests.post(API_URL, files=files) 
                response.raise_for_status() # Lanza un error para respuestas no exitosas 
                results= response.json()
                
                # Extraer los datos de la respuesta 
                class_predicted_number = results.get('prediction') 
                print(f"Clase predicha (número): {class_predicted_number}")
                
                if str(class_predicted_number) in disease_info: 
                    class_predicted_name = disease_info[str(class_predicted_number)].get('name') 
                    description = disease_info[str(class_predicted_number)].get('description') 
                    treatment = disease_info[str(class_predicted_number)].get('treatment')
                    
                    # Mostrar información adicional
                    with st.expander("Información completa sobre la enfermedad"): 
                        st.markdown(f"**Predicción:** {class_predicted_name}") 
                        st.markdown(f"**Descripción (EN):** {description['en']}") 
                        st.markdown(f"**Descripción (ES):** {description['es']}") 
                        st.markdown(f"**Tratamiento (EN):** {treatment['en']}") 
                        st.markdown(f"**Tratamiento (ES):** {treatment['es']}")
                else:
                    st.error("⚠️Error: Clase predicha no encontrada en el archivo JSON.")
                    
            except requests.exceptions.RequestException as e: 
                st.error(f"⚠️Error en la conexión a la API: {e}") 
            except KeyError as e: 
                st.error(f"⚠️Error: No se encontró la clave {e} en el archivo JSON.") 
            
    else: 
        st.warning("⚠️Por favor, sube una imagen para analizar.") 
        
# Opción para reiniciar 
if st.button("🔄Resetear"): 
    # Limpiar el estado de la aplicación 
    st.session_state.clear() # Borra todo el estado en Streamlit 
    st.experimental_rerun() # Recargar la página completamente 
            