#!/bin/bash

# Crear entorno (environment) para API
python3 -m venv api_env

# Activar entorno para API donde instalaremos todo lo necesario
source api_env/bin/activate 

# Instalación de dependencias para ejecutar API
pip install --upgrade pip
pip install -r requirements.txt

# Mensaje de éxito
echo "Entorno virtual creado y dependencias instaladas."

