#!/usr/bin/env bash

# Iniciar Streamlit  
streamlit run front_streamlit.py --server.port=8534 --server.headless=True &  
if [ $? -ne 0 ]; then  
    echo "Error al iniciar Streamlit"  
fi  

# Iniciar API (FastAPI)  
uvicorn app:app --host 0.0.0.0 --port 8031 &  
if [ $? -ne 0 ]; then  
    echo "Error al iniciar Uvicorn"  
fi  