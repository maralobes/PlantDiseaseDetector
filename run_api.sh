#!/bin/bash

# Iniciar Streamlit (Si no funciona ejecutar una primera vez streamlit fuera del script)
streamlit run front_streamlit.py --server.port=8534 --server.headless=True &

# Iniciar API (FastAPI)
uvicorn app:app --host 0.0.0.0 --port 8031 &