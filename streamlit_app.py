import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Verificar y listar archivos en la carpeta actual
st.write("Archivos en la carpeta actual:", os.listdir("."))

# Ruta directa al archivo .keras en la misma carpeta que el script
modelo_path = 'mobilenet_v2_model2.keras'

# Verificar si el archivo del modelo existe y cargar el modelo
if os.path.exists(modelo_path):
    try:
        model = tf.keras.models.load_model(modelo_path)
        st.success("Modelo cargado correctamente.")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        model = None
else:
    st.error("No se encontró el archivo del modelo en la carpeta actual")

# Carga de archivo para hacer una predicción
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    st.image(uploaded_file, width=300, caption="Imagen cargada")

    # Preprocesamiento de la imagen para hacer la predicción
    img = image.load_img(uploaded_file, target_size=(224, 224))
