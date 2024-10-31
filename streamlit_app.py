import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

st.write(f"TensorFlow version: {tf.__version__}")

# Ruta directa al archivo .keras en la misma carpeta que el script
modelo_path = 'mobilenet_v2_model2.keras'

# Verificar si el archivo del modelo existe y cargar el modelo
if os.path.exists(modelo_path):
    try:
        model = tf.keras.models.load_model(modelo_path)
        st.success("Modelo cargado correctamente.")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        model = None
else:
    st.error("No se encontró el archivo del modelo en la carpeta actual")

# Verificación de carga de archivo
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"], label_visibility="hidden")

if uploaded_file is not None and model is not None:
    # Mostrar la imagen subida
    st.image(uploaded_file, width=300, caption="Imagen cargada")

    # Preprocesamiento de la imagen para hacer la predicción
    img = image.load_img(uploaded_file, target_size=(224, 224))  # Cambiado a (224, 224)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Realizar la predicción
    prediction = model.predict(img_array)

    # Mostrar resultados
    if prediction[0][0] > 0.5:
        st.success('El modelo predice que la imagen es de un **NORMAL**.')
    else:
        st.success('El modelo predice que la imagen es de un **NEUMONIA**.')
