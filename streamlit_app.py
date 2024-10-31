import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Imprimir versión de TensorFlow para asegurar compatibilidad
st.write(f"TensorFlow version: {tf.__version__}")

# Definir la ruta al modelo .keras
modelo_path = 'mobilenet_v2_model2.keras'

# Cargar el modelo solo si el archivo existe
if os.path.exists(modelo_path):
    try:
        model = tf.keras.models.load_model(modelo_path)
        st.success("Modelo cargado correctamente.")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        model = None
else:
    st.error("No se encontró el archivo del modelo en la carpeta actual")

# Cargar y preprocesar la imagen para hacer una predicción
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Mostrar la imagen subida
    st.image(uploaded_file, width=300, caption="Imagen cargada")

    # Cargar y redimensionar la imagen
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Realizar la predicción y mostrar el resultado
    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.success('El modelo predice que la imagen es de un **NORMAL**.')
    else:
        st.success('El modelo predice que la imagen es de un **NEUMONIA**.')
