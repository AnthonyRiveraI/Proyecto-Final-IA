import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Inicializar `session_state` para la imagen subida
if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None

# Mostrar la versión de TensorFlow
st.write(f"TensorFlow version: {tf.__version__}")

# Ruta al modelo
modelo_path = 'mobilenet_v2_model2.keras'

# Cargar el modelo
if os.path.exists(modelo_path):
    try:
        model = tf.keras.models.load_model(modelo_path)
        st.success("Modelo cargado correctamente.")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        model = None
else:
    st.error("No se encontró el archivo del modelo en la carpeta actual")

# Verificación de carga de archivo
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

# Permitir predicción si se sube la misma imagen
if uploaded_file is not None:
    st.session_state['uploaded_file'] = uploaded_file  # Guardar en `session_state`

if st.session_state['uploaded_file'] is not None and model is not None:
    # Mostrar la imagen subida
    st.image(st.session_state['uploaded_file'], width=300, caption="Imagen cargada")

    # Botón para procesar la predicción
    if st.button("Realizar predicción"):
        # Preprocesamiento de la imagen
        img = image.load_img(st.session_state['uploaded_file'], target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Realizar la predicción
        prediction = model.predict(img_array)

        # Mostrar resultados
        if prediction[0][0] > 0.5:
            st.success('El modelo predice que la imagen es de un **NORMAL**.')
        else:
            st.success('El modelo predice que la imagen es de un **NEUMONIA**.')
