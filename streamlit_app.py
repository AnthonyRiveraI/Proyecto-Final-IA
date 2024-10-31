import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import wget
import os

st.write(f"TensorFlow version: {tf.__version__}")

# Descargar el modelo si no existe
def download_model():
    model_url = 'https://dl.dropboxusercontent.com/s/lxb3qm5cdpa9ake5z4q9t/best_model_local.zip?rlkey=z56wewjj51qxk6djzfihi4aq1&st=04u131mj'
    zip_path = 'best_model.zip'
    extract_folder = 'extracted_files'

    # Descargar el archivo zip si no existe
    if not os.path.exists(zip_path):
        try:
            wget.download(model_url, zip_path)
            st.success("Modelo descargado correctamente.")
        except Exception as e:
            st.error(f"Error al descargar el modelo: {e}")
            return None

    # Descomprimir el archivo solo si no está ya descomprimido
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
    
    # Ruta al archivo del modelo .keras descomprimido
    return os.path.join(extract_folder, 'best_model_local.keras')

# Descargar y cargar el modelo
modelo_path = download_model()

# Verificar si el archivo del modelo existe y cargar el modelo
if modelo_path and os.path.exists(modelo_path):
    try:
        model = tf.keras.models.load_model(modelo_path)
        st.success("Modelo cargado correctamente.")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        model = None
else:
    st.error("No se encontró el archivo del modelo")

# Verificación de carga de archivo
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"], label_visibility="hidden")

if uploaded_file is not None and model is not None:
    # Mostrar la imagen subida
    st.image(uploaded_file, width=300, caption="Imagen cargada")

    # Preprocesamiento de la imagen para hacer la predicción
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Realizar la predicción
    prediction = model.predict(img_array)

    # Mostrar resultados
    if prediction[0][0] > 0.5:
        st.success('El modelo predice que la imagen es de un **NORMAL**.')
    else:
        st.success('El modelo predice que la imagen es de un **NEUMONIA**.')
