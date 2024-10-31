import streamlit as st

# Cargar y mostrar una imagen
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        st.image(uploaded_file, width=300, caption="Imagen cargada")
        st.success("Imagen subida y mostrada correctamente.")
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")
else:
    st.info("Por favor, sube una imagen en formato jpg, jpeg o png.")
