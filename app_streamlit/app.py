import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
import os
sys.path.append(os.path.abspath('../src'))
import utils  ### TODO: Intentar que sea relative path


# Cargar el modelo pre-entrenado
model = pickle.load(open('../models/final_model.pkl', 'rb'))

# Cargar diccionario de opciones para botones
dict_clasif = pickle.load(open("../src/options_dict.pkl", 'rb'))
style_dict = dict_clasif["style"].keys()
style_dict = list(style_dict)[:-1]
variety_dict = dict_clasif["style"]
winery_dict = dict_clasif["wineries"]
denominacion_dict = dict_clasif["denominacion"]
aging_dict = dict_clasif["aging"]
aging_dict.append("Ninguna de las anteriores")


# Configuración de la página
st.set_page_config(page_title="Wine Score", layout="centered")
st.markdown("<style>body {background-color: #7B1E44;}</style>", unsafe_allow_html=True)

# Título de la aplicación
st.title("🎨🍷 Wine Score: ¿Qué puntuación tendría este vino en una cata a ciegas?")

 # imagen o visualización
st.image("../docs/imagen_streamlit.png")

# Introducción
st.write("""
¿Alguna vez te has preguntado qué puntuación recibiría un vino en una cata a ciegas? 
Con esta aplicación, podrás predecir el rango de calidad en el que podría clasificarse 
un vino basándonos en sus características clave. ¡Déjate guiar por nuestra herramienta 
y explora el fascinante mundo de la enología!
""")
    
# Escala de calidad
st.header("Escala de Calidad")

st.markdown("""
Cuando la app predice la puntuación de un vino, lo clasifica en uno de los siguientes rangos de calidad:

- 🏆 **Clásico**: *El pináculo de la calidad.*  
    **Puntuación:** 98-100  
    Un vino excepcional que define la excelencia en la enología. Perfecto para coleccionistas y ocasiones especiales.

- 🌟 **Excepcional**: *Un gran logro.*  
    **Puntuación:** 94-97  
    Un vino sobresaliente que no puede faltar en la mesa de los conocedores.

- ✅ **Excelente**: *Altamente recomendado.*  
    **Puntuación:** 90-93  
    Un vino de gran calidad, ideal para quienes buscan algo especial.

- 🍇 **Muy bueno**: *A menudo buena relación calidad-precio; bien recomendado.*  
    **Puntuación:** 87-89  
    Una opción destacada para disfrutar en compañía o para una ocasión casual.

- 🍷 **Bueno**: *Adecuado para el consumo diario, a menudo buena relación calidad-precio.*  
    **Puntuación:** 83-86  
    Un vino que cumple, perfecto para el día a día sin sacrificar calidad.

- 🍂 **Aceptable**: *Puede ser empleado.*  
    **Puntuación:** 80-82  
    Aunque modesto, este vino puede complementar tus comidas sin grandes pretensiones.

- ❓ **Sin calificar**:  
    Por debajo de 80 puntos, este vino no alcanza los estándares para recibir una calificación destacada.
""")

# Cierre inspirador
st.write("✨ **Introduce las características del vino y descubre en qué rango de calidad podría encontrarse. ¡Que comience la experiencia enológica!** 🍷")

# Selección de país
country = st.selectbox("País de origen", options= ['España']) #['Francia', 'Italia', 'España', 'Portugal']

# Selección de estilo de vino
style = st.radio("Estilo de vino", options=style_dict)
winery = st.selectbox("Bodega", options= list(winery_dict))
if style:
    variety = st.selectbox("Variedad de uva", options= variety_dict[style])
denominacion = st.selectbox("Denominación de origen / Denominación geográfica", options= list(denominacion_dict))
aging = st.radio("Clasificación", options= list(aging_dict))
year = st.slider("Año de cosecha", min_value=1900, max_value=2023, step=1)
price = st.number_input("Precio estimado", min_value=0.0)


# Botón para realizar la predicción
if st.button("Predecir valoración"):
    # Preprocesar los datos
    input_data = pd.DataFrame({
        'vintage': [year],
        'winery': [winery],
        'style': [style],
        'variety': [variety],
        'denominacion': [denominacion],
        'price': [price],
        'aging_1': [aging],
    })

    # Realizar la predicción
    prediction = model.predict(input_data)
    pred_num= prediction[0]   
    print("Valor", pred_num)
    category = utils.categorize_points(pred_num)
    print("Categoría ", category)


    # Resultado de la clasificación
    st.subheader("🍷 ¡Aquí está el resultado de tu vino!")
    
    # Mensaje final después de mostrar la clasificación
    st.write(f"""
    🎉 **¡Felicidades!** Según sus características, tu vino podría clasificarse como: 
              
    **{category}**  

    Esta predicción refleja cómo podría ser percibido en una cata a ciegas.  

    Ya sea un vino clásico digno de coleccionistas o una opción más modesta para el día a día, cada vino tiene su momento especial.  

    ✨ **Ahora es tu turno de disfrutarlo y compartirlo!**  
    
    ¿Quién sabe? Tal vez este vino sea la estrella de tu próxima reunión o el compañero perfecto para un momento tranquilo.
    """)

    # Sugerencia adicional
    st.write("🍇 Si tienes otro vino en mente, ¡introduce sus datos y descubre qué puntuación podría alcanzar!")




# para abrir en terminal: streamlit run nombre_app.py