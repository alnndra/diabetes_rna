import streamlit as st, numpy as np, joblib
from tensorflow import keras

st.set_page_config(page_title='Predictor Diabetes RNA', page_icon='🩺', layout='wide')

@st.cache_resource
def cargar():
    return keras.models.load_model('modelo_rna.keras'), joblib.load('scaler.pkl')

modelo, scaler = cargar()
st.title('🩺 Predictor de Diabetes — Red Neuronal Artificial')
st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    edad = st.number_input('Edad', 16, 90, 45)
    genero = st.selectbox('Género', ['Masculino', 'Femenino'])
    poliuria = st.checkbox('Poliuria')
    polidipsia = st.checkbox('Polidipsia')
    perdida = st.checkbox('Pérdida repentina de peso')
    debilidad = st.checkbox('Debilidad')
with col2:
    polifagia = st.checkbox('Polifagia')
    candidiasis = st.checkbox('Candidiasis genital')
    vision = st.checkbox('Visión borrosa')
    picazon = st.checkbox('Picazón')
    irritab = st.checkbox('Irritabilidad')
    cicatriz = st.checkbox('Cicatrización lenta')
with col3:
    paresia = st.checkbox('Paresia parcial')
    rigidez = st.checkbox('Rigidez muscular')
    alopecia = st.checkbox('Alopecia')
    obesidad = st.checkbox('Obesidad')
st.divider()

if st.button('🔍 Predecir Riesgo', use_container_width=True):
    # En el CSV GOLD real, Genero está codificado 0/1 (no 1/2 como dice el enunciado).
    g = 1 if genero == 'Masculino' else 0
    x = np.array([[edad, g, poliuria, polidipsia, perdida, debilidad, polifagia,
                   candidiasis, vision, picazon, irritab, cicatriz, paresia,
                   rigidez, alopecia, obesidad]], dtype=float)
    x = scaler.transform(x)
    prob = float(modelo.predict(x, verbose=0)[0][0])
    if prob >= 0.5:
        st.error(f'⚠️ RIESGO ALTO de Diabetes — {prob*100:.1f}%')
    else:
        st.success(f'✅ RIESGO BAJO de Diabetes — {prob*100:.1f}%')
    st.progress(min(max(prob, 0.0), 1.0))
    st.caption('⚠️ Resultado orientativo. Consulte siempre a un médico.')
