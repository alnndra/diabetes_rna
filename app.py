import streamlit as st, numpy as np, joblib
from tensorflow import keras

st.set_page_config(page_title='Predictor de Diabetes', page_icon='🩺', layout='wide')

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp { background: linear-gradient(160deg,#0b1220 0%,#0f172a 100%); color:#e2e8f0;
  font-family:'Inter',sans-serif; }
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding-top:2.2rem; max-width:1150px; }
.hero { display:flex; align-items:center; gap:14px; margin-bottom:4px; }
.hero h1 { font-family:'Inter'; font-weight:800; font-size:30px; color:#f1f5f9; margin:0; letter-spacing:-.5px;}
.hero .ico { width:46px; height:46px; background:#0ea5e9; border-radius:12px; display:flex;
  align-items:center; justify-content:center; box-shadow:0 8px 24px rgba(14,165,233,.35);}
.lead { color:#94a3b8; font-size:15px; margin:0 0 22px 60px; }
.card { background:#111c30; border:1px solid #26324a; border-radius:16px; padding:22px 24px;
  box-shadow:0 10px 30px rgba(0,0,0,.25); }
.card h3 { font-size:13px; font-weight:700; letter-spacing:.6px; text-transform:uppercase;
  color:#38bdf8; margin:0 0 16px; }
.card h3.mt { margin-top:22px; }
.result-label { font-weight:700; font-size:19px; margin:6px 0 2px; }
.interp { color:#cbd5e1; font-size:14px; line-height:1.5; margin-top:8px; }
.disc { color:#64748b; font-size:12px; margin-top:16px; border-top:1px solid #26324a; padding-top:12px; }
.ph { color:#64748b; font-size:14px; text-align:center; padding:30px 10px; }
label, [data-testid="stWidgetLabel"] p { font-family:'Inter' !important; color:#cbd5e1 !important; font-size:14px;}
.stButton>button { font-family:'Inter'; font-weight:700; font-size:15px; background:#0ea5e9; color:#06121f;
  border:0; border-radius:10px; padding:12px; transition:.15s; }
.stButton>button:hover { background:#38bdf8; color:#06121f; }
.stButton>button:active { transform:translateY(1px); }
[data-testid="stCheckbox"] { margin-bottom:-6px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

@st.cache_resource
def cargar():
    return keras.models.load_model('modelo_rna.keras'), joblib.load('scaler.pkl')
modelo, scaler = cargar()

SYMPTOMS = [
    ('poliuria','Poliuria'), ('polidipsia','Polidipsia'),
    ('perdida','Pérdida repentina de peso'), ('debilidad','Debilidad'),
    ('polifagia','Polifagia'), ('candidiasis','Candidiasis genital'),
    ('vision','Visión borrosa'), ('picazon','Picazón'),
    ('irritab','Irritabilidad'), ('cicatriz','Cicatrización lenta'),
    ('paresia','Paresia parcial'), ('rigidez','Rigidez muscular'),
    ('alopecia','Alopecia'), ('obesidad','Obesidad'),
]

def gauge(prob, color):
    total = 282.74
    fill = prob * total
    return (f'<div style="text-align:center">'
            f'<svg viewBox="0 0 220 128" width="100%" style="max-width:270px">'
            f'<path d="M20,112 A90,90 0 0 1 200,112" fill="none" stroke="#1e293b" '
            f'stroke-width="18" stroke-linecap="round"/>'
            f'<path d="M20,112 A90,90 0 0 1 200,112" fill="none" stroke="{color}" '
            f'stroke-width="18" stroke-linecap="round" stroke-dasharray="{fill:.1f} {total:.1f}"/>'
            f'<text x="110" y="98" text-anchor="middle" font-family="Inter,sans-serif" '
            f'font-size="40" font-weight="800" fill="{color}">{prob*100:.0f}%</text>'
            f'<text x="110" y="120" text-anchor="middle" font-family="Inter,sans-serif" '
            f'font-size="12" fill="#64748b">probabilidad estimada</text></svg></div>')

HEART = ('<div class="ico"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" '
         'stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
         '<path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.6l-1-1a5.5 5.5 0 0 0-7.8 7.8l1 1L12 21l7.8-7.6 1-1a5.5 5.5 0 0 0 0-7.8z"/>'
         '</svg></div>')

if 'revealed' not in st.session_state:
    st.session_state.revealed = False

st.markdown(f'<div class="hero">{HEART}<h1>Predictor de Diabetes</h1></div>', unsafe_allow_html=True)
st.markdown('<p class="lead">Evaluación de riesgo mediante una Red Neuronal Artificial '
            '(MLP) entrenada sobre 520 pacientes.</p>', unsafe_allow_html=True)

left, right = st.columns([1.1, 0.9], gap='large')

with left:
    st.markdown('<div class="card"><h3>Datos del paciente</h3>', unsafe_allow_html=True)
    edad = st.slider('Edad', 16, 90, 45)
    genero = st.radio('Género', ['Masculino', 'Femenino'], horizontal=True)
    st.markdown('<h3 class="mt">Síntomas presentes</h3>', unsafe_allow_html=True)
    c = st.columns(2)
    vals = {}
    for i, (key, label) in enumerate(SYMPTOMS):
        with c[i % 2]:
            vals[key] = st.checkbox(label, key=key)
    st.write('')
    evaluar = st.button('Evaluar riesgo', use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if evaluar:
    st.session_state.revealed = True
    st.rerun()

g = 1 if genero == 'Masculino' else 0
prob = None
if st.session_state.revealed:
    x = np.array([[edad, g] + [int(vals[k]) for k, _ in SYMPTOMS]], dtype=float)
    prob = float(modelo.predict(scaler.transform(x), verbose=0)[0][0])

with right:
    if prob is None:
        st.markdown('<div class="card"><h3>Resultado</h3>'
                    '<div class="ph">Completa el formulario y pulsa <b>Evaluar riesgo</b> '
                    'para ver la estimación del modelo.</div></div>', unsafe_allow_html=True)
    else:
        alto = prob >= 0.5
        color = '#ef4444' if alto else '#10b981'
        label = 'Riesgo alto de diabetes' if alto else 'Riesgo bajo de diabetes'
        interp = ('El modelo estima una probabilidad elevada. Se recomienda una '
                  'evaluación médica para confirmación y seguimiento.') if alto else (
                  'El modelo estima una probabilidad baja según los síntomas indicados.')
        n = sum(vals.values())
        st.markdown(
            f'<div class="card"><h3>Resultado</h3>{gauge(prob, color)}'
            f'<div class="result-label" style="color:{color}">{label}</div>'
            f'<div class="interp">{interp}</div>'
            f'<div class="interp" style="color:#94a3b8">Síntomas seleccionados: {n} de 14 · '
            f'Edad: {edad} · Género: {genero}.</div>'
            f'<div class="disc">Resultado orientativo generado por un modelo estadístico. '
            f'No sustituye el diagnóstico de un profesional de la salud.</div></div>',
            unsafe_allow_html=True)
