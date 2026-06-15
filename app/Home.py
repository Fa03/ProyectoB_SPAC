"""
Frontend con Streamlit para el Sistema de Predicción de Abandono de Clientes Telco

Ejecutar: streamlit run app/Home.py
URL: http://localhost:8501
"""

import streamlit as st
import requests
import sys
from pathlib import Path

# Añadir path del proyecto
sys.path.append(str(Path(__file__).parent.parent))
from src import config

API_URL = f"http://localhost:{config.API_PORT}"

# === CONFIGURACIÓN DE PÁGINA ===
st.set_page_config(
    page_title="Sistema de Predicción de Abandono de Clientes",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === ESTILOS PERSONALIZADOS ===
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1F4E78;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2E75B5;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1F4E78;
    }
    </style>
""", unsafe_allow_html=True)

# === SIDEBAR ===
st.sidebar.title("📡 Sistema de IA")
st.sidebar.markdown("### Predicción de Abandono de Clientes")
st.sidebar.markdown("---")

# Información del proyecto
st.sidebar.markdown("### 📊 Información del Proyecto")
st.sidebar.info("""
**Equipo:**
- Marco Alvarez Quiros
- Fabián Brenes Loría
- Sharon Obando Gómez

**Curso:** IA Aplicada - CUC  
**Año:** 2026
""")

st.sidebar.markdown("---")

# Enlaces útiles
st.sidebar.markdown("### 🔗 Enlaces")
st.sidebar.markdown("[📖 Documentación API](http://localhost:8000/docs)")
st.sidebar.markdown("[📁 GitHub del Proyecto](https://github.com/Fa03/ProyectoB_SPAC.git)")

# === PÁGINA PRINCIPAL ===

# Header
st.markdown('<h1 class="main-header">📡 Sistema de Predicción de Abandono de Clientes</h1>',
            unsafe_allow_html=True)

st.markdown("""
Este sistema utiliza **ANN Binario - Regresión** para predecir el riesgo de abandono
de clientes de una empresa de telecomunicaciones, permitiendo tomar acciones preventivas de retención.
""")

# === TABS PRINCIPALES ===
tab1, tab2, tab3, tab4 = st.tabs(["📋 Descripción", "🎯 Modelos", "🔮 Predicción", "📈 Resultados"])

with tab1:
    st.markdown('<div class="sub-header">📋 Descripción del Proyecto</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Objetivos")
        st.markdown("""
        - Predecir si un cliente abandonará el servicio
        - Clasificar clientes por nivel de riesgo de abandono
        - Automatizar el proceso de detección temprana de Churn
        - Apoyar estrategias de retención de clientes
        """)

        st.markdown("### 📊 Dataset")
        st.markdown("""
        **Telco Customer Churn**
        - 7,043 clientes
        - Variables demográficas y de servicio
        - Variable objetivo: Churn (Abandona/No abandona)
        """)

    with col2:
        st.markdown("### 🔧 Tecnologías")
        st.markdown("""
        - **TensorFlow/Keras**: Redes neuronales
        - **FastAPI**: API REST
        - **Streamlit**: Frontend interactivo
        - **Scikit-learn**: Preprocesamiento
        """)

        st.markdown("### 📁 Navegación")
        st.info("""
        👈 Usa el menú lateral para navegar o ve a:
        - 🔮 Realizar predicciones individuales
        - 📈 Ver métricas de los modelos
        """)

with tab2:
    st.markdown('<div class="sub-header">🎯 Modelos</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔵 Modelo 1: Clasificación Binaria")
        st.markdown("""
        **Objetivo:** Predecir si el cliente abandonará o no

        **Clases:**
        - ✅ No abandona
        - ❌ Abandona

        **Arquitectura:** ANN Secuencial con 4 capas ocultas Relu - Sigmoid y 1 capa de salida lineal

        **Métricas obtenidas:**
        - Accuracy: 72%
        - F1-Score (No abandona): 0.79
        - F1-Score (Abandona): 0.59
        - Precision (No abandona): 0.89
        - Recall (No abandona): 0.49
        """)

    with col2:
        st.markdown("### 🟢 Modelo 2: Nivel de Riesgo")
        st.markdown("""
        **Objetivo:** Clasificar el nivel de riesgo de abandono

        **Niveles:**
        - 🟢 Bajo riesgo: score < 0.4
        - 🟡 Medio riesgo: score entre 0.4 y 0.7
        - 🔴 Alto riesgo: score >= 0.7

        **Arquitectura:** ANN Secuencial con 4 capas ocultas Relu y 1 capa de salida lineal

        """)

with tab3:
    st.markdown('<div class="sub-header">🔮 Realizar Predicción</div>',
                unsafe_allow_html=True)

    st.markdown("Complete los datos del cliente para obtener la predicción:")

    col1, col2 = st.columns(2)

    with col1:
        gender_Male = st.selectbox("Género", options=[("Masculino", 1.0), ("Femenino", 0.0)],
                                   format_func=lambda x: x[0])[1]
        OnlineSecurity_No = st.selectbox("¿Sin seguridad en línea?", options=[("Sí", 1.0), ("No", 0.0)],
                                         format_func=lambda x: x[0])[1]
        OnlineBackup_No = st.selectbox("¿Sin respaldo en línea?", options=[("Sí", 1.0), ("No", 0.0)],
                                       format_func=lambda x: x[0])[1]
        DeviceProtection_No = st.selectbox("¿Sin protección de dispositivo?", options=[("Sí", 1.0), ("No", 0.0)],
                                           format_func=lambda x: x[0])[1]

    with col2:
        TechSupport_No = st.selectbox("¿Sin soporte técnico?", options=[("Sí", 1.0), ("No", 0.0)],
                                      format_func=lambda x: x[0])[1]
        Contract_Month = st.selectbox("¿Contrato mes a mes?", options=[("Sí", 1.0), ("No", 0.0)],
                                      format_func=lambda x: x[0])[1]
        Contract_Two = st.selectbox("¿Contrato dos años?", options=[("Sí", 1.0), ("No", 0.0)],
                                    format_func=lambda x: x[0])[1]

    st.markdown("---")
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button("🔮 Predecir Abandono (Binario)", use_container_width=True):
            payload = {
                "gender_Male": gender_Male,
                "OnlineSecurity_No": OnlineSecurity_No,
                "OnlineBackup_No": OnlineBackup_No,
                "DeviceProtection_No": DeviceProtection_No,
                "TechSupport_No": TechSupport_No,
                "Contract_Month-to-month": Contract_Month,
                "Contract_Two year": Contract_Two
            }
            try:
                response = requests.post(f"{API_URL}/predict/binary", json=payload)
                result = response.json()

                if result["prediction"] == "Abandona":
                    st.error(f"❌ Predicción: **{result['prediction']}**")
                else:
                    st.success(f"✅ Predicción: **{result['prediction']}**")

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Probabilidad de Abandono", f"{result['probability_churn']:.2%}")
                col_b.metric("Probabilidad de No Abandono", f"{result['probability_no_churn']:.2%}")
                col_c.metric("Confianza", f"{result['confidence']:.2%}")

            except Exception as e:
                st.error(f"⚠️ Error conectando al API: {e}")
                st.info("Asegúrese de que el API esté corriendo en http://localhost:8000")

    with col_btn2:
        if st.button("📊 Predecir Nivel de Riesgo (Regresión)", use_container_width=True):
            payload = {
                "gender_Male": gender_Male,
                "OnlineSecurity_No": OnlineSecurity_No,
                "OnlineBackup_No": OnlineBackup_No,
                "DeviceProtection_No": DeviceProtection_No,
                "TechSupport_No": TechSupport_No,
                "Contract_Month-to-month": Contract_Month,
                "Contract_Two year": Contract_Two
            }
            try:
                response = requests.post(f"{API_URL}/predict/risk_level", json=payload)
                result = response.json()

                nivel = result["risk_level"]
                if nivel == "Alto riesgo":
                    st.error(f"🔴 Nivel de Riesgo: **{nivel}**")
                elif nivel == "Medio riesgo":
                    st.warning(f"🟡 Nivel de Riesgo: **{nivel}**")
                else:
                    st.success(f"🟢 Nivel de Riesgo: **{nivel}**")

                st.metric("Nivel de Riesgo", f"{result['risk_score']:.4f}")
                st.info(f"💡 Recomendación: {result['recommendation']}")

            except Exception as e:
                st.error(f" Error conectando al API: {e}")
                st.info("Asegúrese de que el API esté corriendo en http://localhost:8000")

with tab4:
    st.markdown('<div class="sub-header">📈 Resultados y Conclusiones</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Accuracy (Binario)", value="72%")

    with col2:
        st.metric(label="F1-Score (Abandona)", value="0.59")

    with col3:
        st.metric(label="F1-Score (No abandona)", value="0.79")

    st.markdown("---")

    st.markdown("### 🎓 Conclusiones")
    st.markdown("""
    **Hallazgos principales:**
    - El modelo presenta un desempeño general con un accuracy del 72%.
    - El desbalance de clases en el dataset afecta la capacidad del modelo para identificar correctamente los clientes que abandonan.

    **Recomendaciones:**
    - Ofrecer servicios adicionales como soporte técnico y seguridad en línea para reducir el Churn.

    """)

# === FOOTER ===
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Sistema desarrollado para el curso de Inteligencia Artificial Aplicada</p>
    <p>Colegio Universitario de Cartago (CUC) - 2025</p>
</div>
""", unsafe_allow_html=True)

# === INFORMACIÓN DE DEBUG ===
with st.expander("🔧 Información de Debug"):
    st.markdown("### Configuración del Sistema")
    st.json({
        "PROJECT_ROOT": str(config.PROJECT_ROOT),
        "MODELS_DIR": str(config.MODELS_DIR),
        "API_URL": API_URL
    })