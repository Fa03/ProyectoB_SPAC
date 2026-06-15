"""
API REST con FastAPI para el Sistema de Predicción de Abandono de Clientes

Endpoints:
- GET /: Información de la API
- POST /predict/binary: Predicción binaria (Abandona/No abandona)
- POST /predict/risk_level: Predicción de nivel de riesgo
- GET /health: Estado del servidor

Ejecutar: uvicorn main:app --reload
Documentación: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import sys
from pathlib import Path
from tensorflow import keras

# Añadir path del proyecto
sys.path.append(str(Path(__file__).parent.parent))
from src import config

# Rutas de modelos
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

# Crear aplicación FastAPI
app = FastAPI(
    title="Sistema de Predicción de Abandono de Clientes",
    description="API para predicción de Abandono de Clientes usando ANN",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === MODELOS DE DATOS (SCHEMAS) ===

class ChurnApplication(BaseModel):
    """Schema para predicción de abandono de cliente"""
    gender_Male: float = Field(..., description="Género (1=Masculino, 0=Femenino)")
    OnlineSecurity_No: float = Field(..., description="Sin seguridad en línea (1=Si, 0=No)")
    OnlineBackup_No: float = Field(..., description="Sin respaldo en línea (1=Si, 0=No)")
    DeviceProtection_No: float = Field(..., description="Sin protección de dispositivo (1=Si, 0=No)")
    TechSupport_No: float = Field(..., description="Sin soporte técnico (1=Si, 0=No)")
    Contract_Month_to_month: float = Field(..., alias="Contract_Month-to-month", description="Contrato mes a mes (1=Si, 0=No)")
    Contract_Two_year: float = Field(..., alias="Contract_Two year", description="Contrato dos años (1=Si, 0=No)")

    class Config:
        populate_by_name = True
        schema_extra = {
            "example": {
                "gender_Male": 0.0,
                "OnlineSecurity_No": 1.0,
                "OnlineBackup_No": 0.0,
                "DeviceProtection_No": 1.0,
                "TechSupport_No": 1.0,
                "Contract_Month-to-month": 1.0,
                "Contract_Two year": 0.0
            }
        }

class BinaryPredictionResponse(BaseModel):
    """Respuesta de predicción binaria"""
    prediction: str = Field(..., description="No abandona o Abandona")
    probability_churn: float = Field(..., description="Probabilidad de abandono")
    probability_no_churn: float = Field(..., description="Probabilidad de no abandono")
    confidence: float = Field(..., description="Confianza de la predicción")

class RiskLevelResponse(BaseModel):
    """Respuesta de nivel de riesgo"""
    risk_level: str = Field(..., description="Bajo riesgo, Medio riesgo, Alto riesgo")
    risk_score: float = Field(..., description="Puntaje de riesgo entre 0 y 1")
    recommendation: str = Field(..., description="Recomendación")

# === CARGAR MODELOS ===

binary_model = None
risk_model = None

try:
    binary_model = keras.models.load_model(MODELS_DIR / 'modelo_TELCO.keras')
    risk_model = keras.models.load_model(MODELS_DIR / 'modelo_risk_score.keras')
    print(" Modelos cargados correctamente")
except Exception as e:
    print(f" Error cargando modelos: {e}")
    print("Los endpoints de predicción no funcionarán hasta cargar los modelos")

# === FUNCIONES AUXILIARES ===

def preprocess_input(data: ChurnApplication):
    """Preprocesa los datos de entrada sin escalar"""
    input_dict = data.dict()
    input_array = np.array(list(input_dict.values())).reshape(1, -1)
    return input_array

def nivel_riesgo(score: float) -> str:
    """Determina el nivel de riesgo según el score"""
    if score >= 0.7:
        return 'Alto riesgo'
    elif score >= 0.4:
        return 'Medio riesgo'
    else:
        return 'Bajo riesgo'

# === ENDPOINTS ===

@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "API de Predicción de Abandono de Clientes",
        "version": "1.0.0",
        "endpoints": {
            "/docs": "Documentación interactiva",
            "/predict/binary": "Predicción binaria (Abandona/No abandona)",
            "/predict/risk_level": "Predicción de nivel de riesgo",
            "/health": "Estado del servidor"
        }
    }

@app.get("/health")
async def health_check():
    """Verifica el estado del servidor y modelos"""
    models_loaded = binary_model is not None and risk_model is not None

    return {
        "status": "healthy" if models_loaded else "models_not_loaded",
        "binary_model": binary_model is not None,
        "risk_model": risk_model is not None,
        "message": "API funcionando correctamente" if models_loaded else "Verificar modelos"
    }

@app.post("/predict/binary", response_model=BinaryPredictionResponse)
async def predict_binary(application: ChurnApplication):
    """
    Predice si el cliente abandonará o no el servicio
    Returns:
        Predicción binaria con probabilidades
    """
    if binary_model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    try:
        X = preprocess_input(application)
        proba = float(binary_model.predict(X)[0][0])

        prediction = "Abandona" if proba > 0.5 else "No abandona"
        confidence = max(proba, 1 - proba)

        return BinaryPredictionResponse(
            prediction=prediction,
            probability_churn=proba,
            probability_no_churn=1 - proba,
            confidence=confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.post("/predict/risk_level", response_model=RiskLevelResponse)
async def predict_risk_level(application: ChurnApplication):
    """
    Predice el nivel de riesgo de abandono del cliente
    Returns:
        Nivel de riesgo y recomendación
    """
    if risk_model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    try:
        X = preprocess_input(application)
        score = float(risk_model.predict(X)[0][0])
        nivel = nivel_riesgo(score)

        recommendations = {
            "Bajo riesgo":  "Cliente estable. No requiere intervención.",
            "Medio riesgo": "Cliente en riesgo moderado. Contactar por medio de ofertas.",
            "Alto riesgo":  "Cliente en riesgo de abandono. Intervención urgente requerida!."
        }

        return RiskLevelResponse(
            risk_level=nivel,
            risk_score=score,
            recommendation=recommendations[nivel]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

# === EJECUTAR ===
if __name__ == "__main__":
    import uvicorn
    print("Iniciando servidor FastAPI...")
    print("Documentación: http://localhost:8000/docs")
    uvicorn.run("api:app", host=config.API_HOST, port=config.API_PORT, reload=config.API_RELOAD)