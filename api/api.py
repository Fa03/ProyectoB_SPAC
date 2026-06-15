from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os
from pathlib import Path

# 1. Inicializar FastAPI
app = FastAPI(title="API para utilizar modelo binario creado")

# 2. Cargar el modelo neuronal
try:
    carpeta_models = Path.cwd().parent / 'models'  # ubicación de la carpeta
    model = tf.keras.models.load_model(carpeta_models / 'modelo_TELCO.keras')
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None


# Función auxiliar para abrir la ventana Pop-up de Windows/Mac/Linux
def abrir_ventana_seleccion():
    # Crear una ventana oculta de Tkinter para que no aparezca un cuadro vacío de fondo
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)  # Fuerza a la ventana a ponerse al frente

    # Abrir el explorador de archivos filtrando solo por archivos CSV
    ruta_seleccionada = filedialog.askopenfilename(
        title="Selecciona el set de datos CSV para la predicción",
        filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
    )

    root.destroy()  # Cerrar por completo la instancia de tkinter
    return ruta_seleccionada


# 3. Endpoint modificado
@app.post("/predict-from-popup")
def predict_from_popup():
    if model is None:
        raise HTTPException(status_code=500, detail="El modelo no está disponible.")

    # Ejecutar la ventana emergente para obtener la ruta tipo C:/...
    print("Abriendo ventana emergente para seleccionar archivo CSV...")
    ruta_csv = abrir_ventana_seleccion()
    print(f"Ruta seleccionada: {ruta_csv}")

    # Si el usuario cancela la ventana emergente sin elegir nada
    if not ruta_csv:
        return {"status": "cancelled", "message": "No se seleccionó ningún archivo."}

    try:
        # Leer el CSV seleccionado desde la ruta obtenida por el Pop-up
        # df_datos = pd.read_csv(ruta_csv)

        # --- NOTA: Aquí debes adaptar los datos del DataFrame al formato de tu modelo ---
        # Ejemplo: Convertir el dataframe a matriz de numpy si ya viene limpio
        # input_array = df_datos.to_numpy()

        # Realizar la predicción
        prediction = model.prediccion(model, ruta_csv)
        # resultado = prediction.tolist()

        return {
            "status": "success",
            "archivo_procesado": os.path.basename(ruta_csv),
            "ruta_completa": ruta_csv,
            "prediction": prediction
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar el archivo o en la predicción: {str(e)}")


@app.get("/")
def read_root():
    return {"message": "API activa. Usa /predict-from-popup para abrir el buscador de archivos."}