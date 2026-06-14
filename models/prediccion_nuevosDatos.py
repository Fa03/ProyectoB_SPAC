import sklearn

from clasificacion_binaria import ClasificacionBinaria
import pandas as pd
from sklearn.preprocessing import StandardScaler

class PrediccionNuevosDatos:
    def __init__(self):
        pass

    def prediccion(self,model):
        # Lectura datos a predecir
        df_pred = pd.read_csv('C:/Users/fab_t/Downloads/telco_churn_generado_1000.csv')
        # df_pred = df_pred[
        #     df.columns]  # Filtramos el dataframe predicción para quedarnos con las mismas columnas de interés que el df histórico
        X_pred = df_pred.drop('Churn_Yes',
                              axis=1)  # Eliminamos la columna objetivo (vacía) si la hubiera en nuestro df con los datos a predecir
        X_pred = scaler.transform(X_pred)
        # Realizamos predicción
        resultado = model.predict_classes(X_pred)
        # Unimos en un dataframe los datos a predecir con su predicción
        df_pred['PREDICCIÓN'] = pd.DataFrame(resultado)
        print(df_pred)


nueva_prediccion = PrediccionNuevosDatos()
modelo_clasificacionBi = ClasificacionBinaria()
X,Y, X_train, X_test, y_train, y_test = modelo_clasificacionBi.dividir_df(modelo_clasificacionBi.df)
modelo_creado = modelo_clasificacionBi.crearModeloBinario(X,Y, X_train, X_test, y_train, y_test)
nueva_prediccion.prediccion(modelo_creado)