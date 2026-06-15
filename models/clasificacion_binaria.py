

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense,Dropout
from pathlib import Path
from sklearn.preprocessing import StandardScaler


class ClasificacionBinaria:



    def __init__(self):
        self.df = pd.read_csv('C:/Users/fab_t/Downloads/Excel_telco_churn_reducido_Copy.csv')


    def dividir_df(self, df):
        scaler = StandardScaler()
        X = df.drop('Churn', axis=1).values
        y = df['Churn'].values.astype(int)

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.25, random_state=101, stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=101, stratify=y_train_val
        )
        # Escalado de variables
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        print('Distribución de clases:')
        print('Train:', np.bincount(y_train))
        print('Val  :', np.bincount(y_val))
        print('Test :', np.bincount(y_test))

        # Para guardar datos de entrenamiento en la carpeta "data/processed", se crean DataFrames a partir de arrays
        df_Train = pd.DataFrame(X_train)
        df_Test = pd.DataFrame(X_test)

        # Indicación de rutas de guardado
        carpeta = Path.cwd().parent / 'data' / 'processed'  # ubicación de la carpeta
        df_Train.to_csv(carpeta / 'train_binario.csv', index=False)
        df_Test.to_csv(carpeta / 'test_binario.csv', index=False)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def crearModeloBinario(self,X,Y, X_train, X_test, y_train, y_test):
        model = Sequential()
        num_neuronas = X.shape[1] #~** número de neuronas según tamaño del DF
        model.add(Dense(units=num_neuronas,activation='relu'))
        model.add(Dropout(0.4))  # la mitad de las neuronas en cada epoch para esta capa

        model.add(Dense(units=int(np.round(num_neuronas/2)),activation='relu'))  #~** // agregado para que de un número entero
        model.add(Dropout(0.4))  # la mitad de las neuronas en cada epoch para esta capa

        model.add(Dense(units=1,activation='sigmoid')) # ~** neuronas de salida igual a variables a predecir, en este caso solo 1 variable "Churn

        # Para clasificación binaria "binary_crossentropy"
        model.compile(loss='binary_crossentropy', optimizer='adam')

        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20) # ~** patience disminuida


        # Entrenamiento del modelo
        model.fit(x=X_train,
                  y=y_train,
                  epochs=500, #~** reducido de 600
                  validation_data=(X_test, y_test), verbose=1,
                  callbacks=[early_stop]
                  )

        model_loss = pd.DataFrame(model.history.history)
        model_loss.plot()
        plt.show()

        return model


    def evaluar_modelo(self,X_test,y_test, model):
        predictions = (model.predict(X_test) > 0.5).astype("int32")
        print(classification_report(y_test, predictions))
        print(confusion_matrix(y_test, predictions))

    def guardar_modelo(self,model, nombre_modelo):
        model.save(nombre_modelo + ".keras")

    def prediccion(self, model):
        # Lectura datos a predecir
        df_pred = pd.read_csv('C:/Users/fab_t/Downloads/telco_churn_generado_3000.csv')
        # df_pred = df_pred[
        #     df.columns]  # Filtramos el dataframe predicción para quedarnos con las mismas columnas de interés que el df histórico
        X_pred = df_pred.drop('Churn',
                              axis=1)  # Eliminamos la columna objetivo (vacía) si la hubiera en nuestro df con los datos a predecir
        #X_pred = scaler.transform(X_pred)
        # Realizamos predicción
        resultado = (model.predict(X_pred) > 0.5).astype("int32")
        # Unimos en un dataframe los datos a predecir con su predicción
        df_pred['PREDICCIÓN'] = pd.DataFrame(resultado)
        print(df_pred)


prueba_binaria = ClasificacionBinaria()
X_train, X_val, X_test, y_train, y_val, y_test = prueba_binaria.dividir_df(prueba_binaria.df)
modelo_creado = prueba_binaria.crearModeloBinario(X_val,y_val, X_train, X_test, y_train, y_test)
prueba_binaria.evaluar_modelo(X_test, y_test, modelo_creado)
prueba_binaria.guardar_modelo(modelo_creado,"modelo_TELCO")

prueba_binaria.prediccion(modelo_creado)





























