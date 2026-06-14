# Importamos las libreria.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense,Dropout

#------------------------------------------------------------------------------------------------------------#

# Creamos la clase para nuestro Modelo.
class ModeloRegresion:
    def __init__(self): # Constructor de la clase.
        self.df = pd.read_csv('C:/Users/sharo/Documents/pys examen/telco_churn_reducido.csv') # Llamamos el csv limpio

#-----------------------------------Creamos los Metodos-----------------------------------------------------------#

# 1. Entrenamiento y prueba.
    def entrena_prueba(self, df):
        X = drop('Churn', axis=1) # Le indicamos que usa todas las variables de entrada menos la variable objetivo
        y = self.df['Churn_Yes'] # Le indicamos que solo usa la variable objetivo

    # Aqui dividimos los datos para poder hacer el entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

        print('Resultado de X_train.shape', X_train.shape)
        print('Resultado de y_test.shape', y.shape)
        return X, y, X_train, X_test, y_train, y_test # Le indicamos que nos devuelva los valores

#------------------------------------------------------------------------------------------------------------#

# 2. Creacion del Modelo.
    def creacion_model(self, X, y, X_train, X_test, y_train, y_test):
        self.model = Sequential()
        num_neuronas = X_train.shape[1]  # Número de neuronas según tamaño del DF
        self.model.add(Dense(units=num_neuronas, activation='relu'))
        self.model.add(Dropout(0.5))  # la mitad de las neuronas en cada epoch para esta capa

        self.model.add(Dense(units=int(np.round(num_neuronas // 2)),
                        activation='relu'))  # Agregado para que de un número entero
        self.model.add(Dropout(0.5))  # la mitad de las neuronas en cada epoch para esta capa

        self.model.add(Dense(units=1, activation='sigmoid'))  # Neuronas de salida igual a variables a predecir, en este caso solo 1 variable "Churn

        # Usamos el 'binary_crossentropy' para poder realizar la probabilidad de churn.
        self.model.compile(loss='binary_crossentropy', optimizer='adam') # Utilizamos adam para empezar a dar saltos grandes y saltos pequeños antes de llegar al 0.

        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=23)

        self.model.fit(x=X_train,
                  y=y_train,
                  epochs=600,
                  validation_data=(X_test, y_test), verbose=1,
                  callbacks=[early_stop])

        losses = pd.DataFrame(self.model.history.history)  # perdidas de muestro modelo(loss) y el de test(val_loss)
        losses.plot()

#---------------------------------------------------------------------------------------------------------------------#

# Creacion para predecir el nivel de riesgo (risk_score).
    def predecir_riegos(self, X.test):






