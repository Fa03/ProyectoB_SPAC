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
from sklearn.metrics import classification_report,confusion_matrix

#------------------------------------------------------------------------------------------------------------#

# 1. Creamos la clase para nuestro Modelo.
class ModeloRegresion:
    def __init__(self): # Constructor de la clase.
        self.df = pd.read_csv('C:/Users/sharo/Documents/pys examen/telco_churn_reducido.csv') # Llamamos el csv limpio

#-----------------------------------Creamos los Metodos-----------------------------------------------------------#

# 2. Entrenamiento y prueba.
    def entrena_prueba(self, df):
        X = df.drop('Churn', axis=1) # Le indicamos que usa todas las variables de entrada menos la variable objetivo
        y = df['Churn'] # Le indicamos que solo usa la variable objetivo

    # Aqui dividimos los datos para poder hacer el entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

        print('Resultado de X_train.shape', X_train.shape)
        print('Resultado de y_test.shape', y.shape)
        return X, y, X_train, X_test, y_train, y_test # Le indicamos que nos devuelva los valores

#------------------------------------------------------------------------------------------------------------#

# 3. Creacion del Modelo.
    def creacion_model(self, X, y, X_train, X_test, y_train, y_test):
        self.model = Sequential()
        num_neuronas = X_train.shape[1]  # Número de neuronas según tamaño del DF
        self.model.add(Dense(units=num_neuronas, activation='relu'))
        self.model.add(Dropout(0.3))  # la mitad de las neuronas en cada epoch para esta capa

        self.model.add(Dense(units=int(np.round(num_neuronas // 2)),
                        activation='relu'))  # Agregado para que de un número entero
        self.model.add(Dropout(0.3))  # la mitad de las neuronas en cada epoch para esta capa

        self.model.add(Dense(units=1, activation='sigmoid'))  # Neuronas de salida igual a variables a predecir, en este caso solo 1 variable "Churn

        # Usamos el 'binary_crossentropy' para poder realizar la probabilidad de churn.
        self.model.compile(loss='binary_crossentropy', optimizer='adam') # Utilizamos adam para empezar a dar saltos grandes y saltos pequeños antes de llegar al 0.

        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)

        self.model.fit(x=X_train,
                  y=y_train,
                  epochs=600,
                  validation_data=(X_test, y_test), verbose=1,
                  callbacks=[early_stop])

        losses = pd.DataFrame(self.model.history.history)  # perdidas de muestro modelo(loss) y el de test(val_loss)
        losses.plot()
        plt.show()

#---------------------------------------------------------------------------------------------------------------------#

# 4. Creacion para predecir el nivel de riesgo (risk_score) (Consultado con Claude).
    def predecir_riesgo(self, X_test): # Creamos el metodo de predecir el riesgo y como parametro agregamos el X.test
            scores = self.model.predict(X_test)

            resultados = pd.DataFrame()
            resultados['risk_score'] = scores.flatten()

            def nivel(score): # creamos el nivel de riesgo.
                if score >= 0.7:
                    return 'Alto riesgo'
                elif score >= 0.4:
                    return 'Medio riesgo'
                else:
                    return 'Bajo riesgo'

            resultados['nivel_riesgo'] = resultados['risk_score'].apply(nivel)

            print('\nDistribución por nivel de riesgo:')
            print(resultados['nivel_riesgo'].value_counts())
            return resultados

#---------------------------------------------------------------------------------------------------------------------#

# 5. Predicción sobre el conjunto de Test.
    def prediccion(self, X_test, y_test):
        predictions = self.model.predict(X_test)

        print('Error absoluto medio', mean_absolute_error(predictions, y_test))
        print('Error cuadratico medio', np.sqrt(mean_squared_error(y_test,predictions)))
# #Calcula la covarianza en y_test (real) y la predicción, cuanto más cercano a 1 mejor
# (significa que conforme más aumenta o disminuye el valor real, más aumenta o disminuye
# el valor predecido)
        print('Varianza', explained_variance_score(y_test,predictions))
        print('Error absoluto medio', mean_absolute_error(y_test,predictions)/self.df['Churn'].mean())
        print('Error absoluto para mediana', mean_absolute_error(y_test,predictions)/self.df['Churn'].median())

# Visualizamos nuestra prediccion
        plt.scatter(y_test, predictions)
        plt.xlabel('Valor Real')
        plt.ylabel('Prediccion')

#Predicción perfecta
        plt.plot([0, 1], [0, 1], 'r--', label='Predicción perfecta') # Consultado con Claude (error con grafico)
        plt.show()

# Rotar y_test para poder comparar con predictions
        errors = y_test.values.reshape(y_test.shape[0], 1) - predictions
        sns.distplot(errors)  # Idealmente debe estar concentrado el error en 0
        plt.show()

#----------------------------------------------------------------------------------------------------------#

# 6. Valores de accurancy y blabla.
    def prediccion_acc(self, X_test, y_test):
        predictions = self.model.predict(X_test)

# predictions = model.predict_classes(X_test)
        predictions = (self.model.predict(X_test) > 0.5).astype("int32")  # Si nueva versión de scikit-learn

# Accurancy
        print(classification_report(y_test, predictions))

# Matriz de Confuncion
        print(confusion_matrix(y_test, predictions))

#------------------------------------------------------------------------------------------------------------#

# 7. Guardar modelo.
    def guardar_modelo(self):
        self.model.save('modelo_risk_score.keras') # Lo guardamos .keras
        print('Se ha guardado el modelo_risk_score.keras con éxito!')

#-----------------------------------------------------------------------------------------------------------#

modelo = ModeloRegresion() # Llamamos la clase del modelo
X, y, X_train, X_test, y_train, y_test = modelo.entrena_prueba(modelo.df) # Dividmos los datos en entrenamiento y prueba
modelo.creacion_model(X, y, X_train, X_test, y_train, y_test)  # Entrenamos el modelo y visualizamos su val_loss
modelo.prediccion(X_test, y_test) # Metricas de error
modelo.prediccion_acc(X_test, y_test) # Llamamos el metodo de accurancy
resultados = modelo.predecir_riesgo(X_test) # Llamamos el metodo del nivel de riesgo
print(resultados.head(10)) # Imprime los resultados de riesgo
modelo.guardar_modelo() # Guardamos el modelo
