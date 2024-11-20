import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict, cross_validate, train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, RobustScaler

from sklearn import linear_model, metrics, model_selection
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
import matplotlib as mpl
from sklearn.ensemble import VotingRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.metrics import  mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import xgboost
import warnings
import pickle
from sklearn.feature_selection import SelectKBest, f_regression, r_regression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.cluster import KMeans
from lightgbm import LGBMRegressor
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import plot_importance
from sklearn.neural_network import MLPRegressor
from evaluation import evaluate_model
import json
import yaml
import tensorflow as tf
from tensorflow import keras

# Carga de los datos procesados
df= pd.read_csv("../data/processed/data_processed.csv")

df_train, df_test = train_test_split(df, 
                                     test_size=0.20,
                                     random_state=63)


#Eliminamos outliers de precio en train
df_train = df_train[df_train["price"]<=150]


# Guardamos los dataframes de train y test en csv 
df_train.to_csv("../data/train/data_train.csv", index=False)
df_test.to_csv("../data/test/data_test.csv", index=False)

# Definimos las variables de X_train e y_train de los datos de train
X_train = df_train.drop(columns="points")
y_train = df_train['points']

# También los de test
X_test = df_test.drop(columns="points")
y_test = df_test['points']


# Definimos las columnas categóricas y numéricas
categorical_many_unique = ['winery', 'variety', 'denominacion']  # Usar OrdinalEncoder
categorical_few_unique = ['style', 'aging_1']  # Usar OneHotEncoder
numeric_columns = ['price', 'vintage']

estimator = 'Keras'
decision = 1
# Paso 1: Crear transformers
many_unique_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)  # Para muchas categorías
few_unique_transformer = OneHotEncoder(handle_unknown='ignore')  # Para pocas categorías
numeric_transformer = StandardScaler()  # Para numéricos

# Paso 2: Combinar transformers en un ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('many_unique', many_unique_transformer, categorical_many_unique),
        ('few_unique', few_unique_transformer, categorical_few_unique),
        ('num', numeric_transformer, numeric_columns)
    ])

# Transformar los datos
X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)

# Convertir a arrays de NumPy
X_train_np = np.asarray(X_train_trans.todense() if hasattr(X_train_trans, 'todense') else X_train_trans)
X_test_np = np.asarray(X_test_trans.todense() if hasattr(X_test_trans, 'todense') else X_test_trans)

# Definición del modelo con Keras
model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=[X_train_np.shape[1]]),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(25, activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1)  # Salida para regresión
])

# Compilar el modelo
model.compile(
    loss="mean_absolute_error",
    optimizer="adam",
    metrics=[keras.metrics.RootMeanSquaredError(name='rmse')]
)

# Callbacks
checkpoint_cb = keras.callbacks.ModelCheckpoint("callback_model.keras", save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3)

# Entrenamiento del modelo
history = model.fit(
    X_train_np,
    y_train,
    epochs=50,
    validation_data=(X_test_np, y_test),
    callbacks=[checkpoint_cb, early_stopping_cb]
)

# Evaluación del modelo
mse_test = model.evaluate(X_test_np, y_test)

# Predicciones
y_pred = model.predict(X_test_np)

# Imprimir métricas de evaluación
print("Mean Absolute Error:", mse_test[0])
print("Root Mean Squared Error:", mse_test[1])
metrics_df = pd.DataFrame({
    "Model": ["Keras Sequential"],
    "MAE": [mse_test[0]],  
    "RMSE": [mse_test[1]]
})
metrics_df.to_csv(f"../src/{estimator}/results_{estimator}.csv", index=False)


pickle.dump(history, open(f"../src/{estimator}/keras_model.pkl", "wb"))
print(f"Modelo guardado en PICKLE.")

print("Proceso completado.")