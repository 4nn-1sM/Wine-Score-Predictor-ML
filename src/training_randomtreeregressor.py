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

estimator = 'RandomTreeRegressor'
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

# Creamos función para entrenar los modelos
def entrenar_modelo(pipeline, params, X_train, y_train, X_test, y_test, modelo, decision):
    results_train = pd.DataFrame(columns=['Model', 'Partition', 'R^2', 'MAE', 'RMSE'])
    results_test = pd.DataFrame(columns=['Model', 'Partition', 'R^2', 'MAE', 'RMSE'])
    model = RandomizedSearchCV(pipeline, params, n_iter=200, cv=5, scoring='neg_root_mean_squared_error', random_state=63, verbose=0, n_jobs=-1)
    model.fit(X_train, y_train)
    model_best_estimator = model.best_estimator_
    model_best_params = model.best_params_
    model_best_score = model.best_score_
    model_best_index = model.best_index_
    print("-" * 50)
    print("Resultados Grid Search")
    print("-" * 50)
    print(f'{modelo}{decision}_best_estimator:', model_best_estimator)
    print(f'{modelo}{decision}_best_params:', model_best_params)
    print(f'{modelo}{decision}_best_score:', model_best_score)
    print(f'{modelo}{decision}_best_index:', model_best_index)
    y_predict_train = model_best_estimator.predict(X_train)
    y_predict_test = model_best_estimator.predict(X_test)
    results_train = pd.concat([results_train, pd.DataFrame([evaluate_model(f'{modelo} {decision}', y_train, y_predict_train, 'Train')])], ignore_index=True)
    results_test = pd.concat([results_test, pd.DataFrame([evaluate_model(f'{modelo} {decision}', y_test, y_predict_test, 'Test')])], ignore_index=True)
    return {
        "best_estimator": model_best_estimator,
        "best_params": model_best_params,
        "best_score": model_best_score,
        "train_results": results_train,
        "test_results": results_test
    }    


# Paso 3: Crear el pipeline completo
Selectkbest_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Preprocesamiento
    ('scaler',StandardScaler()), 
    ('feature_selection', SelectKBest(score_func=f_regression, k='all')),
    ("regressor", DecisionTreeRegressor(max_depth=5, random_state=63))
])

Cluster_pipe = Pipeline(steps = [
    ('preprocessor', preprocessor),  # Preprocesamiento
    ('scaler',StandardScaler()), 
    ("clustering", KMeans(n_clusters=3, random_state=63)),
    ("regressor", DecisionTreeRegressor(max_depth=5, random_state=63))
])

PCA_pipe = Pipeline(steps = [
    ('preprocessor', preprocessor),  # Preprocesamiento
    ('scaler',StandardScaler()), 
    ("pca", PCA(n_components=4)),
    ("regressor", DecisionTreeRegressor(max_depth=5, random_state=63))
])


# Parámetros
Selectkbest_param = {
    'scaler': [StandardScaler(), MinMaxScaler(), RobustScaler(), None],
    'feature_selection__k': np.arange(1,15),  
    "regressor__max_depth":np.arange(1,20), # Profundidades del árbol. Cuanto más profundo, mas posibilidades de overfitting,
    "regressor__min_samples_split": np.arange(2, 21),  # Mínimo número de muestras para dividir un nodo
    "regressor__min_samples_leaf": np.arange(1, 11),  # Mínimo número de muestras en una hoja
}
Cluster_param = {
    'scaler': [StandardScaler(), MinMaxScaler(), RobustScaler(), None],
    'clustering__n_clusters': np.arange(1,15),
    'clustering__init': ['k-means++'], 
    "regressor__max_depth":np.arange(1,20), # Profundidades del árbol. Cuanto más profundo, mas posibilidades de overfitting,
    "regressor__min_samples_split": np.arange(2, 21),  # Mínimo número de muestras para dividir un nodo
    "regressor__min_samples_leaf": np.arange(1, 11),  # Mínimo número de muestras en una hoja
}
PCA_param = {
    'scaler': [StandardScaler(), MinMaxScaler(), RobustScaler(), None],
    'pca__n_components': np.arange(1,25),  
    "regressor__max_depth":np.arange(1,20), # Profundidades del árbol. Cuanto más profundo, mas posibilidades de overfitting,
    "regressor__min_samples_split": np.arange(2, 21),  # Mínimo número de muestras para dividir un nodo
    "regressor__min_samples_leaf": np.arange(1, 11),  # Mínimo número de muestras en una hoja
}


pipes = [Selectkbest_pipe, Cluster_pipe, PCA_pipe]
params = [Selectkbest_param, Cluster_param, PCA_param]
i = 1


def entrenar_y_guardar_modelos(pipes, params, X_train, y_train, X_test, y_test, modelos, decision_base=1):
    """
    Entrena y guarda los resultados de múltiples pipelines con sus parámetros.
    
    Args:
        pipes (list): Lista de pipelines para entrenar.
        params (list): Lista de diccionarios de parámetros correspondientes a los pipelines.
        X_train (pd.DataFrame): Variables predictoras del conjunto de entrenamiento.
        y_train (pd.Series): Variable objetivo del conjunto de entrenamiento.
        X_test (pd.DataFrame): Variables predictoras del conjunto de prueba.
        y_test (pd.Series): Variable objetivo del conjunto de prueba.
        modelos (list): Lista de nombres de modelos correspondientes a los pipelines.
        decision_base (int): Valor inicial para la numeración de decisiones.

    Returns:
        None
    """
    decision = decision_base

    for pipe, param, modelo in zip(pipes, params, modelos):
        print("-" * 50)
        print(f"Entrenando modelo {modelo} con decisión {decision}...")
        print("-" * 50)
        
        # Entrenar el modelo
        resultado = entrenar_modelo(pipe, param, X_train, y_train, X_test, y_test, modelo, decision)

        # Guardar los resultados
        # Guardar train y test en CSV
        resultado["train_results"].to_csv(f"../models/{estimator}/train_results_{modelo}{decision}.csv", index=False)
        print(f"Resultados train del modelo {modelo} guardados en CSV.")

        resultado["test_results"].to_csv(f"../models/{estimator}/test_results_{modelo}{decision}.csv", index=False)
        print(f"Resultados test del modelo {modelo} guardados en CSV.")

        # Guardar el mejor modelo en PICKLE
        pickle.dump(resultado["best_estimator"], open(f"../models/{estimator}/best_estimator_{modelo}{decision}.pkl", "wb"))
        print(f"Best estimator del modelo {modelo} guardado en PICKLE.")

        # Guardar los mejores parámetros en YAML
        yaml.dump(resultado["best_params"], open(f"../models/{estimator}/best_params_{modelo}{decision}.yaml", "w"), default_flow_style=False, allow_unicode=True)
        print(f"Best params del modelo {modelo} guardados en YAML.")

        # Guardar el mejor score en CSV
        pd.DataFrame({"best_score": [resultado["best_score"]]}, index=[f'{modelo}{decision}']).to_csv(f"../models/{estimator}/best_score_{modelo}{decision}.csv")
        print(f"Best score del modelo {modelo} guardado en CSV.")
        
        # Incrementar el contador de decisión
        decision += 1

    print("Proceso completado para todos los modelos.")

# Lista de nombres de modelos
caracteristicas = ["SelectKBest", "Clustering", "PCA"]

# Llamar a la función con tus datos
entrenar_y_guardar_modelos(pipes, params, X_train, y_train, X_test, y_test, caracteristicas, decision_base=1)