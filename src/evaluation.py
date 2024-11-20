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

# Creamos un dataframe para guardar los valores de las métricas de cada modelo y proceso (train, test)
results_df = pd.DataFrame(columns=['Model', 'Partition', 'R^2', 'MAE', 'RMSE'])
def evaluate_model(model_name, y_true, y_pred, partition):
    r2 =     metrics.r2_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    rmse =  metrics.root_mean_squared_error(y_true, y_pred)
    return {
        'Model': model_name,
        'Partition': partition,
        'R^2': r2,
        'MAE': mae,
        'RMSE': rmse
    }