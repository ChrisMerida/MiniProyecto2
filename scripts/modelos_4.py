"""
    Este script contiene la función que genera distintos modelos predictivos
"""
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score

def modelos():
    """
        Función crear modelos
    """
    data = pd.read_csv("../data/processed/loan_data_prepared.csv")

    # Selección de Variables
    X = data.drop(columns=['loan_status'])
    y = data['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify=y)

    # Codificar Variables
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # Escalado y Ajuste
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Regresión Logística
    # Hiper-parámetros
    param_grid_logit = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    }

    # Regresion Logistica
    logit = LogisticRegression()
    grid_logit = GridSearchCV(estimator=logit, param_grid=param_grid_logit, cv=5, scoring='roc_auc')
    grid_logit.fit(X_train_scaled, y_train_encoded)

    # Predicción y Evaluación
    logit_best = grid_logit.best_estimator_
    logit_predicts = logit_best.predict(X_test_scaled)
    roc_auc = roc_auc_score(y_test_encoded, logit_best.predict_proba(X_test_scaled)[:, 1])

    logit_results = pd.DataFrame(grid_logit.cv_results_).sort_values("rank_test_score", ascending=True)[['rank_test_score', 'params', 'mean_test_score', 'std_test_score']]
    logit_results['algoritmo'] = 'Regresión Logística'
    logit_results = logit_results[['algoritmo', 'rank_test_score', 'params', 'mean_test_score', 'std_test_score']]

    # Decesion Tree
    #Hiper-parámetros
    param_grid_dt = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=2025)
    grid_dt = GridSearchCV(estimator=dt, param_grid=param_grid_dt, cv=5, scoring='roc_auc')
    grid_dt.fit(X_train_scaled, y_train_encoded)

    # Predicción y Evaluación
    dt_best = grid_dt.best_estimator_
    dt_predicts = dt_best.predict(X_test_scaled)
    roc_auc_dt = roc_auc_score(y_test_encoded, dt_predicts)

    dt_results = pd.DataFrame(grid_dt.cv_results_).sort_values("rank_test_score", ascending=True)[['rank_test_score', 'params', 'mean_test_score', 'std_test_score']]
    dt_results['algoritmo'] = 'Decision Tree'
    dt_results = dt_results[['algoritmo', 'rank_test_score', 'params', 'mean_test_score', 'std_test_score']]

    # Random Forest
    # Hiper-parámetros
    param_grid_rf = {
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 20, 30],
        'criterion': ['gini', 'entropy']
    }

    # Random Forest
    rf = RandomForestClassifier()
    grid_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='roc_auc')
    grid_rf.fit(X_train_scaled, y_train_encoded)

    # Predicción y Evaluación
    rf_best = grid_rf.best_estimator_
    rf_predicts = rf_best.predict(X_test_scaled)
    roc_auc_rf = roc_auc_score(y_test_encoded, rf_predicts)

    rf_results = pd.DataFrame(grid_rf.cv_results_).sort_values("rank_test_score", ascending=True)[['rank_test_score', 'params', 'mean_test_score', 'std_test_score']]
    rf_results['algoritmo'] = 'Random Forest'
    rf_results = rf_results[['algoritmo', 'rank_test_score', 'params', 'mean_test_score', 'std_test_score']]

    # AdaBoost
    # Hiper-parámetros
    param_grid_ada = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    }

    # AdaBoost
    ada = AdaBoostClassifier()
    grid_ada = GridSearchCV(estimator=ada, param_grid=param_grid_ada, cv=5, scoring='roc_auc')
    grid_ada.fit(X_train_scaled, y_train_encoded)

    # Predicciones y Evaluación
    ada_best = grid_ada.best_estimator_
    ada_predicts = ada_best.predict(X_test_scaled)
    roc_auc = roc_auc_score(y_test_encoded, ada_predicts)

    ada_results = pd.DataFrame(grid_ada.cv_results_).sort_values("rank_test_score", ascending=True)[['rank_test_score', 'params', 'mean_test_score', 'std_test_score']]
    ada_results['algoritmo'] = 'AdaBoost'
    ada_results = ada_results[['algoritmo', 'rank_test_score', 'params', 'mean_test_score', 'std_test_score']]

    # XGBoost
    # Hiper-parámetros
    param_grid_xgb = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 5, 7]
    }

    # XGBoost
    xgb_model = xgb.XGBClassifier(eval_metric='logloss')
    grid_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, cv=5, scoring='roc_auc')
    grid_xgb.fit(X_train_scaled, y_train_encoded)

    # Predicciones y evaluación
    xgb_best = grid_xgb.best_estimator_
    xgb_predicts = xgb_best.predict(X_test_scaled)
    roc_auc = roc_auc_score(y_test_encoded, xgb_predicts)

    xgb_results = pd.DataFrame(grid_xgb.cv_results_).sort_values("rank_test_score", ascending=True)[['rank_test_score', 'params', 'mean_test_score', 'std_test_score']]
    xgb_results['algoritmo'] = 'XGBoost'
    xgb_results = xgb_results[['algoritmo', 'rank_test_score', 'params', 'mean_test_score', 'std_test_score']]

    # Unificar resultados de modelos
    resultados = pd.concat([
        logit_results,
        dt_results,
        rf_results,
        ada_results,
        xgb_results
    ], ignore_index=True)

    # Ordenar descendentemente los resultados
    resultados_sorted = resultados.sort_values(by='mean_test_score', ascending=False)

    # Modelo Final
    mejor_modelo = resultados_sorted.iloc[0]
    mejor_algoritmo = mejor_modelo['algoritmo']
    mejores_parametros = mejor_modelo['params']

    # Crear el modelo con los mejores parámetros
    if mejor_algoritmo == 'Regresión Logística':
        modelo_final = LogisticRegression(**mejores_parametros)
    elif mejor_algoritmo == 'Decision Tree':
        modelo_final = DecisionTreeClassifier(**mejores_parametros, random_state=2025)
    elif mejor_algoritmo == 'Random Forest':
        modelo_final = RandomForestClassifier(**mejores_parametros, random_state=2025)
    elif mejor_algoritmo == 'AdaBoost':
        modelo_final = AdaBoostClassifier(**mejores_parametros, random_state=2025)
    elif mejor_algoritmo == 'XGBoost':
        modelo_final = xgb.XGBClassifier(**mejores_parametros, eval_metric='logloss', random_state=2025)
    else:
        raise ValueError("No se reconoce el algoritmo del mejor modelo.")

    # Entrenar el modelo con los mejores parámetros
    modelo_final.fit(X_train_scaled, y_train_encoded)

    # Guardar modelo en un archivo pkl
    joblib.dump(modelo_final, '../models/modelo_mejor.pkl')
    #print(f"Modelo guardado: {mejor_algoritmo} con parámetros {mejores_parametros}")
