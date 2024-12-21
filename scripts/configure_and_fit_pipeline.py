"""
    Este script crea el archivo  con los resultados del modelo más preciso
"""
import configparser
import os
import pickle
import mlflow

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import warnings

def configure_and_fit_pipeline():

    warnings.simplefilter("ignore", category=FutureWarning)
    warnings.simplefilter("ignore", category=UserWarning)

    project_path = os.path.dirname(os.getcwd())

    data_path = os.path.join(project_path, "data", "raw", "loan_data.csv")
    data = pd.read_csv(data_path)

    config = configparser.ConfigParser()
    config.read(os.path.join(project_path, "pipeline.cfg"))

    target_var = config.get('GENERAL', 'TARGET')
    if target_var not in data.columns:
        raise ValueError(f"La variable objetivo '{target_var}' no está presente en los datos.")

    X = data.drop(columns=[target_var])
    y = data[target_var]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Cargar pipeline base
    artifacts_path = os.path.join(project_path, 'artifacts')
    pipeline_path = os.path.join(artifacts_path, 'pipeline.pkl')

    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"No se encontró el archivo del pipeline en '{pipeline_path}'.")

    with open(pipeline_path, 'rb') as f:
        base_pipeline = pickle.load(f)

    # Modelos
    models = {
        'LogisticRegression': LogisticRegression(max_iter=500, random_state=42),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42),
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
        'AdaBoostClassifier': AdaBoostClassifier(n_estimators=100, random_state=42),
        'XGBClassifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    # Configurar MLflow
    mlflow.set_experiment("Loan Default Prediction")
    best_model = None
    best_score = -float('inf')
    metric = accuracy_score

    print("Evaluando modelos...")
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            # Crear un pipeline temporal con el modelo
            temp_pipeline = Pipeline(steps=[
                ('preprocessing', base_pipeline),
                ('classifier', model)
            ])

            # Validación cruzada
            scores = cross_val_score(temp_pipeline, X_train, y_train, cv=5, scoring='accuracy')
            avg_score = scores.mean()

            print(f"{model_name}: {avg_score:.4f}")

            # Registrar parámetros y métricas en MLflow
            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("cv_accuracy", avg_score)

            # Actualizar el mejor modelo si supera al actual
            if avg_score > best_score:
                best_model = model
                best_score = avg_score

    print(f"Mejor modelo: {best_model.__class__.__name__} con puntuación: {best_score:.4f}")

    # Crear el pipeline final con el mejor modelo
    final_pipeline = Pipeline(steps=[
        ('preprocessing', base_pipeline),
        ('classifier', best_model)
    ])

    # Ajustar el pipeline final a los datos de entrenamiento
    final_pipeline.fit(X_train, y_train)

    # Evaluar el pipeline en el conjunto de prueba
    y_pred = final_pipeline.predict(X_test)
    test_score = metric(y_test, y_pred)
    print(f"Puntuación en el conjunto de prueba: {test_score:.4f}")

    # Registrar el pipeline final en MLflow
    with mlflow.start_run(run_name="Final Pipeline") as run:
        mlflow.log_metric("test_accuracy", test_score)

        # Guardar el modelo en MLflow
        mlflow.sklearn.log_model(final_pipeline, "model")

        # Guardar el pipeline ajustado como artefacto local
        fitted_pipeline_path = os.path.join(artifacts_path, 'fitted_pipeline.pkl')
        with open(fitted_pipeline_path, 'wb') as f:
            pickle.dump(final_pipeline, f)

        mlflow.log_artifact(fitted_pipeline_path, artifact_path="artifacts")
        print(f"Pipeline final ajustado y guardado en: {fitted_pipeline_path}")
    
    mlflow.end_run()

configure_and_fit_pipeline()
