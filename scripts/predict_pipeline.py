"""
    Este script contiene realiza las predicciones utilizando el mejor modelo
"""
import configparser
import os
import pickle
from datetime import datetime

import pandas as pd
import mlflow

def predict_pipeline():
    # Ruta del proyecto
    project_path = os.path.dirname(os.getcwd())

    # Ruta al archivo del pipeline entrenado
    artifacts_path = os.path.join(project_path, 'artifacts')
    fitted_pipeline_path = os.path.join(artifacts_path, 'fitted_pipeline.pkl')

    if not os.path.exists(fitted_pipeline_path):
        raise FileNotFoundError(f"No se encontró el archivo del pipeline en '{fitted_pipeline_path}'.")

    # Cargar el pipeline entrenado
    with open(fitted_pipeline_path, 'rb') as f:
        fitted_pipeline = pickle.load(f)

    # Leer los datos de prueba
    data_path = os.path.join(project_path, "data", "raw", "loan_data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontraron datos de prueba en '{data_path}'.")

    data = pd.read_csv(data_path)

    # Leer configuración
    config = configparser.ConfigParser()
    config.read(os.path.join(project_path, "pipeline.cfg"))

    target_var = config.get('GENERAL', 'TARGET')

    if target_var not in data.columns:
        raise ValueError(f"La variable objetivo '{target_var}' no está presente en los datos.")

    X_test = data.drop(columns=[target_var])
    y_test = data[target_var]

    # Realizar predicciones
    y_pred = fitted_pipeline.predict(X_test)

    # Crear un DataFrame con las predicciones
    predictions_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })

    # Crear la ruta para guardar el archivo de predicciones
    predictions_dir = os.path.join(project_path, "data", "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    # Generar el nombre del archivo basado en la fecha y hora
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    predictions_file = os.path.join(predictions_dir, f"{timestamp}.csv")

    # Guardar las predicciones en un archivo .csv
    predictions_df.to_csv(predictions_file, index=False)

    # Validar que el archivo fue creado correctamente
    if not os.path.exists(predictions_file):
        raise FileNotFoundError(f"No se pudo crear el archivo de predicciones en '{predictions_file}'.")

    print(f"Predicciones guardadas en: {predictions_file}")

    # Registrar las predicciones como artefacto en MLflow
    with mlflow.start_run(run_name="Predictions"):
        mlflow.log_artifact(predictions_file, artifact_path="predictions")
        print(f"Archivo de predicciones registrado en MLflow.")

predict_pipeline()
