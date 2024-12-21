"""
    Este script crea el archivo pipeline.pkl
"""
import configparser
import os
import pickle

import pandas as pd
from sklearn.pipeline import Pipeline
from feature_engine.encoding import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def create_features_pipeline():
    project_path = os.path.dirname(os.getcwd())

    data_path = os.path.join(project_path, "data", "raw", "loan_data.csv")
    data = pd.read_csv(data_path)

    config = configparser.ConfigParser()
    config.read(os.path.join(project_path, "pipeline.cfg"))

    float_to_category = config.get('CONVERSION', 'FLOAT_TO_CATEGORY').split(',')
    float_to_category = [col.strip() for col in float_to_category]

    for col in float_to_category:
        if col in data.columns:
            data[col] = data[col].astype('category')

    ohe_vars = config.get('CATEGORICAL', 'OHE_VARS').split(',')
    ohe_vars = [var.strip() for var in ohe_vars]

    # Definir pipeline
    loan_status_predict_model = Pipeline([
        ('categorical_encoding_ohe', OneHotEncoder(variables=ohe_vars, drop_last=True)),
        ('feature_scaling', StandardScaler())
    ])

    # Guardar pipeline en un archivo
    artifacts_path = os.path.join(project_path, 'artifacts')
    os.makedirs(artifacts_path, exist_ok=True)

    with open(os.path.join(artifacts_path, 'pipeline.pkl'), 'wb') as f:
        pickle.dump(loan_status_predict_model, f)

    print("Pipeline creado y guardado correctamente.")

create_features_pipeline()
