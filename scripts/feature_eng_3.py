"""
    Este script contiene la función para realizar el proceso de ingeniería
    de características
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def feature_eng():
    data = pd.read_csv("../data/raw/loan_data.csv")

    # Conversión de Variables
    # Variable objetivo loan_status
    data['loan_status'] = data['loan_status'].astype('category')

    # Variables predictoras
    col_categoricas = ['person_gender', 'person_education',
                       'person_home_ownership', 'loan_intent',
                       'previous_loan_defaults_on_file']

    for col in col_categoricas:
        data[col] = data[col].astype('category')
        print(f'{col}: {data[col].dtype}')

    # Clasificar variables numéricas
    col_numericas = data.select_dtypes(include = ['int64',
                                                  'float64']).columns.tolist()

    var_continuas = []
    var_discretas = []

    for col in col_numericas:
        n_unicos = data[col].nunique()
        if n_unicos < 10:
            var_discretas.append(col)
        elif n_unicos < 100:
            var_discretas.append(col)
        else:
            var_continuas.append(col)

    le = LabelEncoder()
    data['loan_status'] = le.fit_transform(data['loan_status'])

    data_encoded = pd.get_dummies(data, columns = ['person_gender','person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file'], drop_first=True)

    scaler = StandardScaler()
    data_continuas = data[var_continuas]
    data_continuas_scaled = scaler.fit_transform(data_continuas)
    data[var_continuas] = data_continuas_scaled

    # Tratamiento de Outliers
    per_bajo = 0.05
    per_alto = 0.95

    for col in var_continuas:
        limite_inf = data[col].quantile(per_bajo)
        limite_sup = data[col].quantile(per_alto)
        data[col] = data[col].clip(lower = limite_inf, upper = limite_sup)

    # Exportr archivo CSV
    data_encoded.to_csv("../data/processed/loan_data_prepared.csv", index = False)
    print("Archivo CSV creado")
