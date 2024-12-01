"""
    Este script contiene la función para realizar un análisis estadístico
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def eda():
    """
        Esta función incluye un análisis estadístico descriptivo sobre las 
        variables e incluye gráficas sobre relaciones y distribución de 
        las variables
    """
    dataset = pd.read_csv("../data/raw/loan_data.csv")

    #Estadísticas descriptivas
    #dataset.describe().T

    # Valores Nulos
    dataset.isnull().mean()

    # Obtener columnas numéricas
    col_numericas = dataset.select_dtypes(include = ['int64', 'float64'])

    # Matriz de Correlación
    plt.figure(figsize=(10, 8))
    corr = col_numericas.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matriz de correlación")
    plt.show()

    # Gráficas Variables Numéricas
    for col in col_numericas:
        plt.figure(figsize = (6, 4))
        sns.histplot(dataset[col], kde = True, bins=30)
        plt.title(f"Distribución {col}")
        plt.xlabel(col)
        plt.ylabel("Frecuencia")
        plt.show()

    # Boxplots
    # Loan Amount y Person Gender
    plt.figure(figsize=(12, 10))
    sns.boxplot(data=dataset, x='person_gender', y='loan_amnt',
                palette="viridis")
    plt.title("Distribución de Loan Amount por Género")
    plt.xlabel("Género")
    plt.ylabel("Loan Amount")
    plt.show()

    # Ingresos y Nivel Educativo
    plt.figure(figsize=(12, 10))
    sns.boxplot(data=dataset, x='person_education', y='person_income',
                palette="viridis")
    plt.title("Distribución de Ingresos por Nivel Educativo")
    plt.xlabel("Nivel Educativo")
    plt.ylabel("Ingresos")
    plt.show()

    # Tipo de Propiedad y Edad
    plt.figure(figsize=(12, 10))
    sns.boxplot(data=dataset, x='person_home_ownership', y='person_age',
                palette="viridis")
    plt.title("Distribución de Tipos de Propiedad por Edad")
    plt.xlabel("Tipo de Propiedad")
    plt.ylabel("Edad")
    plt.show()

    # Monto de Crédito y Propósito de Crédito
    plt.figure(figsize=(12, 10))
    sns.boxplot(data=dataset, x='loan_intent', y='loan_amnt',
                palette="viridis")
    plt.title("Distribución de Monto de Crédito por Propósito del Crédito")
    plt.xlabel("Propósito del Crédito")
    plt.ylabel("Monto del Crédito")
    plt.show()

    # Gráfico de Barras Para Variables Categóricas
    col_categoricas = dataset.select_dtypes(include = ['object']).columns

    # %%
    for col in col_categoricas:
        plt.figure(figsize = (6, 4))
        sns.countplot(data = dataset, x = col, palette = 'viridis')
        plt.title(f"Frecuencia {col}")
        plt.xlabel(col)
        plt.ylabel("Frecuencia")
        plt.show()
