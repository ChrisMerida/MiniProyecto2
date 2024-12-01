"""
    Este script contiene la función para la lectura del Dataset Raw 
"""
import pandas as pd

def carga_datos():
    """
        Lectura de Dataset
    """
    pd.read_csv("../data/raw/loan_data.csv")
