import pandas as pd


def cargar_datos_csv(ruta):
    return pd.read_csv(ruta, sep=";")


def cargar_datos_parquet(ruta):
    return pd.read_parquet(ruta, engine="fastparquet")
