import pandas as pd

from src.utils import cargar_datos_parquet


def guardar_df_parquet(df: pd.DataFrame, ruta_salida: str) -> None:
    """
    Guarda un DataFrame en formato Parquet sin incluir el índice.
    """
    df.reset_index(drop=True).to_parquet(ruta_salida, engine="fastparquet")
    print(f"Archivo guardado en: {ruta_salida}")


def main():
    ruta = r"C:\Users\usuario\EmergenciasHospitalarias\data\pacientes_procesado.parquet"
    datos = cargar_datos_parquet(ruta)

    columnas_necesarias = [
        "Hospital",
        "Fecha",
        "Turno",
        "Nivel de triaje",
        "Edad_media",
        "Porcentaje_hombres",
        "Pacientes_por_triaje",
        "Pacientes",
        "Festivo",
        "Día de la semana",
        "Estación del Año",
        "Mes del año",
    ]

    datos = datos[columnas_necesarias].dropna()

    datos_resumen = datos.groupby(["Hospital", "Fecha", "Turno"], as_index=False).agg(
        {
            "Edad_media": "mean",
            "Porcentaje_hombres": "mean",
            "Pacientes_por_triaje": "sum",
            "Pacientes": "max",
            "Festivo": "first",
            "Día de la semana": "first",
            "Estación del Año": "first",
            "Mes del año": "first",
        }
    )

    # Rolling mean en resumen
    columnas_rolling_resumen = [
        "Edad_media",
        "Porcentaje_hombres",
        "Pacientes_por_triaje",
        "Pacientes",
    ]

    datos_resumen = datos_resumen.sort_values(["Hospital", "Fecha", "Turno"])
    # Añadir rolling específico de Pacientes a 3 y 9 turnos
    for ventana in [3, 9]:
        datos_resumen[f"Rolling_Pacientes_{ventana}"] = datos_resumen.groupby(
            "Hospital"
        )["Pacientes"].transform(
            lambda x: x.rolling(window=ventana, min_periods=ventana).mean()
        )

    # Luego los dummies como siempre
    datos_resumen = pd.get_dummies(
        datos_resumen, columns=["Día de la semana"], prefix="Dia"
    )

    # Agrupación triaje
    datos_triaje = datos.groupby(
        ["Hospital", "Fecha", "Turno", "Nivel de triaje"], as_index=False
    ).agg(
        {
            "Edad_media": "mean",
            "Porcentaje_hombres": "mean",
            "Pacientes_por_triaje": "sum",
            "Festivo": "first",
            "Día de la semana": "first",
            "Estación del Año": "first",
            "Mes del año": "first",
        }
    )

    datos_triaje = datos_triaje.rename(columns={"Pacientes_por_triaje": "Pacientes"})

    # Rolling mean en triaje
    columnas_rolling_triaje = [
        "Edad_media",
        "Porcentaje_hombres",
        "Pacientes",
    ]

    datos_triaje = datos_triaje.sort_values(
        ["Hospital", "Nivel de triaje", "Fecha", "Turno"]
    )

    # Añadir rolling específico de Pacientes a 3 y 9 turnos para triaje
    for ventana in [3, 9]:
        datos_triaje[f"Rolling_Pacientes_{ventana}"] = datos_triaje.groupby(
            ["Hospital", "Nivel de triaje"]
        )["Pacientes"].transform(
            lambda x: x.rolling(window=ventana, min_periods=ventana).mean()
        )
    datos_triaje = pd.get_dummies(
        datos_triaje, columns=["Día de la semana"], prefix="Dia"
    )

    pd.set_option("display.max_columns", None)

    print("\nResumen general por día y turno")
    print(datos_resumen.head(15))
    guardar_df_parquet(
        datos_resumen,
        r"C:\Users\usuario\EmergenciasHospitalarias\data\datos_resumen.parquet",
    )

    print("\nResumen por día, turno y triaje")
    print(datos_triaje.head())
    guardar_df_parquet(
        datos_triaje,
        r"C:\Users\usuario\EmergenciasHospitalarias\data\datos_triaje.parquet",
    )


if __name__ == "__main__":
    main()
