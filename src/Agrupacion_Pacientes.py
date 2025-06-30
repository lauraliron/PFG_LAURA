import pandas as pd


def cargar_datos(ruta):
    return pd.read_parquet(ruta, engine="fastparquet")


def contar_pacientes_por_hospital_dia_turno(
    hospital_triage_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Añade una columna 'Pacientes' que cuenta el número de pacientes por hospital, día y turno.
    El contador se reinicia cada vez que cambia cualquiera de estos tres elementos.

    Args:
        hospital_triage_data (pd.DataFrame): DataFrame con los datos hospitalarios. Debe contener las columnas 'Hospital', 'Fecha de atención' y 'Turno'.

    Returns:
        pd.DataFrame: DataFrame con la nueva columna 'Pacientes' añadida.
    """
    hospital_triage_data["Fecha"] = pd.to_datetime(
        hospital_triage_data["Fecha de atención"]
    ).dt.date
    hospital_triage_data = hospital_triage_data.sort_values(
        by=["Hospital", "Fecha", "Turno"]
    )
    hospital_triage_data["Pacientes"] = (
        hospital_triage_data.groupby(["Hospital", "Fecha", "Turno"]).cumcount() + 1
    )
    return hospital_triage_data


def calcular_media_edad_por_hospital_dia_turno(
    hospital_triage_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula la media de edad por hospital, día y turno y la asigna a cada fila correspondiente.

    Args:
        hospital_triage_data (pd.DataFrame): DataFrame con los datos hospitalarios. Debe contener las columnas 'Hospital', 'Fecha de atención', 'Turno' y 'Edad'.

    Returns:
        pd.DataFrame: DataFrame con una nueva columna 'Edad_media' añadida por grupo.
    """
    hospital_triage_data["Fecha"] = pd.to_datetime(
        hospital_triage_data["Fecha de atención"]
    ).dt.date
    hospital_triage_data = hospital_triage_data.sort_values(
        by=["Hospital", "Fecha", "Turno"]
    )

    # Calcular media por grupo y asignar a cada fila
    hospital_triage_data["Edad_media"] = hospital_triage_data.groupby(
        ["Hospital", "Fecha", "Turno"]
    )["Edad"].transform("mean")

    return hospital_triage_data


def calcular_porcentaje_hombres_por_hospital_dia_turno(
    hospital_triage_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula el porcentaje de hombres por hospital, día y turno, y lo asigna a cada fila del DataFrame.

    Args:
        hospital_triage_data (pd.DataFrame): DataFrame con los datos hospitalarios. Debe contener las columnas 'Hospital', 'Fecha de atención', 'Turno' y 'Hombre' (0=Mujer, 1=Hombre).

    Returns:
        pd.DataFrame: DataFrame con una nueva columna 'Porcentaje_hombres' añadida por grupo.
    """
    hospital_triage_data["Fecha"] = pd.to_datetime(
        hospital_triage_data["Fecha de atención"]
    ).dt.date
    hospital_triage_data = hospital_triage_data.sort_values(
        by=["Hospital", "Fecha", "Turno"]
    )

    porcentaje_hombres = hospital_triage_data.groupby(["Hospital", "Fecha", "Turno"])[
        "Hombre"
    ].transform("mean")
    hospital_triage_data["Porcentaje_hombres"] = porcentaje_hombres

    return hospital_triage_data


def contar_pacientes_por_triaje_hospital_dia_turno(
    hospital_triage_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Añade una columna 'Pacientes_por_triaje' que cuenta el número de pacientes por hospital, día, turno y nivel de triaje.
    El contador se reinicia cada vez que cambia cualquiera de estos cuatro elementos.

    Args:
        hospital_triage_data (pd.DataFrame): DataFrame con los datos hospitalarios. Debe contener las columnas 'Hospital', 'Fecha de atención', 'Turno' y 'Nivel de triaje'.

    Returns:
        pd.DataFrame: DataFrame con la nueva columna 'Pacientes_por_triaje' añadida.
    """
    hospital_triage_data["Fecha"] = pd.to_datetime(
        hospital_triage_data["Fecha de atención"]
    ).dt.date

    hospital_triage_data = hospital_triage_data.sort_values(
        by=["Hospital", "Fecha", "Turno", "Nivel de triaje"]
    )

    hospital_triage_data["Pacientes_por_triaje"] = (
        hospital_triage_data.groupby(
            ["Hospital", "Fecha", "Turno", "Nivel de triaje"]
        ).cumcount()
        + 1
    )

    return hospital_triage_data


def calcular_proporcion_rurales_por_turno(
    hospital_triage_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Añade una columna 'proporcion_rural' al DataFrame con la proporción de pacientes rurales por turno.

    Args:
        hospital_triage_data (pd.DataFrame): DataFrame con columnas 'Turno' y 'Urbano' (1 = Urbano, 0 = Rural).

    Returns:
        pd.DataFrame: Mismo DataFrame con una nueva columna 'proporcion_rural'.
    """
    total_por_turno = hospital_triage_data.groupby("Turno")["Urbano"].count()
    rurales_por_turno = (
        hospital_triage_data[hospital_triage_data["Urbano"] == 0]
        .groupby("Turno")["Urbano"]
        .count()
    )
    proporcion_rural = (rurales_por_turno / total_por_turno * 100).round(2)

    # Asignar a cada fila la proporción del turno correspondiente
    hospital_triage_data["proporcion_rural"] = hospital_triage_data["Turno"].map(
        proporcion_rural
    )

    return hospital_triage_data


def guardar_df_parquet(
    df: pd.DataFrame,
    ruta_salida: str = r"C:\Users\usuario\EmergenciasHospitalarias\data\pacientes_procesado.parquet",
) -> None:
    """
    Guarda un DataFrame en formato Parquet.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a guardar.
    ruta_salida : str, optional
        Ruta de salida para el archivo Parquet (default es 'pacientes_procesado.parquet').

    Returns
    -------
    None
    """
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    for col in ["Año", "Mes", "Día"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    df.reset_index(drop=True).to_parquet(ruta_salida, engine="fastparquet")


def main():
    ruta = (
        r"C:\Users\usuario\EmergenciasHospitalarias\data\hospital_triage_data.parquet"
    )
    pacientes = cargar_datos(ruta)
    pacientes = contar_pacientes_por_hospital_dia_turno(pacientes)
    pacientes = calcular_media_edad_por_hospital_dia_turno(pacientes)
    pacientes = calcular_porcentaje_hombres_por_hospital_dia_turno(pacientes)
    pacientes = contar_pacientes_por_triaje_hospital_dia_turno(pacientes)
    pacientes = calcular_proporcion_rurales_por_turno(pacientes)

    pd.set_option("display.max_rows", 50)
    print(
        pacientes[
            [
                "Hospital",
                "Turno",
                "Pacientes",
                "Edad_media",
                "Edad",
                "Porcentaje_hombres",
                "Pacientes_por_triaje",
                "Nivel de triaje",
                "proporcion_rural",
            ]
        ].head(50)
    )
    guardar_df_parquet(pacientes)


if __name__ == "__main__":
    main()
