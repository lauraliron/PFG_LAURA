import calendar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
from meteostat import Hourly, Point, Stations

from src.utils import cargar_datos_csv

FESTIVOS = [
    "2021-01-01",
    "2021-01-06",
    "2021-04-01",
    "2021-04-02",
    "2021-04-23",
    "2021-05-01",
    "2021-08-16",
    "2021-10-12",
    "2021-11-01",
    "2021-12-06",
    "2021-12-08",
    "2021-12-25",
    "2022-01-01",
    "2022-01-06",
    "2022-04-14",
    "2022-04-15",
    "2022-04-23",
    "2022-05-02",
    "2022-08-15",
    "2022-10-12",
    "2022-11-01",
    "2022-12-06",
    "2022-12-08",
    "2022-12-26",
    "2023-01-02",
    "2023-01-06",
    "2023-04-06",
    "2023-04-07",
    "2023-05-01",
    "2023-07-25",
    "2023-08-15",
    "2023-10-12",
    "2023-11-01",
    "2023-12-06",
    "2023-12-08",
    "2023-12-25",
]

MESES_NOMBRES = [
    "Enero",
    "Febrero",
    "Marzo",
    "Abril",
    "Mayo",
    "Junio",
    "Julio",
    "Agosto",
    "Septiembre",
    "Octubre",
    "Noviembre",
    "Diciembre",
]

DIAS_ESPERADOS = [
    "LUNES",
    "MARTES",
    "MIÉRCOLES",
    "JUEVES",
    "VIERNES",
    "SÁBADO",
    "DOMINGO",
]


# TRANSFORMACIONES DE DATOS
def transformar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma las columnas numéricas y categóricas en el DataFrame y convierte las binarias en 0 y 1.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con los datos originales.

    Returns
    -------
    pd.DataFrame
        DataFrame con los datos transformados.
    """

    df[["Nivel de triaje", "Edad"]] = df[["Nivel de triaje", "Edad"]].apply(
        pd.to_numeric, errors="coerce"
    )

    columnas_categoricas = ["Zona Básica de Salud", "Hospital", "Área", "Provincia"]
    df[columnas_categoricas] = df[columnas_categoricas].astype("string")

    df["Sexo"] = df["Sexo"].map({"Hombre": 1, "Mujer": 0}).astype("Int64")
    df["Ámbito de procedencia"] = (
        df["Ámbito de procedencia"].map({"Urbano": 1, "Rural": 0}).astype("Int64")
    )

    datos_transformados = df.rename(
        columns={"Sexo": "Hombre", "Ámbito de procedencia": "Urbano"}
    )

    print(datos_transformados.dtypes)

    return datos_transformados


def corregir_fechas_y_horas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige y valida las fechas y horas en el DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con las columnas 'Fecha de atención' y 'Hora'.

    Returns
    -------
    pd.DataFrame
        DataFrame con las fechas y horas corregidas.
    """
    print(
        "Valores NaN en 'Fecha de atención':",
        df["Fecha de atención"].isna().sum(),
    )
    print("Valores NaN en 'Hora':", df["Hora"].isna().sum())

    print("Formatos únicos en 'Fecha de atención':")
    print(df["Fecha de atención"].dropna().astype(str).str[:10].value_counts().head(10))

    print("Valores problemáticos en 'Hora':")
    print(
        df[~df["Hora"].astype(str).str.match(r"^\d{2}:\d{2}$", na=False)][
            "Hora"
        ].unique()
    )

    # Convertir la columna 'Fecha de atención' a datetime
    df["Fecha de atención"] = pd.to_datetime(df["Fecha de atención"], errors="coerce")

    # Corregir valores incorrectos en la columna 'Hora'
    df["Hora"] = df["Hora"].astype(str)
    df.loc[df["Hora"] == ":", "Hora"] = np.nan  # Reemplazar ":" por NaN
    df.loc[~df["Hora"].str.match(r"^\d{2}:\d{2}$", na=False), "Hora"] = np.nan
    df["Hora"] = df["Hora"].fillna(
        "00:00"
    )  # Asigna "00:00" a los valores NaN restantes

    df["Hora"] = pd.to_datetime(
        df["Hora"], format="%H:%M", errors="coerce"
    ).dt.strftime("%H:%M")

    # Crear 'Fecha completa' combinando 'Fecha de atención' y 'Hora'
    df["Fecha completa"] = pd.to_datetime(
        df["Fecha de atención"].dt.strftime("%Y-%m-%d") + " " + df["Hora"],
        format="%Y-%m-%d %H:%M",
        errors="coerce",
    )

    print(
        f"Fechas inválidas detectadas después de la corrección: {df['Fecha completa'].isna().sum()}"
    )
    print(
        df[["Fecha de atención", "Día de la semana", "Hora", "Fecha completa"]].head()
    )

    return df


def procesar_componentes_temporales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae componentes de fecha y aplica transformación sinusoidal para variables temporales.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con la columna 'Fecha completa'.

    Returns
    -------
    pd.DataFrame
        DataFrame con nuevas columnas de componentes temporales y sus transformaciones sinusoidales.
    """

    df["Hora del día"] = df["Fecha completa"].dt.hour
    df["Día del año"] = df["Fecha completa"].dt.dayofyear
    df["Mes del año"] = df["Fecha completa"].dt.month

    # Aplicar transformación
    df["Hora_sin"] = np.sin(2 * np.pi * df["Hora del día"] / 24)
    df["Hora_cos"] = np.cos(2 * np.pi * df["Hora del día"] / 24)

    df["Día_sin"] = np.sin(2 * np.pi * df["Día del año"] / 365)
    df["Día_cos"] = np.cos(2 * np.pi * df["Día del año"] / 365)

    df["Mes_sin"] = np.sin(2 * np.pi * df["Mes del año"] / 12)
    df["Mes_cos"] = np.cos(2 * np.pi * df["Mes del año"] / 12)

    print(
        "Componentes temporales extraídos y transformación sinusoidal aplicada correctamente."
    )

    return df


def marcar_festivos(df: pd.DataFrame, festivos: list) -> pd.DataFrame:
    """
    Añade una columna que indica si la fecha es festiva o domingo.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con la columna 'Fecha completa'.
    festivos : list
        Lista de fechas festivas en formato datetime.

    Returns
    -------
    pd.DataFrame
        DataFrame con una nueva columna 'Festivo' (1 si es festivo/domingo, 0 si no).
    """
    festivos = pd.to_datetime(festivos)

    df["Festivo"] = df["Fecha completa"].apply(
        lambda x: 1 if (x in festivos or x.weekday() == 6) else 0
    )

    print("Festivos y domingos marcados correctamente en la columna 'Festivo'.")

    return df


def obtener_estacion(fecha: str) -> str:
    """ "
    Añade una columna que indica en que estación del año se encuentra la fecha.

    Args:
        fecha (pd.Timestamp): Fecha en formato datetime.

    Returns:
        str: Estación del año correspondiente ("Invierno", "Primavera", "Verano" u "Otoño").
    """
    if fecha is pd.NaT:
        return None  # Si la fecha es NaN, devolver None

    mes = fecha.month
    dia = fecha.day

    if (mes == 12 and dia >= 21) or (mes in [1, 2]) or (mes == 3 and dia < 20):
        return "Invierno"
    if (mes == 3 and dia >= 20) or (mes in [4, 5]) or (mes == 6 and dia < 21):
        return "Primavera"
    if (mes == 6 and dia >= 21) or (mes in [7, 8]) or (mes == 9 and dia < 23):
        return "Verano"
    return "Otoño"


def transformar_datos_completos(
    hospital_triage_data: pd.DataFrame, festivos: list
) -> pd.DataFrame:
    """
    Aplica transformaciones generales al DataFrame, incluyendo corrección de fechas,
    extracción de componentes temporales y marcaje de festivos.

    Parameters
    ----------
    hospital_triage_data : pd.DataFrame
        DataFrame con los datos hospitalarios.
    festivos : list
        Lista de fechas festivas en formato datetime.

    Returns
    -------
    pd.DataFrame
        DataFrame con todas las transformaciones aplicadas excepto coordenadas.
    """
    hospital_triage_data = transformar_datos(hospital_triage_data)
    hospital_triage_data = corregir_fechas_y_horas(hospital_triage_data)
    hospital_triage_data = procesar_componentes_temporales(hospital_triage_data)
    hospital_triage_data = marcar_festivos(hospital_triage_data, festivos)

    # Añadir estación del año
    hospital_triage_data["Estación del Año"] = hospital_triage_data[
        "Fecha completa"
    ].apply(obtener_estacion)

    return hospital_triage_data


def asignar_turno(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna un turno de atención basado en la hora de llegada del paciente.

    Turnos:
    - "Madrugada" (00:00 - 07:59)
    - "Día" (08:00 - 15:59)
    - "Noche" (16:00 - 23:59)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con la columna 'Fecha completa' en formato datetime.

    Returns
    -------
    pd.DataFrame
        DataFrame con una nueva columna 'Turno' asignada a cada paciente.
    """
    df["Turno"] = df["Fecha completa"].dt.hour.apply(
        lambda x: "Madrugada" if 0 <= x < 8 else "Día" if 8 <= x < 16 else "Noche"
    )

    print("Turnos de atención asignados correctamente.")

    return df


def obtener_coordenadas(hospital: str) -> tuple[float | None, float | None]:
    """
    Obtiene las coordenadas geográficas de un hospital en España.

    Parameters
    ----------
    hospital : str
        Nombre del hospital.

    Returns
    -------
    tuple[float | None, float | None]
        Tupla (latitud, longitud) si se encuentra la ubicación.
        Devuelve (None, None) si no se encuentra o hay un error.
    """
    geolocator = Nominatim(user_agent="hospital_locator")

    try:
        location = geolocator.geocode(hospital + ", España", timeout=10)
        if location:
            return location.latitude, location.longitude
        return None, None
    except GeocoderTimedOut:
        return None, None


def agregar_coordenadas_hospitales(hospital_triage_data: pd.DataFrame) -> pd.DataFrame:
    """
    Obtiene las coordenadas de los hospitales y las añade al DataFrame.
    Si no se pueden obtener, asigna coordenadas manuales para hospitales específicos.

    Parameters
    ----------
    hospital_triage_data : pd.DataFrame
        DataFrame con la columna "Hospital".

    Returns
    -------
    pd.DataFrame
        DataFrame con las coordenadas de los hospitales añadidas.
    """
    geolocator = Nominatim(user_agent="hospital_locator")

    hospitales_unicos = hospital_triage_data["Hospital"].dropna().unique()

    # Coordenadas de hospitales
    def obtener_coordenadas(hospital):
        try:
            location = geolocator.geocode(hospital + ", España", timeout=10)
            if location:
                return (location.latitude, location.longitude)
            return (None, None)
        except GeocoderTimedOut:
            return (None, None)

    # Obtener coordenadas de hospitales únicos
    coordenadas_hospitales = {
        hospital: obtener_coordenadas(hospital) for hospital in hospitales_unicos
    }

    df_coordenadas = pd.DataFrame.from_dict(
        coordenadas_hospitales, orient="index", columns=["Latitud", "Longitud"]
    ).reset_index()
    df_coordenadas = df_coordenadas.rename(columns={"index": "Hospital"})

    hospital_triage_data = hospital_triage_data.merge(
        df_coordenadas, on="Hospital", how="left"
    )

    coordenadas_manual = {
        "H.C.U. Valladolid": (41.655903, -4.718669),
        "H.U. Río Hortega": (41.628797, -4.711461),
    }

    # Limpiar espacios en la columna "Hospital"
    hospital_triage_data["Hospital"] = hospital_triage_data["Hospital"].str.strip()

    # Aplicar coordenadas manuales donde falten
    hospital_triage_data["Latitud"] = hospital_triage_data.apply(
        lambda row: coordenadas_manual[row["Hospital"]][0]
        if row["Hospital"] in coordenadas_manual and pd.isna(row["Latitud"])
        else row["Latitud"],
        axis=1,
    )

    hospital_triage_data["Longitud"] = hospital_triage_data.apply(
        lambda row: coordenadas_manual[row["Hospital"]][1]
        if row["Hospital"] in coordenadas_manual and pd.isna(row["Longitud"])
        else row["Longitud"],
        axis=1,
    )

    return hospital_triage_data


def contar_valores_nulos(
    df: pd.DataFrame, año: int | None = None
) -> tuple[pd.Series, int]:
    """
    Cuenta los valores NaN por columna en el DataFrame completo.
    Si se especifica un año, cuenta los valores NaN solo para ese año.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a analizar.
    año : int, optional
        Año específico para filtrar los datos. Si es None, analiza todo el DataFrame.

    Returns
    -------
    tuple[pd.Series, int]
        - Serie con el conteo de NaN por columna.
        - Total de valores NaN en el DataFrame o en el año especificado.
    """
    df_filtrado = df[df["Fecha de atención"].dt.year == año] if año else df

    na_por_columna = df_filtrado.isna().sum()
    na_totales = na_por_columna.sum()

    print(f"Valores NaN por columna para {año if año else 'todo el DataFrame'}:")
    print(na_por_columna)
    print(f"\nNúmero total de valores NaN: {na_totales}")

    return na_por_columna, na_totales


def transformar_datos_hospitalarios(
    hospital_triage_data: pd.DataFrame, festivos: list
) -> pd.DataFrame:
    """
    Aplica las transformaciones necesarias al DataFrame, incluyendo corrección de fechas,
    asignación de turnos y agregación de coordenadas.

    Parameters
    ----------
    hospital_triage_data : pd.DataFrame
        DataFrame con los datos hospitalarios.
    festivos : list
        Lista de fechas festivas en formato datetime.

    Returns
    -------
    pd.DataFrame
        DataFrame transformado con todas las modificaciones aplicadas.
    """
    hospital_triage_data = transformar_datos_completos(hospital_triage_data, festivos)
    hospital_triage_data = asignar_turno(hospital_triage_data)
    hospital_triage_data = agregar_coordenadas_hospitales(hospital_triage_data)
    return hospital_triage_data


def analizar_valores_nulos(hospital_triage_data: pd.DataFrame) -> None:
    """
    Realiza un análisis de valores NaN en el DataFrame, verifica coincidencias entre columnas
    y analiza la distribución de áreas por provincia.

    Parameters
    ----------
    hospital_triage_data : pd.DataFrame
        DataFrame con los datos de triaje hospitalario.

    Returns
    -------
    None
    """
    contar_valores_nulos(hospital_triage_data)

    verificar_coincidencia_nulos(hospital_triage_data, "Zona Básica de Salud", "Urbano")
    verificar_coincidencia_nulos(hospital_triage_data, "Edad", "Hombre")

    zona_ambito_hospital = hospital_triage_data.groupby(["Área", "Hospital"])[
        ["Zona Básica de Salud", "Urbano"]
    ].nunique()

    print("Niveles únicos por hospital:\n", zona_ambito_hospital)

    # Análisis de provincias y áreas
    provincias_analisis = (
        hospital_triage_data.groupby("Provincia")
        .agg({"Área": lambda x: x.unique()})
        .reset_index()
    )

    provincias_analisis.columns = ["Provincia", "Áreas"]

    print("Análisis de provincias y áreas:\n", provincias_analisis.head())


def verificar_coincidencia_nulos(
    df: pd.DataFrame, columna1: str, columna2: str
) -> None:
    """
    Comprueba si los valores NaN en dos columnas coinciden exactamente en el DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame que contiene los datos.
    columna1 : str
        Nombre de la primera columna.
    columna2 : str
        Nombre de la segunda columna.

    Returns
    -------
    None
    """
    coinciden = (df[columna1].isna() == df[columna2].isna()).all()
    print(f"¿Coinciden los NaN entre '{columna1}' y '{columna2}'?", coinciden)


def analizar_duplicados_y_nans(hospital_triage_data: pd.DataFrame) -> pd.Series:
    """
    Analiza los duplicados en el DataFrame y calcula la cantidad de valores NaN en "Nivel de triaje" por mes.

    Parameters
    ----------
    hospital_triage_data : pd.DataFrame
        DataFrame con los datos de triaje hospitalario.

    Returns
    -------
    pd.Series
        Serie con el conteo de valores NaN en "Nivel de triaje" agrupados por mes.
    """
    # Identificar duplicados
    duplicados = hospital_triage_data.duplicated()
    total_duplicados = duplicados.sum()

    print(f"Total de filas completamente duplicadas: {total_duplicados}")

    if total_duplicados > 0:
        print("Filas duplicadas:")
        print(hospital_triage_data[duplicados])
    else:
        print("No hay filas completamente duplicadas en el DataFrame.")

    # Crear la columna Año_Mes para agrupaciones mensuales
    hospital_triage_data["Año_Mes"] = hospital_triage_data[
        "Fecha completa"
    ].dt.to_period("M")

    # Contar NaNs en "Nivel de triaje" por mes
    return (
        hospital_triage_data[hospital_triage_data["Nivel de triaje"].isna()]
        .groupby("Año_Mes")
        .size()
    )


def verificar_dias_invalidos(df):
    problemas = []
    for año in df["Año"].unique():
        for mes in df["Mes"].unique():
            # Obtener último día del mes para el año dado
            ultimo_dia = calendar.monthrange(año, mes)[1]
            # Filtrar filas donde el día excede el último día del mes
            dias_invalidos = df[
                (df["Año"] == año) & (df["Mes"] == mes) & (df["Día"] > ultimo_dia)
            ]
            if not dias_invalidos.empty:
                problemas.append(dias_invalidos)
    return problemas


def obtener_hospitales_por_año(
    hospital_triage_data: pd.DataFrame, año: int
) -> tuple[list, list]:
    """
    Obtiene la lista de hospitales únicos en un año específico y en toda la base de datos.

    Parameters
    ----------
    hospital_triage_data : pd.DataFrame
        Base de datos con la información hospitalaria.
    año : int
        Año para filtrar los hospitales.

    Returns
    -------
    tuple[list, list]
        - Lista de hospitales únicos en el año especificado.
        - Lista de todos los hospitales en la base de datos.
    """
    hospitales_año = sorted(
        hospital_triage_data[hospital_triage_data["Fecha de atención"].dt.year == año][
            "Hospital"
        ].unique()
    )
    hospitales_totales = sorted(hospital_triage_data["Hospital"].unique())

    return hospitales_año, hospitales_totales


def analizar_hospitales_y_nulos(df: pd.DataFrame, año: int) -> None:
    """
    Muestra información sobre los hospitales en un año específico y cuenta los valores nulos.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con los datos de triaje hospitalario.
    año : int
        Año para analizar los hospitales y valores nulos.
    """
    hospitales_año, hospitales_totales = obtener_hospitales_por_año(df, año)

    print(f"Hospitales en {año}:")
    print("\n".join(hospitales_año))
    print(f"\nTotal de hospitales en {año}: {len(hospitales_año)}\n")

    print("Hospitales totales:")
    print("\n".join(hospitales_totales))
    print(f"\nTotal de hospitales totales: {len(hospitales_totales)}")

    contar_valores_nulos(df, año)


def analizar_edades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analiza la distribución de edades y detecta valores fuera de rango.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con la columna "Edad".

    Returns
    -------
    pd.DataFrame
        DataFrame con los registros que tienen edades fuera del rango [0, 117].
    """
    print("Descripción estadística de la columna 'Edad':")
    print(df["Edad"].describe())

    # Filtrar edades fuera de rango
    edades_fuera_de_rango = df[(df["Edad"] < 0) | (df["Edad"] > 117)]

    if not edades_fuera_de_rango.empty:
        print("\n Edades fuera de rango encontradas:")
        print(edades_fuera_de_rango.head())
    else:
        print("\n No se encontraron edades fuera de rango.")

    return edades_fuera_de_rango


def verificar_dias_semana(df: pd.DataFrame, dias_esperados: list) -> pd.DataFrame:
    """
    Verifica si los días de la semana en el DataFrame son válidos y detecta valores incorrectos.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con la columna "Día de la semana".
    dias_esperados : list
        Lista con los valores esperados para los días de la semana.

    Returns
    -------
    pd.DataFrame
        DataFrame con las filas que contienen días no válidos.
    """
    dias_unicos = df["Día de la semana"].unique()
    print("Días únicos en la columna:", dias_unicos)

    # Identificar días que no están en la lista esperada
    dias_no_validos = [dia for dia in dias_unicos if dia not in dias_esperados]

    if dias_no_validos:
        print("\nDías no válidos encontrados:", dias_no_validos)
    else:
        print("\nTodos los días están escritos de forma adecuada.")

    # Filtrar las filas con días no válidos
    filas_dias_no_validos = df[df["Día de la semana"].isin(dias_no_validos)]

    if not filas_dias_no_validos.empty:
        print("\n Filas con días no válidos:")
        print(filas_dias_no_validos.head())
    else:
        print("\n No hay filas con días no válidos.")

    return filas_dias_no_validos


def validar_fechas(df: pd.DataFrame) -> None:
    """
    Verifica la validez de las fechas en el DataFrame, detectando valores nulos y días fuera de rango.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con la columna 'Fecha de atención'.

    Returns
    -------
    None
    """
    # Detectar fechas nulas
    fechas_invalidas = df[df["Fecha de atención"].isna()]

    if fechas_invalidas.empty:
        print(" Todas las fechas son válidas.")
    else:
        print(" Fechas no válidas encontradas:")
        print(fechas_invalidas)

    # Extraer componentes de la fecha
    df["Día"] = df["Fecha de atención"].dt.day
    df["Mes"] = df["Fecha de atención"].dt.month

    # Verificar días fuera de rango
    problemas_dias = verificar_dias_invalidos(df)

    if problemas_dias:
        print("\nSe encontraron fechas con días fuera de rango:")
        for problema in problemas_dias:
            print(problema)
    else:
        print("\n No se encontraron días fuera de rango.")


def filtrar_nivel_triaje_nan(hospital_triage_data: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra las filas donde la columna "Nivel de triaje" es NaN y muestra valores únicos de ciertas columnas.

    Parameters
    ----------
    hospital_triage_data : pd.DataFrame
        DataFrame con los datos del triaje hospitalario.

    Returns
    -------
    pd.DataFrame
        DataFrame con las filas donde "Nivel de triaje" es NaN.
    """
    df_nivel_triaje_nan = hospital_triage_data[
        hospital_triage_data["Nivel de triaje"].isna()
    ]

    pd.set_option(
        "display.expand_frame_repr", False
    )  # Evita que se divida en varias líneas
    pd.set_option(
        "display.max_columns", None
    )  # Muestra todas las columnas sin truncarlas

    print(df_nivel_triaje_nan["Año"].unique())
    print(df_nivel_triaje_nan["Urbano"].unique())
    print(df_nivel_triaje_nan["Zona Básica de Salud"].unique())

    return df_nivel_triaje_nan


def analizar_pacientes_por_mes(hospital_triage_data: pd.DataFrame):
    """
    Analiza el número de pacientes por mes, normaliza los valores a un mes de 30 días
    y encuentra los meses con más y menos pacientes por año.

    Parameters
    ----------
    hospital_triage_data : pd.DataFrame
        DataFrame con los datos del triaje hospitalario.

    Returns
    -------
    tuple
        Contiene los siguientes DataFrames:
        - resultados: Meses con más y menos pacientes por año.
        - meses_ordenados: Pacientes normalizados ordenados de mayor a menor.
        - pacientes_por_dia_mes: Pacientes por día, mes y año.
    """
    # Contar el número de pacientes por mes en cada año
    pacientes_por_mes = hospital_triage_data.groupby(["Año", "Mes"]).size()

    # Obtener el número de días únicos en cada mes del conjunto de datos
    dias_en_cada_mes = hospital_triage_data.groupby(["Año", "Mes"])[
        "Fecha de atención"
    ].nunique()

    # Ajustar los valores como si cada mes tuviera 30 días
    pacientes_normalizados = (pacientes_por_mes / dias_en_cada_mes) * 30

    # Encontrar el mes con más y menos pacientes por cada año
    mes_max_pacientes = pacientes_normalizados.groupby(level=0).idxmax()
    mes_min_pacientes = pacientes_normalizados.groupby(level=0).idxmin()

    # Obtener la cantidad de pacientes normalizados para cada mes máximo y mínimo
    cantidad_max_pacientes = pacientes_normalizados.loc[
        mes_max_pacientes.to_numpy()
    ].to_numpy()
    cantidad_min_pacientes = pacientes_normalizados.loc[
        mes_min_pacientes.to_numpy()
    ].to_numpy()

    # Crear un DataFrame con los resultados y las cantidades correspondientes
    resultados = pd.DataFrame(
        {
            "Año": mes_max_pacientes.index,
            "Mes con más pacientes": [
                m[1] for m in mes_max_pacientes.to_numpy()
            ],  # Extraer solo el mes
            "Cantidad más pacientes (Normalizada)": cantidad_max_pacientes,
            "Mes con menos pacientes": [
                m[1] for m in mes_min_pacientes.to_numpy()
            ],  # Extraer solo el mes
            "Cantidad menos pacientes (Normalizada)": cantidad_min_pacientes,
        }
    )

    # Ordenar los meses con más pacientes de mayor a menor cantidad
    meses_ordenados = pacientes_normalizados.sort_values(ascending=False).reset_index()
    meses_ordenados.columns = ["Año", "Mes", "Cantidad de Pacientes (Normalizada)"]

    print(meses_ordenados)

    # Contar el número de pacientes por día, mes y año
    pacientes_por_dia_mes = hospital_triage_data.groupby(["Año", "Mes", "Día"]).size()

    return resultados, meses_ordenados, pacientes_por_dia_mes


def calcular_pacientes_normalizados(hospital_triage_data: pd.DataFrame) -> pd.Series:
    """
    Calcula el número de pacientes normalizados por mes en cada año, ajustando los valores a un mes de 30 días.

    Parameters
    ----------
    hospital_triage_data : pd.DataFrame
        DataFrame con los datos del triaje hospitalario.

    Returns
    -------
    pd.Series
        Serie con los pacientes normalizados por mes y año.
    """
    pacientes_por_mes = hospital_triage_data.groupby(["Año", "Mes"]).size()
    dias_en_cada_mes = hospital_triage_data.groupby(["Año", "Mes"])[
        "Fecha de atención"
    ].nunique()
    pacientes_normalizados = (pacientes_por_mes / dias_en_cada_mes) * 30
    return pacientes_normalizados


def obtener_meses_extremos(pacientes_normalizados: pd.Series) -> pd.DataFrame:
    """
    Encuentra el mes con más y menos pacientes normalizados por cada año.

    Parameters
    ----------
    pacientes_normalizados : pd.Series
        Serie con los pacientes normalizados por mes y año.

    Returns
    -------
    pd.DataFrame
        DataFrame con los meses de mayor y menor afluencia de pacientes por año.
    """
    mes_max_pacientes = pacientes_normalizados.groupby(level=0).idxmax()
    mes_min_pacientes = pacientes_normalizados.groupby(level=0).idxmin()

    cantidad_max_pacientes = pacientes_normalizados.loc[
        mes_max_pacientes.to_numpy()
    ].to_numpy()
    cantidad_min_pacientes = pacientes_normalizados.loc[
        mes_min_pacientes.to_numpy()
    ].to_numpy()

    resultados = pd.DataFrame(
        {
            "Año": mes_max_pacientes.index,
            "Mes con más pacientes": [
                m[1] for m in mes_max_pacientes.to_numpy()
            ],  # Extraer solo el mes
            "Cantidad más pacientes (Normalizada)": cantidad_max_pacientes,
            "Mes con menos pacientes": [
                m[1] for m in mes_min_pacientes.to_numpy()
            ],  # Extraer solo el mes
            "Cantidad menos pacientes (Normalizada)": cantidad_min_pacientes,
        }
    )
    return resultados


def contar_pacientes_por_dia(hospital_triage_data: pd.DataFrame) -> pd.Series:
    """
    Cuenta el número de pacientes por día, mes y año.

    Parameters
    ----------
    hospital_triage_data : pd.DataFrame
        DataFrame con los datos del triaje hospitalario.

    Returns
    -------
    pd.Series
        Serie con la cantidad de pacientes por día, mes y año.
    """
    return hospital_triage_data.groupby(["Año", "Mes", "Día"]).size()


#######
def graficar_nans_triaje(nan_triaje_por_mes, proporcion_nan):
    """
    Genera dos gráficos:
    1. Cantidad de valores NaN en 'Nivel de triaje' por mes.
    2. Proporción de valores NaN en 'Nivel de triaje' por mes.

    Args:

    Returns:

    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfico 1: Cantidad de NaN por mes
    nan_triaje_por_mes.plot(kind="bar", color="#123456", alpha=1, ax=axes[0])
    axes[0].set_title('Cantidad de valores faltantes en "Nivel de Triaje" por mes')
    axes[0].set_xlabel("Mes")
    axes[0].set_ylabel("Cantidad de NaN")
    axes[0].tick_params(axis="x", rotation=45)

    # Gráfico 2: Proporción de NaN por mes
    proporcion_nan.plot(kind="bar", color="#123456", alpha=0.7, ax=axes[1])
    axes[1].set_title(
        'Proporción de valores faltantes en "Nivel de Triaje" por mes (%)'
    )
    axes[1].set_xlabel("Mes")
    axes[1].set_ylabel("Proporción de NaN (%)")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return fig


def graficar_cantidad_datos_por_año(años):
    """
    Genera un gráfico de barras con la cantidad de registros en la base de datos por año.

    Args:
        años (Series): Serie de pandas con los años de los registros.

    Returns:
        fig (matplotlib.figure.Figure): Figura del gráfico generado.
    """
    conteo_por_año = años.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(conteo_por_año.index, conteo_por_año.values, color="#123456")
    ax.set_title("Cantidad de datos por año", fontsize=16)
    ax.set_xlabel("Año", fontsize=14)
    ax.set_ylabel("Cantidad de datos", fontsize=14)
    ax.set_xticks(conteo_por_año.index)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    return fig


def graficar_datos_por_semana(hospital_triage_data):
    """
    Genera un gráfico de dispersión con la cantidad de datos registrados por semana en cada año.

    Args:
        hospital_triage_data (DataFrame): Base de datos con las fechas de atención.

    Returns:
        fig (matplotlib.figure.Figure): Figura del gráfico.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    for year, color in zip(
        [2021, 2022, 2023], ["#123456", "#123456", "#123456"], strict=False
    ):
        # Filtrar los datos del año
        datos_por_semana = (
            hospital_triage_data[
                hospital_triage_data["Fecha de atención"].dt.year == year
            ]
            .groupby(hospital_triage_data["Fecha de atención"].dt.isocalendar().week)
            .size()
        )

        ax.scatter(
            datos_por_semana.index,
            datos_por_semana.values,
            label=f"Año {year}",
            color=color,
            s=50,
            alpha=0.7,
        )

    # Personalización del gráfico
    ax.set_title("Cantidad de datos por semana para cada año", fontsize=16)
    ax.set_xlabel("Semana del año", fontsize=14)
    ax.set_ylabel("Cantidad de datos", fontsize=14)
    ax.set_xticks(range(1, 54))  # Semanas de 1 a 53
    ax.legend(title="Año", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    return fig


def calcular_pacientes_por_mes(
    hospital_triage_data: pd.DataFrame,
) -> dict[int, pd.Series]:
    """
    Calcula la cantidad de pacientes atendidos por mes en los años 2021, 2022 y 2023.

    Parameters
    ----------
    hospital_triage_data : pd.DataFrame
        DataFrame con los datos de urgencias.

    Returns
    -------
    dict[int, pd.Series]
        Diccionario con claves 2021, 2022 y 2023, y valores con Series de pacientes por mes.
    """
    pacientes_por_mes_2021 = (
        hospital_triage_data[hospital_triage_data["Fecha de atención"].dt.year == 2021][
            "Fecha de atención"
        ]
        .dt.month.value_counts()
        .sort_index()
    )
    pacientes_por_mes_2022 = (
        hospital_triage_data[hospital_triage_data["Fecha de atención"].dt.year == 2022][
            "Fecha de atención"
        ]
        .dt.month.value_counts()
        .sort_index()
    )
    pacientes_por_mes_2023 = (
        hospital_triage_data[hospital_triage_data["Fecha de atención"].dt.year == 2023][
            "Fecha de atención"
        ]
        .dt.month.value_counts()
        .sort_index()
    )

    return {
        2021: pacientes_por_mes_2021,
        2022: pacientes_por_mes_2022,
        2023: pacientes_por_mes_2023,
    }


def graficar_diferencia_porcentual(
    datos_pacientes: pd.DataFrame, nombres_meses: list
) -> plt.Figure:
    """
    Genera un gráfico de líneas que muestra la diferencia porcentual de pacientes
    entre los años 2021-2022 y 2021-2023.

    Args:
        datos_pacientes (pd.DataFrame): DataFrame con las columnas 'Mes', '2021', '2022' y '2023'.
        nombres_meses (list): Lista con los nombres de los meses.

    Returns:
        fig: Figura del gráfico generado.
    """
    # Calcular diferencias porcentuales con respecto a 2021
    datos_pacientes["Dif_2021_vs_2022"] = (
        (datos_pacientes["2022"] - datos_pacientes["2021"]) / datos_pacientes["2022"]
    ) * 100
    datos_pacientes["Dif_2021_vs_2023"] = (
        (datos_pacientes["2023"] - datos_pacientes["2021"]) / datos_pacientes["2023"]
    ) * 100

    # Crear la figura
    fig, ax = plt.subplots(figsize=(12, 6))

    # Graficar la diferencia porcentual mes a mes
    ax.plot(
        datos_pacientes["Mes"],
        datos_pacientes["Dif_2021_vs_2022"],
        marker="o",
        linestyle="-",
        label="Diferencia 2021 vs 2022",
        color="#AA0000",
    )
    ax.plot(
        datos_pacientes["Mes"],
        datos_pacientes["Dif_2021_vs_2023"],
        marker="o",
        linestyle="-",
        label="Diferencia 2021 vs 2023",
        color="#008000",
    )

    # Configuración del gráfico
    ax.axvline(x=5, color="black", linestyle="--", label="Mayo")
    ax.set_title(
        "Diferencia Porcentual de Pacientes (2021 vs 2022 y 2023)", fontsize=16
    )
    ax.set_xlabel("Mes", fontsize=14)
    ax.set_ylabel("Diferencia Porcentual (%)", fontsize=14)
    ax.set_xticks(np.arange(1, 13))
    ax.set_xticklabels(nombres_meses, rotation=45)  # Usando la variable externa
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def graficar_pacientes_por_mes(pacientes_por_mes, nombres_meses):
    """
    Genera un gráfico de líneas que compara la cantidad de pacientes por mes en los años 2021, 2022 y 2023.

    Args:
        pacientes_por_mes (dict): Diccionario con los datos de pacientes por mes en cada año.
        nombres_meses (list): Lista con los nombres de los meses.

    Returns:
        plt.Figure: Figura con el gráfico generado.
    """
    meses = np.arange(1, 13)
    fig, ax = plt.subplots(figsize=(12, 6))

    colores = ["#123456", "#AA0000", "#008000"]  # Colores uniformes
    for (año, datos), color in zip(pacientes_por_mes.items(), colores, strict=False):
        datos = datos.reindex(meses, fill_value=0)  #
        ax.plot(meses, datos, marker="o", linestyle="-", label=str(año), color=color)

    # Configuración del gráfico
    ax.set_title("Comparación de Pacientes por Mes", fontsize=16)
    ax.set_xlabel("Mes", fontsize=14)
    ax.set_ylabel("Número de Pacientes", fontsize=14)
    ax.set_xticks(meses)
    ax.set_xticklabels(nombres_meses, rotation=45)
    ax.legend(title="Año")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return fig


##################################################################################


def cargar_y_transformar_datos(ruta_csv: str, FESTIVOS: pd.DataFrame) -> pd.DataFrame:
    """
    Carga los datos desde un archivo CSV y aplica la transformación inicial específica del hospital.

    Args:
        ruta_csv (str): Ruta al archivo CSV.
        FESTIVOS (pd.DataFrame): DataFrame con las fechas festivas.

    Returns:
        pd.DataFrame: DataFrame con los datos transformados.
    """
    hospital_triage_data = cargar_datos_csv(ruta_csv)
    return transformar_datos_hospitalarios(hospital_triage_data, FESTIVOS)


def analizar_nulos_y_duplicados(
    hospital_triage_data: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """
    Analiza los valores nulos y duplicados en el DataFrame.

    Args:
        hospital_triage_data (pd.DataFrame): DataFrame con los datos hospitalarios.

    Returns:
        tuple[pd.Series, pd.Series]: Series con los nulos por mes y su proporción.
    """
    analizar_valores_nulos(hospital_triage_data)
    nan_triaje_por_mes = analizar_duplicados_y_nans(hospital_triage_data)
    total_por_mes = hospital_triage_data.groupby("Año_Mes").size()
    proporcion_nan = (nan_triaje_por_mes / total_por_mes) * 100
    return nan_triaje_por_mes, proporcion_nan


def procesar_y_guardar_graficos_nans(
    nan_triaje_por_mes: pd.Series, proporcion_nan: pd.Series
) -> None:
    """
    Genera y guarda los gráficos de análisis de valores nulos por triaje.

    Args:
        nan_triaje_por_mes (pd.Series): Serie con nulos por mes.
        proporcion_nan (pd.Series): Serie con proporción de nulos.

    Returns:
        None
    """
    fig = graficar_nans_triaje(nan_triaje_por_mes, proporcion_nan)
    fig.savefig("img/nan_triaje_por_mes.pdf", bbox_inches="tight")
    fig.savefig("img/nan_triaje_por_mes.png", dpi=300, bbox_inches="tight")
    plt.show()


def generar_graficos_actividad(hospital_triage_data: pd.DataFrame) -> None:
    """
    Genera gráficos de pacientes por año y por semana.

    Args:
        hospital_triage_data (pd.DataFrame): DataFrame con los datos hospitalarios.

    Returns:
        None
    """

    fig = graficar_cantidad_datos_por_año(hospital_triage_data["Año"])
    fig.savefig("img/pacientes_por_año.pdf", bbox_inches="tight")
    fig.savefig("img/pacientes_por_año.png", dpi=300, bbox_inches="tight")
    plt.show()
    fig = graficar_datos_por_semana(hospital_triage_data)
    fig.savefig("img/pacientes_por_semana.pdf", bbox_inches="tight")
    fig.savefig("img/pacientes_por_semana.png", dpi=300, bbox_inches="tight")
    plt.show()


def procesar_pacientes_por_mes(hospital_triage_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la cantidad de pacientes por mes y genera el gráfico correspondiente.

    Args:
        hospital_triage_data (pd.DataFrame): DataFrame con los datos hospitalarios.

    Returns:
        pd.DataFrame: Tabla de pacientes por mes.
    """
    pacientes_por_mes = calcular_pacientes_por_mes(hospital_triage_data)
    fig = graficar_pacientes_por_mes(pacientes_por_mes, MESES_NOMBRES)
    fig.savefig("img/pacientes_por_mes.pdf", bbox_inches="tight")
    fig.savefig("img/pacientes_por_mes.png", dpi=300, bbox_inches="tight")
    plt.show()
    return pacientes_por_mes


def generar_comparativa_anual(pacientes_por_mes: pd.DataFrame) -> None:
    """
    Genera una comparativa porcentual de pacientes por mes entre los años disponibles.

    Args:
        pacientes_por_mes (pd.DataFrame): Tabla de pacientes por mes.

    Returns:
        None
    """
    datos_pacientes = pd.DataFrame(
        {
            "Mes": np.arange(1, 13),
            "2021": pacientes_por_mes[2021],
            "2022": pacientes_por_mes[2022],
            "2023": pacientes_por_mes[2023],
        }
    )
    # fig = graficar_diferencia_porcentual(datos_pacientes, MESES_NOMBRES)
    # fig.savefig("img/diferencia_porcentual.pdf", bbox_inches="tight")
    # fig.savefig("img/diferencia_porcentual.png", dpi=300, bbox_inches="tight")
    # plt.show()


def analisis_calidad_datos(
    hospital_triage_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica análisis de calidad a edad, día de la semana y fechas.

    Args:
        hospital_triage_data (pd.DataFrame): DataFrame con los datos hospitalarios.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Edades fuera de rango y días no válidos.
    """
    analizar_hospitales_y_nulos(hospital_triage_data, 2021)
    edades_fuera_rango_df = analizar_edades(hospital_triage_data)
    dias_no_validos_df = verificar_dias_semana(hospital_triage_data, DIAS_ESPERADOS)
    validar_fechas(hospital_triage_data)
    print(hospital_triage_data["Nivel de triaje"].unique())
    print(hospital_triage_data["Nivel de triaje"].value_counts())
    return edades_fuera_rango_df, dias_no_validos_df


def limpieza_final(hospital_triage_data: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina registros inválidos según año, edad y nivel de triaje.

    Args:
        hospital_triage_data (pd.DataFrame): DataFrame con los datos hospitalarios.

    Returns:
        pd.DataFrame: DataFrame filtrado.
    """
    return hospital_triage_data[
        ~(
            (hospital_triage_data["Fecha de atención"].dt.year.eq(2021))
            & (hospital_triage_data["Fecha de atención"].dt.month.lt(5))
        )
        & hospital_triage_data["Edad"].between(0, 117)
    ].dropna(subset=["Nivel de triaje"])


def resumen_pacientes(hospital_triage_data: pd.DataFrame) -> None:
    """
    Muestra los días con mayor y menor número de pacientes por año.

    Args:
        hospital_triage_data (pd.DataFrame): DataFrame con los datos hospitalarios.

    Returns:
        None
    """
    pacientes_por_dia_mes = contar_pacientes_por_dia(hospital_triage_data)
    dia_mes_menos_pacientes = pacientes_por_dia_mes.groupby("Año").idxmin()
    cantidad_menos_pacientes = pacientes_por_dia_mes.groupby("Año").min()
    resultado_menos = pd.DataFrame(
        {
            "Día y Mes con menos pacientes": dia_mes_menos_pacientes,
            "Cantidad de pacientes": cantidad_menos_pacientes,
        }
    )
    dia_mes_mas_pacientes = pacientes_por_dia_mes.groupby("Año").idxmax()
    cantidad_mas_pacientes = pacientes_por_dia_mes.groupby("Año").max()
    resultado_mas = pd.DataFrame(
        {
            "Día y Mes con más pacientes": dia_mes_mas_pacientes,
            "Cantidad de pacientes": cantidad_mas_pacientes,
        }
    )
    print(resultado_menos)
    print(resultado_mas)


def get_precipitation(coords_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """
    Asocia datos horarios de precipitación a cada fila del dataset hospitalario
    según coordenadas de cada área y fecha de atención.

    Args:
        coords_df (pd.DataFrame): DataFrame con columnas 'Área', 'Latitud', 'Longitud'.
        data (pd.DataFrame): DataFrame con todas las columnas hospitalarias, incluyendo 'Área', 'Latitud', 'Longitud', 'Fecha de atención'.

    Returns:
        pd.DataFrame: DataFrame original con una nueva columna 'Precipitation'.
    """
    ls = []
    for _, row in coords_df.iterrows():
        # Buscar la estación más cercana con datos horarios
        stations = Stations()
        stations = (
            stations.nearby(row["Latitud"], row["Longitud"])
            .inventory("hourly")
            .fetch()
            .head(1)
        )
        station_lat = stations["latitude"].item()
        station_lon = stations["longitude"].item()

        # Obtener datos horarios de precipitación
        hourly_data = Hourly(Point(station_lat, station_lon)).fetch()
        prcp_series = hourly_data["prcp"].fillna(0)

        # Preparar datos horarios con índice datetime
        prcp_data = prcp_series.to_frame(name="prcp").reset_index()

        # Extraer subconjunto relevante del DataFrame original
        filtered_data = data.query(
            f"Latitud == {row['Latitud']} & Longitud == {row['Longitud']}"
        ).reset_index(drop=False)

        # Repetir cada fila 8 veces
        new_df: pd.DataFrame = filtered_data.loc[filtered_data.index.repeat(8)]

        # Restar horas desde 'Fecha de atención' (ya está en formato datetime)
        new_df["Fecha de atención"] = pd.to_datetime(new_df["Fecha de atención"])
        new_df["Fecha de atención"] = new_df["Fecha de atención"] - pd.to_timedelta(
            new_df.groupby(level=0).cumcount(), unit="h"
        )
        new_df = new_df.sort_values("Fecha de atención")

        # Combinar con datos horarios por fecha
        temp_data = pd.merge_asof(
            new_df,
            prcp_data.sort_values(["time"]),
            left_on="Fecha de atención",
            right_on="time",
            direction="nearest",
        )
        temp_data["prcp"] = temp_data.groupby("index")["prcp"].transform("sum")
        temp_data = temp_data.sort_values(
            ["index", "Fecha de atención"], ascending=[True, False]
        ).drop_duplicates(subset=["index"], keep="first")

        ls.append(temp_data)

    # Unir todo y limpiar
    output = (
        pd.concat(ls)
        .drop(columns=["time", "index"])
        .rename(columns={"prcp": "Precipitation"})
        .reset_index(drop=True)
    )

    # Verificar longitud
    assert len(output) == len(data), (
        f"Mismatch in rows: expected {len(data)}, got {len(output)}"
    )

    return output


def guardar_df_final_parquet(
    hospital_triage_data: pd.DataFrame,
    ruta_salida: str = r"C:\Users\usuario\EmergenciasHospitalarias\data\hospital_triage_data.parquet",
) -> None:
    """
    Guarda el DataFrame procesado en formato Parquet.

    Args:
        hospital_triage_data (pd.DataFrame): DataFrame final.
        ruta_salida (str): Ruta donde guardar el archivo Parquet.

    Returns:
        None
    """
    # Convertir columnas Period a string si las hay
    for col in hospital_triage_data.columns:
        if isinstance(hospital_triage_data[col].dtype, pd.PeriodDtype):
            hospital_triage_data[col] = hospital_triage_data[col].astype(str)

    hospital_triage_data.reset_index(drop=True).to_parquet(
        ruta_salida, engine="fastparquet"
    )


def main():
    hospital_triage_data = cargar_y_transformar_datos(
        r"C:\Users\usuario\EmergenciasHospitalarias\data\urgencias-hospitalarias-atendidas.csv",
        FESTIVOS,
    )
    hospital_triage_data.head()
    nan_triaje_por_mes, proporcion_nan = analizar_nulos_y_duplicados(
        hospital_triage_data
    )
    # procesar_y_guardar_graficos_nans(nan_triaje_por_mes, proporcion_nan)
    hospital_triage_data["Fecha de atención"] = pd.to_datetime(
        hospital_triage_data["Fecha de atención"], errors="coerce"
    )
    hospital_triage_data["Año"] = hospital_triage_data["Fecha de atención"].dt.year
    # generar_graficos_actividad(hospital_triage_data)
    pacientes_por_mes = procesar_pacientes_por_mes(hospital_triage_data)
    generar_comparativa_anual(pacientes_por_mes)  # hay graficos
    analisis_calidad_datos(hospital_triage_data)
    hospital_triage_data = limpieza_final(hospital_triage_data)
    procesar_pacientes_por_mes(hospital_triage_data)
    resumen_pacientes(hospital_triage_data)
    coords_df = (
        hospital_triage_data[["Área", "Latitud", "Longitud"]].dropna().drop_duplicates()
    )
    # hospital_triage_data = get_precipitation(coords_df, hospital_triage_data)
    # Asegurar que Nivel de triaje y Edad sean enteros
    hospital_triage_data["Nivel de triaje"] = hospital_triage_data[
        "Nivel de triaje"
    ].astype(int)
    hospital_triage_data["Edad"] = hospital_triage_data["Edad"].astype(int)

    guardar_df_final_parquet(hospital_triage_data)


if __name__ == "__main__":
    main()
