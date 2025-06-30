# Importamos las librerias
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from src.utils import cargar_datos_parquet


def inspeccionar_columnas(zamora: pd.DataFrame) -> list[str]:
    """
    Muestra las columnas del DataFrame y un head completo sin recorte de columnas.
    También devuelve una lista de columnas candidatas a eliminar.
    """
    print(zamora.columns)

    columnas_a_eliminar = [
        "Fecha",
        "Hospital",
    ]
    pd.set_option("display.max_columns", None)
    print(zamora.head())

    return columnas_a_eliminar


def entrenar_gradient_boosting_timeseries(
    X: pd.DataFrame,
    y: pd.Series,
    fechas: pd.Series,
    turnos: pd.Series,
    n_splits: int = 2,
    gap: int = 48,
    max_train_size: int = 10000,
    test_size: int = 1000,
) -> GradientBoostingRegressor:
    """
    Entrena un modelo Gradient Boosting con validación temporal usando TimeSeriesSplit
    y muestra el gráfico con 100 predicciones ordenadas por fecha y turno.
    """
    ts_cv = TimeSeriesSplit(
        n_splits=n_splits, gap=gap, max_train_size=max_train_size, test_size=test_size
    )

    for train_idx, test_idx in ts_cv.split(X):
        pass  # Nos quedamos con el último split

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    fechas_test = fechas.iloc[test_idx]
    turnos_test = turnos.iloc[test_idx]

    modelo = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        min_samples_split=2,
        min_samples_leaf=3,
        max_depth=3,
        random_state=None,
        n_iter_no_change=None,
    )
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f" MSE: {mse:.2f}")
    print(f" R²: {r2:.4f}")

    # Preparar DataFrame para gráfico
    df_plot = pd.DataFrame(
        {
            "fecha": fechas_test,
            "turno": turnos_test,
            "y_real": y_test.values,
            "y_pred": y_pred,
        }
    )

    orden_turnos = ["Madrugada", "Día", "Tarde", "Noche"]
    df_plot["turno"] = pd.Categorical(
        df_plot["turno"], categories=orden_turnos, ordered=True
    )
    df_plot = df_plot.sort_values(by=["fecha", "turno"])

    df_plot["fecha_turno_str"] = (
        df_plot["fecha"].astype(str) + " - " + df_plot["turno"].astype(str)
    )
    df_plot["x"] = range(len(df_plot))

    # Mostrar número de predicciones por turno
    print("\n🔍 Nº de predicciones por turno en los datos de test:")
    print(df_plot["turno"].value_counts())

    # Mostrar solo las primeras 100 predicciones
    df_plot = df_plot.iloc[:100]

    # Gráfico
    plt.figure(figsize=(16, 6))
    plt.plot(df_plot["x"], df_plot["y_real"], label="Real", marker="o")
    plt.plot(df_plot["x"], df_plot["y_pred"], label="Predicción", marker="x")
    plt.xticks(df_plot["x"][::3], df_plot["fecha_turno_str"][::3], rotation=45)
    plt.title("Pacientes por turno - Real vs Predicción (100 muestras)")
    plt.xlabel("Fecha - Turno")
    plt.ylabel("Pacientes")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    return modelo


def main():
    ruta = r"C:\PFG\PlantillaPFG\data\datos_resumen.parquet"
    hospital_data = cargar_datos_parquet(ruta)

    df_zamora = hospital_data[
        hospital_data["Hospital"].str.contains("Zamora", case=False)
    ]
    zamora = df_zamora.dropna()

    fechas = zamora["Fecha"]
    turnos = zamora["Turno"]

    columnas_a_eliminar = inspeccionar_columnas(zamora)

    X = zamora.drop(
        columns=columnas_a_eliminar + ["Turno", "Pacientes"], errors="ignore"
    )
    y = zamora["Pacientes"]

    columnas_categoricas = X.select_dtypes(include="object").columns.tolist()
    if columnas_categoricas:
        print(f"Convertimos columnas categóricas: {columnas_categoricas}")
        X = pd.get_dummies(X, columns=columnas_categoricas, drop_first=True)

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    fechas = fechas.reset_index(drop=True)
    turnos = turnos.reset_index(drop=True)

    entrenar_gradient_boosting_timeseries(X, y, fechas, turnos)


if __name__ == "__main__":
    main()
