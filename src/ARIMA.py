import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA

from src.utils import cargar_datos_parquet


def inspeccionar_columnas(df: pd.DataFrame) -> list[str]:
    print(df.columns)
    columnas_a_eliminar = ["Fecha", "Hospital"]
    pd.set_option("display.max_columns", None)
    print(df.head())
    return columnas_a_eliminar


def entrenar_y_predecir_arima(y: pd.Series):
    print(" ARIMA no usa variables independientes. Se usará solo la serie 'y'.")
    tscv = TimeSeriesSplit(n_splits=3)
    errores = []
    orden = (2, 1, 2)

    for i, (train_idx, test_idx) in enumerate(tscv.split(y)):
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        try:
            modelo = ARIMA(y_train, order=orden)
            modelo_fit = modelo.fit()
            pred = modelo_fit.forecast(steps=len(y_test))
            error = mean_absolute_error(y_test, pred)
            errores.append(error)
            print(f"Split {i + 1} MAE: {error:.4f}")
        except Exception as e:
            print(f"Error en split {i + 1}: {e}")
            continue

    print("MAE promedio (ARIMA):", np.mean(errores))
    return ARIMA(y, order=orden).fit()


def main():
    ruta = r"C:\\Users\\usuario\\EmergenciasHospitalarias\\data\\datos_resumen.parquet"
    df = cargar_datos_parquet(ruta)

    df = df[df["Hospital"] == "C.A. Zamora"]
    columnas_a_eliminar = inspeccionar_columnas(df)

    hospitales = df["Hospital"].dropna().unique()

    for hospital in hospitales:
        print(f"\n===== Procesando hospital: {hospital} =====")
        df_h = df[df["Hospital"] == hospital].dropna()
        if df_h.shape[0] < 100:
            print("Demasiado pocos registros, se omite.")
            continue

        last_date = df_h["Fecha"].max()
        cutoff_date = last_date - pd.DateOffset(months=3)

        train = df_h[df_h["Fecha"] <= cutoff_date].copy()
        test = df_h[df_h["Fecha"] > cutoff_date].copy()

        if train.empty or test.empty:
            print("Sin datos suficientes para entrenar o testear. Se omite.")
            continue

        X_train = train.drop(
            columns=columnas_a_eliminar
            + ["Turno", "Pacientes", "Pacientes_por_triaje"],
            errors="ignore",
        )
        y_train = train["Pacientes"]
        X_test = test.drop(
            columns=columnas_a_eliminar
            + ["Turno", "Pacientes", "Pacientes_por_triaje"],
            errors="ignore",
        )
        y_test = test["Pacientes"]

        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        print("Entrenando modelo para:", hospital)
        best_model = entrenar_y_predecir_arima(y_train)
        y_pred = best_model.forecast(steps=len(y_test))

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print(f"MAE para {hospital}:", mae)
        print(f"MSE para {hospital}:", mse)

        shift_times = {"Madrugada": "08:00:00", "Día": "16:00:00", "Noche": "00:00:00"}
        test["PlotDateTime"] = pd.to_datetime(
            test["Fecha"].dt.strftime("%Y-%m-%d") + " " + test["Turno"].map(shift_times)
        )

        # Asegurar longitudes iguales
        N = min(len(y_pred), len(y_test), len(test["PlotDateTime"]))
        comparison_df = pd.DataFrame(
            {
                "Fecha": test["PlotDateTime"].iloc[:N].values,
                "Actual": y_test.iloc[:N].values,
                "Predicted": y_pred[:N],
            }
        ).sort_values(by="Fecha")

        plt.figure(figsize=(14, 6))
        plt.plot(
            comparison_df["Fecha"], comparison_df["Actual"], label="Actual", marker="o"
        )
        plt.plot(
            comparison_df["Fecha"],
            comparison_df["Predicted"],
            label="Predicted",
            marker="x",
        )
        plt.xlabel("Fecha y Turno")
        plt.ylabel("Pacientes")
        plt.title(f"Predicción para {hospital}")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
