# Importamos las librerías necesarias
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.utils import cargar_datos_parquet


def inspeccionar_columnas(df: pd.DataFrame) -> list[str]:
    pd.set_option("display.max_columns", None)
    print("\nColumnas disponibles:", df.columns.tolist())
    print("\nPrimeras filas del DataFrame:\n", df.head())
    return ["Fecha", "Hospital"]


def grid_search_hist_gradient_boosting(X: pd.DataFrame, y: pd.Series):
    categorical_features = X.select_dtypes(include="object").columns.tolist()

    preprocessor = make_column_transformer(
        (
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_features,
        ),
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", HistGradientBoostingRegressor(random_state=42)),
        ]
    )

    param_grid = {
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_iter": [100, 200],
        "model__max_depth": [4, 8, None],
        "model__min_samples_leaf": [20, 50],
        "model__l2_regularization": [0.0, 0.1, 1.0],
    }

    tscv = TimeSeriesSplit(n_splits=3)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )

    grid_search.fit(X, y)

    print("  - Mejores hiperparámetros:", grid_search.best_params_)
    print("  - Mejor MAE negativo:", grid_search.best_score_)
    return grid_search.best_estimator_


def procesar_hospital(df: pd.DataFrame, nombre_hospital: str):
    print(f"\n Procesando hospital: {nombre_hospital}")

    datos = df[df["Hospital"] == nombre_hospital].dropna()
    if len(datos) < 100:
        print("  No hay suficientes datos, se omite.")
        return

    last_date = datos["Fecha"].max()
    cutoff_date = last_date - pd.DateOffset(months=3)

    train_data = datos[datos["Fecha"] <= cutoff_date].copy()
    test_data = datos[datos["Fecha"] > cutoff_date].copy()

    if train_data.empty or test_data.empty:
        print("  División temporal inválida (train o test vacío).")
        return

    columnas_a_eliminar = inspeccionar_columnas(datos)

    X_train = train_data.drop(
        columns=columnas_a_eliminar + ["Turno", "Pacientes", "Pacientes_por_triaje"],
        errors="ignore",
    )
    y_train = train_data["Pacientes"]

    X_test = test_data.drop(
        columns=columnas_a_eliminar + ["Turno", "Pacientes", "Pacientes_por_triaje"],
        errors="ignore",
    )
    y_test = test_data["Pacientes"]

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    best_model = grid_search_hist_gradient_boosting(X_train, y_train)

    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"  MAE en test para {nombre_hospital}: {mae:.2f}")
    print(f"  MSE en test para {nombre_hospital}: {mse:.2f}")

    # Generación de fechas con turno para el gráfico
    shift_times = {"Madrugada": "08:00:00", "Día": "16:00:00", "Noche": "00:00:00"}
    test_data["PlotDateTime"] = pd.to_datetime(
        test_data["Fecha"].dt.strftime("%Y-%m-%d")
        + " "
        + test_data["Turno"].map(shift_times)
    )

    comparison_df = pd.DataFrame(
        {
            "Fecha": test_data["PlotDateTime"],
            "Actual": y_test.to_numpy(),
            "Predicted": y_pred,
        }
    ).sort_values(by="Fecha")

    # Graficamos los resultados
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
    plt.title(f"{nombre_hospital} - Actual vs Predicted Pacientes (Test)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    print(" Cargando y preparando los datos...")
    ruta = r"C:\Users\usuario\EmergenciasHospitalarias\data\datos_resumen.parquet"
    df = cargar_datos_parquet(ruta)

    hospitales = df["Hospital"].dropna().unique().tolist()

    for hospital in hospitales:
        procesar_hospital(df, hospital)


if __name__ == "__main__":
    main()
