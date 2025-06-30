# Importamos las librerías necesarias
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import cargar_datos_parquet


def inspeccionar_columnas(df: pd.DataFrame) -> list[str]:
    print(df.columns)
    columnas_a_eliminar = ["Fecha", "Hospital"]
    pd.set_option("display.max_columns", None)
    print(df.head())
    return columnas_a_eliminar


def grid_search_gradient_boosting(
    X: pd.DataFrame, y: pd.Series
) -> GradientBoostingRegressor:
    categorical_features = X.select_dtypes(include="object").columns.tolist()
    numeric_features = [col for col in X.columns if col not in categorical_features]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", GradientBoostingRegressor(random_state=42)),
        ]
    )

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_depth": [2, 4, 8],
        "model__min_samples_split": [2, 4, 8],
        "model__min_samples_leaf": [2, 4, 8],
    }

    tscv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=tscv,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )

    grid_search.fit(X, y)
    print("\nMejores hiperparámetros encontrados:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")

    return grid_search.best_estimator_


def procesar_hospital(df: pd.DataFrame, nombre_hospital: str):
    print(f"\n===== Procesando hospital: {nombre_hospital} =====")

    hospital_data = df[df["Hospital"] == nombre_hospital].dropna()
    if len(hospital_data) < 100:
        print("Datos insuficientes, se omite este hospital.")
        return

    columnas_a_eliminar = inspeccionar_columnas(hospital_data)
    last_date = hospital_data["Fecha"].max()
    cutoff_date = last_date - pd.DateOffset(months=3)

    train_data = hospital_data[hospital_data["Fecha"] <= cutoff_date].copy()
    test_data = hospital_data[hospital_data["Fecha"] > cutoff_date].copy()

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

    best_model = grid_search_gradient_boosting(X_train, y_train)

    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MAE para {nombre_hospital}: {mae:.2f}")
    print(f"Test MSE para {nombre_hospital}: {mse:.2f}")

    # Agregar hora ficticia según turno
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
    plt.title(f"Predicción de Pacientes - {nombre_hospital}")
    plt.xlabel("Fecha y Turno")
    plt.ylabel("Pacientes")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    ruta = r"C:\Users\usuario\EmergenciasHospitalarias\data\datos_resumen.parquet"
    df = cargar_datos_parquet(ruta)

    df["Fecha"] = pd.to_datetime(df["Fecha"])
    hospitales_unicos = df["Hospital"].dropna().unique()

    for hospital in hospitales_unicos:
        procesar_hospital(df, hospital)


if __name__ == "__main__":
    main()
