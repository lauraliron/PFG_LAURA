# Importamos las librerías necesarias
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import cargar_datos_parquet


def inspeccionar_columnas(df: pd.DataFrame) -> list[str]:
    print(df.columns)
    # Definimos las columnas que no estarán disponibles en tiempo de predicción
    columnas_a_eliminar = ["Fecha", "Hospital"]
    pd.set_option("display.max_columns", None)
    print(df.head())
    return columnas_a_eliminar


def grid_search_gradient_boosting(
    X: pd.DataFrame, y: pd.Series
) -> GradientBoostingRegressor:
    """
    Realiza la búsqueda de hiperparámetros usando GridSearchCV y TimeSeriesSplit sobre un pipeline
    que incluye un preprocesador (para numéricos y categóricos) y un GradientBoostingRegressor.
    Devuelve el mejor modelo ajustado.
    """
    # Identificamos las columnas categóricas y numéricas
    categorical_features = X.select_dtypes(include="object").columns.tolist()
    numeric_features = [col for col in X.columns if col not in categorical_features]

    # Creamos transformadores para las variables numéricas y categóricas
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    # Combinamos ambos transformadores en un ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Construimos el pipeline completo: preprocesamiento + modelo
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", GradientBoostingRegressor(random_state=42)),
        ]
    )

    # Configuramos la búsqueda de hiperparámetros; usamos llaves con el prefijo "model__"
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_depth": [2, 4, 8],
        "model__min_samples_split": [2, 4, 8],
        "model__min_samples_leaf": [2, 4, 8],
    }

    # Usamos una validación en serie temporal similar a la de tu versión correcta
    tscv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",  # para consistencia con la métrica MSE negativa
        cv=tscv,
        n_jobs=-1,
        verbose=2,
        refit=True,
    )

    grid_search.fit(X, y)

    print("\nMejores hiperparámetros encontrados:")
    print(grid_search.best_params_)
    print(f"Mejor puntuación (neg MSE) en validación: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def main():
    # Cargamos los datos usando la función provista
    ruta = r"C:\Users\usuario\EmergenciasHospitalarias\data\datos_resumen.parquet"
    hospital_data = cargar_datos_parquet(ruta)

    # Filtramos los datos para quedarnos solo con los casos en Zamora
    df_zamora = hospital_data[
        hospital_data["Hospital"].str.contains("Zamora", case=False)
    ]
    zamora = df_zamora.dropna()

    # Calculamos la fecha límite para los últimos 3 meses
    last_date = zamora["Fecha"].max()
    cutoff_date = last_date - pd.DateOffset(months=3)

    # Dividimos los datos en conjuntos de entrenamiento y prueba
    train_data = zamora[zamora["Fecha"] <= cutoff_date].copy()
    test_data = zamora[zamora["Fecha"] > cutoff_date].copy()

    # Inspeccionamos y definimos las columnas que deseamos eliminar
    columnas_a_eliminar = inspeccionar_columnas(zamora)

    # Preparamos los datos de entrenamiento
    X_train = train_data.drop(
        columns=columnas_a_eliminar + ["Turno", "Pacientes", "Pacientes_por_triaje"],
        errors="ignore",
    )
    y_train = train_data["Pacientes"]

    # Preparamos los datos de prueba
    X_test = test_data.drop(
        columns=columnas_a_eliminar + ["Turno", "Pacientes", "Pacientes_por_triaje"],
        errors="ignore",
    )
    y_test = test_data["Pacientes"]

    # Aseguramos que los índices estén reiniciados
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print(
        "\n Ejecutando GridSearchCV con TimeSeriesSplit en el conjunto de entrenamiento..."
    )
    best_model = grid_search_gradient_boosting(X_train, y_train)

    # Hacemos las predicciones sobre el conjunto de prueba
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print("Test MAE:", mae)

    # Extraemos la columna "Fecha" del conjunto de prueba para poder graficar
    dates_test = test_data["Fecha"].reset_index(drop=True)

    # Map 'Turno' to specific times
    shift_times = {"Madrugada": "08:00:00", "Día": "16:00:00", "Noche": "00:00:00"}

    # Create a new datetime column that combines the 'Fecha' column and the mapped time from 'Turno'
    test_data["PlotDateTime"] = pd.to_datetime(
        test_data["Fecha"].dt.strftime("%Y-%m-%d")
        + " "
        + test_data["Turno"].map(shift_times)
    )

    # Create DataFrame for plotting using the new datetime
    comparison_df = pd.DataFrame(
        {
            "Fecha": test_data["PlotDateTime"],
            "Actual": y_test.to_numpy(),
            "Predicted": y_pred,
        }
    ).sort_values(by="Fecha")

    # Plot the results using the adjusted datetime
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
    plt.title("Actual vs Predicted Pacientes (Conjunto de Prueba)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
