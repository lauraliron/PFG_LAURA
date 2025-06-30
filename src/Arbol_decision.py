import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.utils import cargar_datos_parquet


def inspeccionar_columnas(df: pd.DataFrame) -> list[str]:
    print(df.columns)
    columnas_a_eliminar = ["Fecha", "Hospital"]
    pd.set_option("display.max_columns", None)
    print(df.head())
    return columnas_a_eliminar


def grid_search_decision_tree(X: pd.DataFrame, y: pd.Series):
    categorical_features = X.select_dtypes(include="object").columns.tolist()
    numeric_features = [col for col in X.columns if col not in categorical_features]

    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", DecisionTreeRegressor(random_state=42)),
        ]
    )

    param_grid = {
        "model__max_depth": [4, 8, 12],
        "model__min_samples_split": [2, 10, 20],
        "model__min_samples_leaf": [1, 5, 10],
    }

    tscv = TimeSeriesSplit(n_splits=3)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        n_jobs=-1,
        verbose=2,
        refit=True,
    )

    grid_search.fit(X, y)
    print("Mejores hiperparámetros:", grid_search.best_params_)
    print("Mejor MAE negativo:", grid_search.best_score_)
    return grid_search.best_estimator_


def main():
    ruta = r"C:\Users\usuario\EmergenciasHospitalarias\data\datos_resumen.parquet"
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
        best_model = grid_search_decision_tree(X_train, y_train)

        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"MAE para {hospital}:", mae)

        shift_times = {"Madrugada": "08:00:00", "Día": "16:00:00", "Noche": "00:00:00"}
        test["PlotDateTime"] = pd.to_datetime(
            test["Fecha"].dt.strftime("%Y-%m-%d") + " " + test["Turno"].map(shift_times)
        )

        comparison_df = pd.DataFrame(
            {
                "Fecha": test["PlotDateTime"],
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
