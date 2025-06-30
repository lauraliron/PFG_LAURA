import matplotlib.pyplot as plt
import pandas as pd

# Cargar el archivo parquet ya generado con el rolling
ruta = r"C:\Users\usuario\EmergenciasHospitalarias\data\datos_resumen.parquet"
df = pd.read_parquet(ruta)

# Elegimos un hospital para el ejemplo
hospital = "Zamora"
df_hospital = df[df["Hospital"].str.contains(hospital, case=False)].copy()

# Aseguramos el orden temporal
df_hospital = df_hospital.sort_values(["Fecha", "Turno"])

# Creamos una columna índice combinando fecha y turno para que se vea mejor en el eje X
df_hospital["FechaTurno"] = (
    df_hospital["Fecha"].astype(str) + " - " + df_hospital["Turno"].astype(str)
)

# Reducimos el número de muestras a las últimas 50 para visualizar mejor
df_plot = df_hospital.tail(50)

# Gráfico
plt.figure(figsize=(14, 6))
plt.plot(
    df_plot["FechaTurno"], df_plot["Edad_media"], label="Edad media real", marker="o"
)
plt.plot(
    df_plot["FechaTurno"],
    df_plot["Rolling_Edad_media"],
    label="Rolling mean (7 turnos)",
    marker="x",
)
plt.xticks(rotation=45)
plt.xlabel("Fecha - Turno")
plt.ylabel("Edad media")
plt.title(f"Edad media vs Rolling mean (últimos 50 turnos) - Hospital {hospital}")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
