
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# ---------- Generador 1: Cuadrados Medios ----------
def cuadrados_medios(seed, n, digits=4):
    results = []
    current = seed
    for _ in range(n):
        squared = str(current**2).zfill(2*digits)
        mid = len(squared) // 2
        next_val = int(squared[mid - digits//2: mid + digits//2])
        results.append(next_val / (10**digits))
        current = next_val
    return results

# ---------- Generador 2: Generador Congruencial Lineal (GCL) ----------
def gcl(seed, a, c, m, n):
    values = []
    x = seed
    for _ in range(n):
        x = (a * x + c) % m
        values.append(x / m)
    return values

# ---------- Generador 3: XorShift ----------
def xorshift(seed, n):
    results = []
    x = seed
    for _ in range(n):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17)
        x ^= (x << 5) & 0xFFFFFFFF
        results.append((x & 0xFFFFFFFF) / 0xFFFFFFFF)
    return results

# ---------- Pruebas ----------
def test_frecuencia(valores, bins=10):
    counts, _ = np.histogram(valores, bins=bins, range=(0.0, 1.0))
    esperado = len(valores) / bins
    chi2 = sum((o - esperado) ** 2 / esperado for o in counts)
    return chi2, counts

def test_autocorrelacion(valores, lag=1):
    n = len(valores)
    mean = np.mean(valores)
    num = sum((valores[i] - mean)*(valores[i + lag] - mean) for i in range(n - lag))
    den = sum((valores[i] - mean)**2 for i in range(n))
    return num / den if den != 0 else 0

def test_corridas(valores):
    runs = 1
    for i in range(1, len(valores)):
        if (valores[i] > valores[i - 1]) != (valores[i - 1] > valores[i - 2] if i > 1 else True):
            runs += 1
    return runs

def test_media(valores):
    mean = np.mean(valores)
    return mean

# ---------- Ejecutar y mostrar resultados ----------
n = 1000
seed = 5731

cuadrados_vals = cuadrados_medios(seed, n)
gcl_vals = gcl(seed=7, a=5, c=3, m=16, n=n)
xorshift_vals = xorshift(seed=123456789, n=n)
python_vals = [random.random() for _ in range(n)]

# Guardar resultados en DataFrame
df = pd.DataFrame({
    "Cuadrados": cuadrados_vals,
    "GCL": gcl_vals,
    "XorShift": xorshift_vals,
    "Python": python_vals
})

# Pruebas y análisis
tests = {}
for name in df.columns:
    vals = df[name]
    tests[name] = {
        "Media": test_media(vals),
        "Autocorrelación": test_autocorrelacion(vals),
        "Corridas": test_corridas(vals),
        "Chi2 (Frecuencia)": test_frecuencia(vals)[0]
    }

resultados_df = pd.DataFrame(tests).T

# ---------- Graficar histogramas ----------
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axs.flatten(), df.columns):
    ax.hist(df[col], bins=10, range=(0.0, 1.0), alpha=0.7, color="skyblue", edgecolor="black")
    ax.set_title(f"Histograma: {col}")
plt.tight_layout()
plt.show()

print(resultados_df)
