import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os

# ---------- Generador 1: Cuadrados Medios ----------
def cuadrados_medios(seed, n, digits=4):
    results = []
    current = seed
    for _ in range(n):
        squared = str(current**2).zfill(2*digits)
        mid = len(squared) // 2
        next_val = int(squared[mid - digits//2: mid + digits//2]) #se toman los digitos centrales
        results.append(next_val / (10**digits)) #se normaliza
        current = next_val
    return results

# ---------- Generador 2: Generador Congruencial Lineal (GCL) ----------
def gcl(seed, a, c, m, n): #a = multiplicador, c = constante aditiva, m = módulo normalizador
    # GCL: X_n+1 = (a * X_n + c) mod m
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
    for _ in range(n):  #0xFFFFFFFF es el valor máximo de 32 bits, recortando a 32 bits
        x ^= (x << 13) & 0xFFFFFFFF #desplzamos hacia la izquierda 13 bits de los bits de x(si x=123456789, el bin 32 es 01111000101011010111100100010101) y se compara bit a bit, y se guarda el camparacion, si son iguales 0 y sino 1
        x ^= (x >> 17) #se desplaza 17 bits a la derecha y se compara bit a bit
        x ^= (x << 5) & 0xFFFFFFFF #se desplaza 5 bits a la izquierda y se compara bit a bit 
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
n = 10000
seed = 84133294


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


# ---------- Graficar histogramas en archivos separados ----------
for col in df.columns:
    plt.figure(figsize=(6, 4))
    plt.hist(df[col], bins=10, range=(0.0, 1.0), alpha=0.7, color="skyblue", edgecolor="black")
    plt.title(f"Histograma: {col}")
    plt.xlabel("Valor")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(f"graficas/histograma_{col.lower()}.png")
    plt.close()
print(resultados_df)


output_dir = "graficas"
os.makedirs(output_dir, exist_ok=True)

# Función para generar y guardar la imagen
def generar_imagen(datos, nombre_archivo):
    if len(datos) != 10000:
        raise ValueError("Cada columna debe tener 10.000 números")

    matriz = np.array(datos).reshape((100, 100))
    plt.imshow(matriz, cmap='gray', interpolation='nearest')
    plt.axis('off')
    ruta_completa = os.path.join(output_dir, f"{nombre_archivo}.png")
    plt.savefig(ruta_completa, bbox_inches='tight', pad_inches=0)
    plt.close()

# Recorrer las columnas del DataFrame
for nombre_columna in df.columns:
    generar_imagen(df[nombre_columna], nombre_columna)

