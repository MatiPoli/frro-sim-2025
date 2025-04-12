#Matias Luhmann 51498
#Santino Cataldi 50384
#Maria Paz Battistoni 49641
#Marcos Godoy 51192

import random
import matplotlib.pyplot as plt
import statistics as st
import sys

if len(sys.argv) != 7 or sys.argv[1]!="-c" or sys.argv[3]!="-n" or sys.argv[5]!="-e" or (int(sys.argv[6]) not in range(0,37)):
  print("Uso correcto: Python TrabajoPractico1.py -c <Cantidad de tiradas> -n <numero de corridas> -e <numero elegido>")
  sys.exit(1)

# python TrabajoPractico1.py -c 1 -n 10 -e 5


nroTiradas = int(sys.argv[4]) 
nroCorridas = int(sys.argv[2])
nroElegido = int(sys.argv[6])

nroCorridas = 1

resultados = [[0] * nroTiradas for _ in range(nroCorridas)]

def ruleta():
    return random.randint(0, 36)


for corrida in range(nroCorridas):
    for tirada in range(nroTiradas):
        resultados[corrida][tirada]  = ruleta()







# # ----- GRAFICO DE FRECUENCIA RELATIVA -----
# frecuencias_relativas = []
# for corrida in resultados:
#     apariciones = 0
#     for tirada in corrida:
#         if tirada == nroElegido:
#             apariciones = apariciones +1
#         frecuencia_relativa = apariciones / nroTiradas
#         frecuencias_relativas.append(frecuencia_relativa)

# print(frecuencias_relativas)

# # Gráfico de barras
# plt.figure(figsize=(10, 6))
# plt.bar(range(1, nroTiradas + 1), frecuencias_relativas, color='red')
# plt.axhline(1/37, color='blue', linestyle='--', label='Prob. teórica (1/37)')
# plt.xlabel('Tirada')
# plt.ylabel('Frecuencia relativa')
# plt.title(f'Frecuencia relativa del número {nroElegido}')
# plt.xticks(range(1, nroTiradas + 1))
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()



# # ----- GRAFICO DEL PROMEDIO ACUMULADO DE LAS TIRADAS -----

# promedios_acumulados = []
# suma = 0

# for i in range(nroTiradas):
#     suma += resultados[0][i]  # Usamos la primera corrida
#     promedio = suma / (i + 1)
#     promedios_acumulados.append(promedio)
    
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, nroTiradas + 1), promedios_acumulados, color='green', label='Promedio acumulado')
# plt.axhline(18, color='purple', linestyle='--', label='Valor esperado (18)')
# plt.xlabel('Número de tiradas')
# plt.ylabel('Promedio acumulado')
# plt.title('Evolución del promedio de tiradas (Corrida 1)')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()


datos = [i for i in range(0, 37)]
media = sum(datos) / len(datos) # 18
varianza = sum((x - media) ** 2 for x in datos) / (len(datos) - 1)
desvio_estandar = varianza ** 0.5

print("Desvío estándar:", desvio_estandar)


# ----- GRAFICO DEL DESVÍO ESTÁNDAR ACUMULADO -----

desvios_estandar = []

for i in range(1, nroTiradas + 1):
    sub_lista = resultados[0][:i]  # Tomamos las primeras i tiradas de la primera corrida
    desvio = st.stdev(sub_lista) if i > 1 else 0  # stdev necesita al menos 2 valores
    desvios_estandar.append(desvio)



for i in range(nroCorridas):
    print(f"Corrida {i+1}: {resultados[i]}")

plt.figure(figsize=(10, 6))
plt.plot(range(2, nroTiradas + 1), desvios_estandar[1:], color='orange', label='Desvío estándar acumulado')
plt.axhline(desvio_estandar, color='purple', linestyle='--', label=f'Valor esperado {desvio_estandar}')
plt.xlabel('Número de tirada')
plt.ylabel('Desvío estándar')
plt.title('Evolución del desvío estándar de las tiradas (Corrida 1)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ----- GRAFICO DE LA VARIANZA ACUMULADA -----

varianzas = []

for i in range(1, nroTiradas + 1):
    sub_lista = resultados[0][:i]  # Primeras i tiradas de la primera corrida
    varianza = st.variance(sub_lista) if i > 1 else 0
    varianzas.append(varianza)

plt.figure(figsize=(10, 6))
plt.plot(range(2, nroTiradas + 1), varianzas[1:], color='darkcyan', label='Varianza acumulada')
plt.axhline(varianza, color='purple', linestyle='--', label=f'Valor esperado {varianza}')
plt.xlabel('Número de tiradas')
plt.ylabel('Varianza')
plt.title('Evolución de la varianza de las tiradas (Corrida 1)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
