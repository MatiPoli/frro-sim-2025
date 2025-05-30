import math
from matplotlib import cm, pyplot as plt
import numpy as np



def graficar_saldo(corridas_saldo: list[list[int]]) -> None:
    
    max_tiradas = max(len(s) for s in corridas_saldo)
    for i, saldo in enumerate(corridas_saldo):
        saldo_padded = saldo + [None] * (max_tiradas - len(saldo))
        plt.plot(range(0, max_tiradas), saldo_padded, label=f"Corrida {i + 1}")

    plt.axhline(y=1000, color="red", linestyle="--", label="Saldo Inicial")
    plt.xlabel("Número de tiradas")
    plt.ylabel("Saldo")
    plt.legend()
    plt.grid(True)
    plt.savefig("Graficos/saldos_x_tirada.png")


def generar_grafico_frecuencia_apuesta_favorable(frecuencias_relativas: list[list[float]]) -> None:
    plt.figure(figsize=(12, 8))
    cantidad_tiradas = len(frecuencias_relativas[0])

    for i, fr in enumerate(frecuencias_relativas):
        plt.plot(range(1, cantidad_tiradas + 1), fr, label=f"Corrida {i + 1}")

    plt.axhline(
        y=18 / 37, color="red", linestyle="--", label="Frecuencia Relativa Esperada (18/37)"
    )
    plt.xlabel("Número de tiradas")
    plt.ylabel("Frecuencia relativa acumulada")
    plt.title("Evolución de la frecuencia relativa de obtener la apuesta favorable")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig("Graficos/frecuencia_relativa_apuesta_favorable.png")


def graficar_apuesta_realizada(corridas_apuestas: list[list[int]]):
    apuesta_inicial = corridas_apuestas[0][0]  # Obtener la primera apuesta
    plt.figure(figsize=(10, 6))
    for i, apuesta_por_tirada in enumerate(corridas_apuestas):
        plt.plot(range(1, len(apuesta_por_tirada) + 1), apuesta_por_tirada, label=f"Corrida {i + 1}")
    plt.axhline(y=apuesta_inicial, color="orange", linestyle="--", label="Apuesta inicial")  # Línea constante
    plt.xlabel("Número de tiradas")
    plt.ylabel("Monto apostado")
    plt.title("Evolución de la apuesta realizada a lo largo de las tiradas")
    plt.legend()
    plt.grid(True)
    plt.savefig("Graficos/apuesta_realizada_por_tirada.png")


