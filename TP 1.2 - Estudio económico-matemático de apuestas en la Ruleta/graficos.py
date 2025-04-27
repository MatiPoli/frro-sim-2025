import math
from matplotlib import pyplot as plt
import numpy as np



#plt.savefig("Graficos/desvio_estandar_x_tiradas.png")

def graficar_saldo(saldo_por_tirada: list[int], tiradas_bancarrota: list[int]) -> None:
    
    saldo_inicial = saldo_por_tirada[0]  # Obtener el saldo inicial
    plt.figure(figsize=(10, 6))
    for tirada in tiradas_bancarrota:
        plt.scatter(tirada, saldo_por_tirada[tirada], color="red", zorder=5)
    plt.text(0.95, 0.01, f"Bancarrotas: {len(tiradas_bancarrota)}", transform=plt.gca().transAxes, 
         fontsize=12, color="black", ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.5))
    plt.plot(saldo_por_tirada, label="Saldo", color="blue")
    plt.axhline(y=saldo_inicial, color="green", linestyle="--", label="Saldo inicial")  # Línea constante
    plt.xlabel("Número de tiradas")
    plt.ylabel("Saldo")
    plt.title("Evolución del saldo a lo largo de las tiradas")
    plt.legend()
    plt.grid(True)
    plt.savefig("Graficos/saldo_por_tirada.png")
    plt.show()


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
    plt.show()


def graficar_apuesta_realizada(apuesta_por_tirada: list[int]):
    apuesta_inicial = apuesta_por_tirada[0]  # Obtener la primera apuesta (si querés marcarla)
    plt.figure(figsize=(10, 6))
    plt.plot(apuesta_por_tirada, label="Apuesta realizada", color="red")
    plt.axhline(y=apuesta_inicial, color="orange", linestyle="--", label="Apuesta inicial")  # Línea constante
    plt.xlabel("Número de tiradas")
    plt.ylabel("Monto apostado")
    plt.title("Evolución de la apuesta realizada a lo largo de las tiradas")
    plt.legend()
    plt.grid(True)
    plt.savefig("Graficos/apuesta_realizada_por_tirada.png")
    plt.show()
