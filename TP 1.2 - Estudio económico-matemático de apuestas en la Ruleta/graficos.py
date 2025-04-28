import math
from matplotlib import cm, pyplot as plt
import numpy as np



#plt.savefig("Graficos/desvio_estandar_x_tiradas.png")

def graficar_saldo(corridas_saldo: list[list[int]], corridas_bancarrota: list[list[int]]) -> None:
    
    colormap = cm.get_cmap("tab20") 
    num_colores = len(corridas_saldo)

    plt.figure(figsize=(12, 8))
    tot_bancarrotas = 0
    cantidad_tiradas = len(corridas_saldo[0])
    saldo_inicial = corridas_saldo[0][0]  # Obtener el saldo inicial  
    plt.axhline(y=saldo_inicial, color="green", linestyle="--", label="Saldo inicial")   
    for i, (saldo_por_tirada, tiradas_bancarrota) in enumerate(zip(corridas_saldo, corridas_bancarrota)):
        color = colormap(i / num_colores)
        plt.plot(range(1, cantidad_tiradas + 1), saldo_por_tirada, label=f"Corrida {i + 1}", color=color)
        for tirada in tiradas_bancarrota:
            plt.scatter(tirada, saldo_por_tirada[tirada], zorder=5, color=color)
        tot_bancarrotas += len(tiradas_bancarrota)


    prom_bancarrotas = tot_bancarrotas / len(corridas_bancarrota) if len(corridas_bancarrota) > 0 else 0    
    plt.text(0.95, 0.01, f"Promedio Bancarrotas: {prom_bancarrotas}", transform=plt.gca().transAxes,fontsize=12, color="black", ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.5))
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


def graficar_apuesta_realizada(corridas_apuestas: list[list[int]]):
    apuesta_inicial = corridas_apuestas[0][0]  # Obtener la primera apuesta (si querés marcarla)
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
    plt.show()
