import math
from matplotlib import pyplot as plt
import numpy as np


def generar_grafico_desvio_estandar(desvios_estandar: list[list[float]]) -> None:
    plt.figure(figsize=(12, 8))
    cantidad_tiradas = len(desvios_estandar[0])
    desvio_esperado = math.sqrt((37**2 - 1) / 12)

    for i, desv in enumerate(desvios_estandar):
        plt.plot(range(2, cantidad_tiradas + 1), desv[1:], label=f"Corrida {i + 1}")

    plt.axhline(
        y=desvio_esperado, color="red", linestyle="--", label="Desvio estandar esperado"
    )
    plt.xlabel("Número de tiradas")
    plt.ylabel("Desvío estándar acumulado")
    plt.title("Evolución del desvío estándar en cada corrida")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig("Graficos/desvio_estandar_x_tiradas.png")
    plt.close()


def generar_grafico_frecuencia_relativa(
    frecuencias_relativas: list[list[float]], numero_elegido: int
) -> None:
    plt.figure(figsize=(12, 8))
    cantidad_tiradas = len(frecuencias_relativas[0])

    for i, fr in enumerate(frecuencias_relativas):
        plt.plot(range(1, cantidad_tiradas + 1), fr, label=f"Corrida {i + 1}")

    plt.axhline(
        y=1 / 37, color="red", linestyle="--", label="Frecuencia Relativa Esperada"
    )
    plt.xlabel("Número de tiradas")
    plt.ylabel("Frecuencia relativa acumulada")
    plt.title(f"Evolución de la frecuencia relativa del número {numero_elegido}")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig("Graficos/frecuencia_relativa_x_tiradas.png")
    plt.close()


def generar_grafico_valor_promedio(promedios: list[list[float]]) -> None:
    plt.figure(figsize=(12, 8))
    cantidad_tiradas = len(promedios[0])

    for i, prom in enumerate(promedios):
        plt.plot(range(1, cantidad_tiradas + 1), prom, label=f"Corrida {i + 1}")

    plt.axhline(
        y=sum([x for x in range(0, 37)]) / 37,
        color="red",
        linestyle="--",
        label="Valor Promedio Esperado",
    )
    plt.xlabel("Número de tiradas")
    plt.ylabel("Valor promedio acumulado")
    plt.title("Evolución del valor promedio")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig("Graficos/valor_promedio_x_tiradas.png")
    plt.close()


def generar_grafico_varianza(varianzas: list[list[float]]) -> None:
    plt.figure(figsize=(12, 8))
    cantidad_tiradas = len(varianzas[0])

    varianza_esperada = (37**2 - 1) / 12

    for i, var in enumerate(varianzas):
        plt.plot(range(2, cantidad_tiradas + 1), var[1:], label=f"Corrida {i + 1}")

    plt.axhline(
        y=varianza_esperada, color="red", linestyle="--", label="Varianza esperada"
    )
    plt.xlabel("Número de tiradas")
    plt.ylabel("Varianza acumulada")
    plt.title("Evolución de la varianza")
    plt.grid()
    plt.tight_layout()
    plt.legend()
    plt.savefig("Graficos/varianza_x_tiradas.png")
    plt.close()


def generar_grafico_frecuencia_por_numero(todas_las_tiradas: list[int]) -> None:
    conteo = np.zeros(37)
    for num in todas_las_tiradas:
        conteo[num] += 1

    frecuencia_relativa = conteo / len(todas_las_tiradas)

    plt.figure(figsize=(12, 6))
    plt.bar(
        range(37),
        frecuencia_relativa,
        color="skyblue",
        edgecolor="black",
        label="Frecuencia relativa simulada",
    )
    plt.axhline(
        y=1 / 37, color="red", linestyle="--", label="Frecuencia esperada (1/37)"
    )
    plt.xlabel("Número de la ruleta")
    plt.ylabel("Frecuencia relativa")
    plt.title("Frecuencia relativa de cada número en la simulación")
    plt.xticks(range(37))
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig("Graficos/frecuencia_relativa_por_numero.png")
    plt.close()


def generar_heat_map_frecuencia_absoluta(
    frecuencias_absolutas: list[list[int]],
) -> None:
    plt.figure(figsize=(12, 8))
    plt.imshow(frecuencias_absolutas, cmap="inferno", aspect="auto")
    plt.colorbar(label="Frecuencia")
    plt.title("Heatmap de frecuencias por número (0-36) en cada corrida")
    plt.xlabel("Número de ruleta")
    plt.ylabel("Corrida simulada")
    plt.xticks(ticks=range(37), labels=range(37), rotation=90)
    plt.yticks(
        ticks=range(len(frecuencias_absolutas)),
        labels=[f"Corrida {i + 1}" for i in range(len(frecuencias_absolutas))],
        rotation=0,
    )
    plt.tight_layout()
    plt.savefig("Graficos/heatmap_frecuencia_absoluta.png")
    plt.close()
