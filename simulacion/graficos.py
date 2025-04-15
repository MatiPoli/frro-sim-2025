import math
from matplotlib import pyplot as plt
import statistics

from simulacion.estadisticas import calcular_desvios_estandar, calcular_frecuencias_relativas, calcular_promedios, frecuencias_relativa_por_corrida, promedios_por_corrida


def generar_grafico_desvio_estandar(corridas: list[list[int]]) -> None:
    desvios_estandar = calcular_desvios_estandar(corridas)
    cantidad_tiradas = len(corridas[0])
    desvio_esperado = math.sqrt((37**2 - 1) / 12)

    for i, desv in enumerate(desvios_estandar):
        plt.plot(range(2, cantidad_tiradas + 1), desv[1:], label=f"Corrida {i + 1}")

    plt.axhline(
        y=desvio_esperado,
        color="red",
        linestyle="--",
        label="Desvío Estándar Esperado"
    )
    plt.xlabel("Número de tiradas")
    plt.ylabel("Desvío estándar acumulado")
    plt.title("Evolución del desvío estándar en cada corrida")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("desvio_estandar_x_tiradas.png")
    plt.close()




def generar_grafico_frecuencia_relativa(
    corridas: list[list[int]], numero_elegido: int
) -> None:
    frecuencias_relativas = calcular_frecuencias_relativas(corridas, numero_elegido)
    cantidad_tiradas = len(frecuencias_relativas[0])

    for i, fr in enumerate(frecuencias_relativas):
        plt.plot(range(1, cantidad_tiradas + 1), fr, label=f"Corrida {i + 1}")

    plt.axhline(y=1 / 37, color="red", linestyle="--", label="Frecuencia Relativa Esperada")
    plt.xlabel("Número de tiradas")
    plt.ylabel("Frecuencia relativa acumulada")
    plt.title(f"Evolución de la frecuencia relativa del número {numero_elegido}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("frecuencia_relativa_x_tiradas.png")
    plt.close()


def generar_grafico_valor_promedio(corridas: list[list[int]]) -> None:
    # Calcular los promedios acumulados para cada corrida
    promedios = calcular_promedios(corridas)

    cantidad_tiradas = len(promedios[0])

    for i, prom in enumerate(promedios):
        plt.plot(range(1, cantidad_tiradas + 1), prom, label=f"Corrida {i + 1}")

    plt.axhline(
        y=sum(range(0, 37)) / 37,  # Valor esperado de una ruleta de 0 a 36 (uniforme)
        color="red",
        linestyle="--",
        label="Valor Promedio Esperado"
    )
    plt.xlabel("Número de tiradas")
    plt.ylabel("Valor promedio acumulado")
    plt.title("Evolución del valor promedio")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("valor_promedio_x_tiradas.png")
    plt.close()