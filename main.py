import sys
import random
import argparse 
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
                    prog='main.py',
                    description='Simulación de ruleta',
                    epilog='Text at the bottom of help')
    parser.add_argument("-c", "--corridas", type=int, required=True, help="Cantidad de corridas")
    parser.add_argument("-n", "--tiradas", type=int, required=True, help="Cantidad de tiradas por corrida")
    parser.add_argument("-e", "--elegido", type=int, required=True, choices=range(0, 37), help="Número elegido (entre 0 y 36)")


def tirar_numero() -> int:
    return random.randint(0, 36)


def generar_corridas(cantidad_corridas: int, cantidad_tiradas: int) -> list[list[int]]:
    corridas = []
    for _ in range(cantidad_corridas):
        tiradas = []
        for _ in range(cantidad_tiradas):
            numero = tirar_numero()
            tiradas.append(numero)
        corridas.append(tiradas)
    return corridas


def frecuencias_relativa_por_corrida(
    numero_elegido: int, corrida: list[int]
) -> list[float]:
    contador = 0
    frecuencias_relativas = []
    for nro_tirada, valor in enumerate(corrida, start=1):
        if valor == numero_elegido:
            contador += 1
        frecuencias_relativas.append(contador / nro_tirada)
    return frecuencias_relativas


def promedios_por_corrida(corrida: list[int]) -> list[float]:
    suma = 0
    promedios = []
    for nro_tirada, valor in enumerate(corrida, start=1):
        suma = suma + valor
        promedios.append(suma / nro_tirada)
    return promedios


def generar_grafico_frecuencia_relativa(
    frecuencias_relativas: list[list[float]], numero_elegido: int
) -> None:
    cantidad_tiradas = len(frecuencias_relativas[0])

    for i, fr in enumerate(frecuencias_relativas):
        plt.plot(range(1, cantidad_tiradas + 1), fr, label=f"Corrida {i + 1}")

    plt.axhline(y=1 / 37, color="red", linestyle="--", label="Frecuencia Relativa Esperada")
    plt.xlabel("Número de tiradas")
    plt.ylabel("Frecuencia relativa acumulada")
    plt.title(f"Evolución de la frecuencia relativa del número {numero_elegido}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("frecuencia_relativa_x_tiradas.png")
    plt.close()


def generar_grafico_valor_promedio(promedios: list[list[float]]) -> None:
    cantidad_tiradas = len(promedios[0])

    for i, prom in enumerate(promedios):
        plt.plot(range(1, cantidad_tiradas + 1), prom, label=f"Corrida {i + 1}")

    plt.axhline(y=sum([x for x in range(0, 37)])/37, color="red", linestyle="--", label="Valor Promedio Esperado")
    plt.xlabel("Número de tiradas")
    plt.ylabel("Valor promedio acumulado")
    plt.title(f"Evolución del valor promedio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("valor_promedio_x_tiradas.png")
    plt.close()


def main(cantidad_corridas: int, cantidad_tiradas: int, numero_elegido: int) -> None:
    corridas = generar_corridas(cantidad_corridas, cantidad_tiradas)

    frecuencias_relativas = [
        frecuencias_relativa_por_corrida(numero_elegido, corrida)
        for corrida in corridas
    ]
    generar_grafico_frecuencia_relativa(frecuencias_relativas, numero_elegido)

    promedios = [
        promedio_por_corrida(corrida)
        for corrida in corridas
    ]
    generar_grafico_valor_promedio(promedios)


if __name__ == "__main__":
    args = parse_args()

    #cantidad_corridas= args.corridas
    #cantidad_tiradas= args.tirradas
    #numero_elegido= args.elegido    
    
    cantidad_corridas = 10
    cantidad_tiradas = 100
    numero_elegido = 17

    main(cantidad_corridas, cantidad_tiradas, numero_elegido)
