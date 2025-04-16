import argparse

from simulador import generar_corridas
from estadisticas import (
    calcular_promedios,
    calcular_varianzas,
    calcular_desvios_estandar,
    calcular_frecuencias_relativas,
)
from graficos import (
    generar_grafico_frecuencia_por_numero,
    generar_grafico_varianza,
    generar_grafico_valor_promedio,
    generar_grafico_desvio_estandar,
    generar_grafico_frecuencia_relativa,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulación de ruleta",
    )
    parser.add_argument(
        "-c", "--corridas", type=int, required=True, help="Cantidad de corridas"
    )
    parser.add_argument(
        "-n",
        "--tiradas",
        type=int,
        required=True,
        help="Cantidad de tiradas por corrida",
    )
    parser.add_argument(
        "-e",
        "--elegido",
        type=int,
        required=True,
        choices=range(0, 37),
        help="Número elegido (entre 0 y 36)",
    )
    return parser.parse_args()


def main(cantidad_corridas: int, cantidad_tiradas: int, numero_elegido: int) -> None:
    corridas = generar_corridas(cantidad_corridas, cantidad_tiradas)

    frecuencias_relativas = calcular_frecuencias_relativas(corridas, numero_elegido)
    promedios = calcular_promedios(corridas)
    desvios_estandar = calcular_desvios_estandar(corridas)
    varianzas = calcular_varianzas(corridas)

    generar_grafico_varianza(varianzas)
    generar_grafico_valor_promedio(promedios)
    generar_grafico_desvio_estandar(desvios_estandar)
    generar_grafico_frecuencia_relativa(frecuencias_relativas, numero_elegido)
    generar_grafico_frecuencia_por_numero(corridas[0])

if __name__ == "__main__":
    args = parse_args()

    cantidad_corridas = args.corridas
    cantidad_tiradas = args.tiradas
    numero_elegido = args.elegido

    main(cantidad_corridas, cantidad_tiradas, numero_elegido)
