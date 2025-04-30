import argparse

from simulador import generar_corridas

from graficos import (
    generar_grafico_saldos,
    generar_grafico_frecuencia_apuesta_favorable,
)

from estadisticas import calcular_frecuencias_relativas
import os

if not os.path.exists("Graficos"):
    os.makedirs("Graficos")


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
        "-s",
        "--estrategia",
        type=str,
        required=True,
        choices=["m", "d", "f", "o"],
        default=None,
        help="Tipo de estrategia a utilizar: martingala (m), d'Alembert (d), Fibonacci (f) u otra (o)",
    )
    parser.add_argument(
        "-a",
        "--capital",
        type=str,
        required=True,
        choices=["i", "f"],
        default=None,
        help="Tipo de capital a utilizar: infinito (i) o finito (f)",
    )

    return parser.parse_args()


def main(
    cantidad_corridas: int, cantidad_tiradas: int, estrategia: str, capital: str
) -> None:
    corridas_saldo, corridas = generar_corridas(
        cantidad_corridas, cantidad_tiradas, estrategia, capital
    )
    generar_grafico_saldos(corridas_saldo, capital)

    # ------#
    frecuencias_relativas = calcular_frecuencias_relativas(corridas, 1)
    generar_grafico_frecuencia_apuesta_favorable(frecuencias_relativas)
    print(corridas)


if __name__ == "__main__":
    args = parse_args()

    cantidad_corridas = args.corridas
    cantidad_tiradas = args.tiradas
    estrategia = args.estrategia
    capital = args.capital

    main(cantidad_corridas, cantidad_tiradas, estrategia, capital)
