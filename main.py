import math
import sys
import random
import argparse 
import matplotlib.pyplot as plt
import statistics as st

from argumentos import parse_args
from simulacion.estadisticas import frecuencias_relativa_por_corrida, promedios_por_corrida
from simulacion.graficos import generar_grafico_desvio_estandar, generar_grafico_frecuencia_relativa, generar_grafico_valor_promedio
from simulacion.simulador import simular_corridas



if __name__ == "__main__":
    args = parse_args()

    #cantidad_corridas= args.corridas
    #cantidad_tiradas= args.tirradas
    #numero_elegido= args.elegido    
    
    cantidad_corridas = 10
    cantidad_tiradas = 1000
    numero_elegido = 17

    corridas = simular_corridas(cantidad_corridas, cantidad_tiradas)

    frecuencias_relativas = [
        frecuencias_relativa_por_corrida(numero_elegido, corrida)
        for corrida in corridas
    ]
    generar_grafico_frecuencia_relativa(frecuencias_relativas, numero_elegido)

    promedios = [
        promedios_por_corrida(corrida)
        for corrida in corridas
    ]
    generar_grafico_valor_promedio(promedios)
    generar_grafico_desvio_estandar(corridas)
