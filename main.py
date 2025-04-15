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

    generar_grafico_frecuencia_relativa(corridas, numero_elegido)
    generar_grafico_valor_promedio(corridas)
    generar_grafico_desvio_estandar(corridas)
