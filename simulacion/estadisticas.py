import statistics

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


def calcular_promedios(corridas: list[list[int]]) -> list[list[float]]:
    return [promedios_por_corrida(corrida) for corrida in corridas]


def calcular_frecuencias_relativas(corridas: list[list[int]], numero_elegido: int) -> list[list[float]]:
    return [frecuencias_relativa_por_corrida(numero_elegido, corrida) for corrida in corridas]


def desvios_estandar_por_corrida(corrida: list[int]) -> list[float]:
    return [
        statistics.stdev(corrida[:i]) if i > 1 else 0
        for i in range(1, len(corrida) + 1)
    ]


def calcular_desvios_estandar(corridas: list[list[int]]) -> list[list[float]]:
    return [desvios_estandar_por_corrida(corrida) for corrida in corridas]