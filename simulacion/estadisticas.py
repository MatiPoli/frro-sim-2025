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