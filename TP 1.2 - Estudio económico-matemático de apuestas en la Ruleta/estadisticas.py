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


def calcular_frecuencias_relativas(
    corridas: list[list[int]], numero_elegido: int
) -> list[list[float]]:
    return [
        frecuencias_relativa_por_corrida(numero_elegido, corrida)
        for corrida in corridas
    ]
