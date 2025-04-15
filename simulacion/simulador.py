import random

def simular_tirada() -> int:
    return random.randint(0, 36)


def simular_corridas(cantidad_corridas: int, cantidad_tiradas: int) -> list[list[int]]:
    corridas = []
    for _ in range(cantidad_corridas):
        tiradas = []
        for _ in range(cantidad_tiradas):
            numero = simular_tirada()
            tiradas.append(numero)
        corridas.append(tiradas)
    return corridas