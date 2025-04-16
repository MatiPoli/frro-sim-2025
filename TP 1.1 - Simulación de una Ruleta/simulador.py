import random


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
