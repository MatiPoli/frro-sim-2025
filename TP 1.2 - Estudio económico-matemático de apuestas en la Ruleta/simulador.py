import random
from sympy import fibonacci

color_rojo = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
cantidad_bancarrotas = 0

fibonacci_secuencia = [fibonacci(i) for i in range(100)]
indice_fibonacci = 0


def tirar_numero() -> int:
    return random.randint(0, 36)


def generar_corridas(
    cantidad_corridas: int, cantidad_tiradas: int, estrategia: str, capital: str
) -> list[list[int]]:
    saldo_inicial = 1000
    apuesta_inicial = 100
    corridas_saldo = []

    for _ in range(cantidad_corridas):
        saldo = saldo_inicial
        apuesta = apuesta_inicial

        saldo_por_tirada = [saldo]

        for _ in range(cantidad_tiradas):
            numero = tirar_numero()
            if saldo < apuesta and capital == "f":
                break
            else:
                if numero in color_rojo:
                    saldo += apuesta
                    resultado = 1
                else:
                    saldo -= apuesta
                    resultado = -1

                saldo_por_tirada.append(saldo)
                proxima_apuesta = calcular_prox_apuesta(
                    apuesta_inicial, estrategia, resultado, apuesta
                )
                apuesta = proxima_apuesta
        corridas_saldo.append(saldo_por_tirada)

    return corridas_saldo


def calcular_prox_apuesta(
    apuesta_inicial, estrategia: str, resultado: int, ultima_apuesta
):
    if estrategia == "m":
        if resultado == -1:
            return ultima_apuesta * 2
        elif resultado == 1:
            return apuesta_inicial


"""
    elif(estrategia == "d"):
        if resultado == -1:
            apuesta = apuesta + apuesta_inicial
        else:
            apuesta =apuesta -apuesta_inicial
    
    elif(estrategia == "f"):
        if resultado == -1:
            apuesta = apuesta_inicial * fibonacci_secuencia[indice_fibonacci]
            indice_fibonacci += 1
        else:
            if indice_fibonacci > 1:
                indice_fibonacci -= 2
            apuesta = apuesta_inicial * fibonacci_secuencia[indice_fibonacci]

    # Otra estrategia: Paroli
    elif estrategia == "o":  
        if resultado == 1:  # Si gana, duplica la apuesta
            apuesta = apuesta * 2
        else:  # Si pierde, vuelve a la apuesta inicial
            apuesta = apuesta_inicial
"""
