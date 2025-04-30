import random
from sympy import fibonacci

saldo = 10000
apuesta = 100
apuesta_inicial = 1
color_rojo = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
cantidad_bancarrotas = 0

fibonacci_secuencia = [fibonacci(i) for i in range(100)]
indice_fibonacci = 0


def tirar_numero() -> int:
    return random.randint(0, 36)

def generar_corridas(cantidad_corridas: int, cantidad_tiradas: int, estrategia: str, capital: str) -> tuple[list[list[int]], list[int], list[list[float]]]:
    corridas = []
    corridas_saldo = []  
    for _ in range(cantidad_corridas):
        saldo = 1000
        apuesta = 10
        tiradas = []
        saldo_por_tirada = [saldo]

        for tirada in range(cantidad_tiradas):
            numero = tirar_numero()
            if saldo < apuesta:
                #saldo = 0
                #saldo_por_tirada.append(saldo)
            else:
                if numero in color_rojo:
                    saldo += apuesta
                    resultado = 1
                    tiradas.append("rojo")
                else:
                    saldo -= apuesta
                    resultado = -1
                    tiradas.append("negro")

                saldo_por_tirada.append(saldo)
                proxima_apuesta = calcular_prox_apuesta(estrategia, resultado, apuesta)
                apuesta = proxima_apuesta
        corridas.append(tiradas)
        corridas_saldo.append(saldo_por_tirada)

    return corridas, corridas_saldo
    

def calcular_prox_apuesta(estrategia: str, resultado:int, ultima_apuesta):
#    global apuesta, apuesta_inicial, indice_fibonacci, fibonacci_secuencia
    apuesta_inicial = 10
    if(estrategia == "m"):
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
        


