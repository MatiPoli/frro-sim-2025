import random
from sympy import fibonacci

saldo = 1000
apuesta = 1
apuesta_inicial = 1
color_rojo = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}


fibonacci_secuencia = [fibonacci(i) for i in range(100)]
indice_fibonacci = 0


def tirar_numero() -> int:
    return random.randint(0, 36)

def generar_corridas(cantidad_corridas: int, cantidad_tiradas: int, estrategia: str, capiatl: str) -> list[list[int]]:
    global saldo, apuesta
    corridas = []
    for _ in range(cantidad_corridas):
        tiradas = []
        for _ in range(cantidad_tiradas):
            numero = tirar_numero()
            resultado = evaluar_apuesta_actualiza_saldo(numero)
            calcular_prox__apuesta(estrategia,resultado)
            tiradas.append(numero)
        corridas.append(tiradas)
    return corridas

def evaluar_apuesta_actualiza_saldo(numero_obtenido: int) -> int:
    global saldo, apuesta , color_rojo

    saldo = saldo - apuesta
    if numero_obtenido in color_rojo:
        saldo = saldo + apuesta*2
        return 1
    else:
        return -1
    
def calcular_prox__apuesta(estrategia: str, resultado:int) -> int:
    global apuesta, apuesta_inicial, indice_fibonacci, fibonacci_secuencia
    
    if(estrategia == "m"):
        if resultado == -1:
            apuesta = apuesta * 2

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


