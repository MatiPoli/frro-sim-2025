import random
from sympy import fibonacci

saldo = 100
apuesta = 1
apuesta_inicial = 1
color_rojo = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}


fibonacci_secuencia = [fibonacci(i) for i in range(100)]
indice_fibonacci = 0


def tirar_numero() -> int:
    return random.randint(0, 36)

def generar_corridas(cantidad_corridas: int, cantidad_tiradas: int, estrategia: str, capital: str) -> tuple[list[list[int]], list[int], list[list[float]]]:
    global saldo, apuesta
    corridas = []
    saldo_por_tirada = []  # Lista para registrar el saldo después de cada tirada
    frecuencias_relativas = []  # Lista para registrar las frecuencias relativas acumuladas

    # Registrar el saldo inicial
    saldo_por_tirada.append(saldo)

    for _ in range(cantidad_corridas):
        tiradas = []
        exitos = 0  # Contador de éxitos (números rojos)
        frecuencias_corrida = []  # Frecuencias relativas para esta corrida

        for tirada in range(1, cantidad_tiradas + 1):
            numero = tirar_numero()
            resultado = evaluar_apuesta_actualiza_saldo(numero, capital)

            # Si el saldo es insuficiente, detener la simulación
            if capital == "f" and resultado == 0:
                break

            calcular_prox__apuesta(estrategia, resultado)
            tiradas.append(numero)
            saldo_por_tirada.append(saldo)  # Registrar el saldo actual

            # Verificar si el número es rojo (apuesta favorable)
            if numero in color_rojo:
                exitos += 1

            # Calcular la frecuencia relativa acumulada
            frecuencia_relativa = exitos / tirada
            frecuencias_corrida.append(frecuencia_relativa)

            print(f"Tirada: {tirada}, Número: {numero}, Saldo: {saldo}, Apuesta: {apuesta}, Frecuencia Relativa: {frecuencia_relativa}")

        corridas.append(tiradas)
        frecuencias_relativas.append(frecuencias_corrida)

    return corridas, saldo_por_tirada, frecuencias_relativas

def evaluar_apuesta_actualiza_saldo(numero_obtenido: int, capital: str) -> int:
    global saldo, apuesta, color_rojo

    # Verificar si el saldo es suficiente para apostar
    if capital == "f" and saldo < apuesta:
        print("Saldo insuficiente para continuar apostando.")
        return 0  # Indica que no se puede continuar

    saldo = saldo - apuesta
    if numero_obtenido in color_rojo:
        saldo = saldo + apuesta * 2
        return 1  # Gana
    else:
        return -1  # Pierde
    
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

    # Otra estrategia: Paroli
    elif estrategia == "o":  
        if resultado == 1:  # Si gana, duplica la apuesta
            apuesta = apuesta * 2
        else:  # Si pierde, vuelve a la apuesta inicial
            apuesta = apuesta_inicial


