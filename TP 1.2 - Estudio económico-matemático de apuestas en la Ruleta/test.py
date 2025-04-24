import numpy as np
import matplotlib.pyplot as plt


# ----------------------
# Ruleta Simulada
# ----------------------

class Ruleta:
    def __init__(self):
        self.numeros = np.arange(37)
        self.color_rojo = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
        self.color_negro = {2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35}

    def girar(self):
        n = np.random.choice(self.numeros)
        if n == 0:
            color = 'verde'
        elif n in self.color_rojo:
            color = 'rojo'
        else:
            color = 'negro'
        return {'numero': n, 'color': color}


# ----------------------
# Estrategias
# ----------------------

def martingala(ruleta, n_jugadas=1000, apuesta_inicial=1, capital_inicial=1000, color='rojo', capital_infinito=True):
    capital = capital_inicial
    historial = [capital]
    apuesta = apuesta_inicial

    for _ in range(n_jugadas):
        resultado = ruleta.girar()
        if resultado['color'] == color:
            capital += apuesta
            apuesta = apuesta_inicial
        else:
            capital -= apuesta
            apuesta *= 2
            if not capital_infinito and apuesta > capital:
                historial.extend([0] * (n_jugadas - len(historial)))
                break
        historial.append(capital)
    return historial


def dalembert(ruleta, n_jugadas=1000, apuesta_inicial=1, capital_inicial=1000, color='rojo', capital_infinito=False):
    capital = capital_inicial
    historial = [capital]
    apuesta = apuesta_inicial

    for _ in range(n_jugadas):
        resultado = ruleta.girar()
        if resultado['color'] == color:
            capital += apuesta
            apuesta = max(1, apuesta - 1)
        else:
            capital -= apuesta
            apuesta += 1
            if not capital_infinito and apuesta > capital:
                historial.extend([0] * (n_jugadas - len(historial)))
                break
        historial.append(capital)
    return historial


def fibonacci(ruleta, n_jugadas=1000, capital_inicial=1000, color='rojo', capital_infinito=False):
    secuencia = [1, 1]
    capital = capital_inicial
    historial = [capital]
    indice = 1

    for _ in range(n_jugadas):
        apuesta = secuencia[indice]
        resultado = ruleta.girar()
        if resultado['color'] == color:
            capital += apuesta
            indice = max(1, indice - 2)
        else:
            capital -= apuesta
            indice += 1
            if indice >= len(secuencia):
                secuencia.append(secuencia[-1] + secuencia[-2])
            if not capital_infinito and secuencia[indice] > capital:
                historial.extend([0] * (n_jugadas - len(historial)))
                break
        historial.append(capital)
    return historial


# ----------------------
# Simulación
# ----------------------

ruleta = Ruleta()
n_jugadas = 300

hist_martingala = martingala(ruleta, n_jugadas=n_jugadas, capital_infinito=False)
hist_dalembert = dalembert(ruleta, n_jugadas=n_jugadas, capital_infinito=False)
hist_fibonacci = fibonacci(ruleta, n_jugadas=n_jugadas, capital_infinito=False)

# ----------------------
# Gráfica
# ----------------------

plt.figure(figsize=(12, 6))
plt.plot(hist_martingala, label='Martingala')
plt.plot(hist_dalembert, label='D\'Alembert')
plt.plot(hist_fibonacci, label='Fibonacci')
plt.title('Estrategias de Apuestas en Ruleta (Capital Finito)')
plt.xlabel('Número de Jugadas')
plt.ylabel('Capital')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()