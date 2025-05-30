import math
from base_generator import generar_U01


# 1. UNIFORME (continua) - Método: Transformada Inversa
def generar_uniforme(a, b):
    if a >= b:
        raise ValueError("El parámetro 'a' debe ser menor que 'b'.")
    u = generar_U01()
    return a + (b - a) * u


# 2. EXPONENCIAL (continua) - Método: Transformada Inversa
def generar_exponencial(lam):
    if lam <= 0:
        raise ValueError("El parámetro 'lam' (lambda) debe ser positivo.")
    u = generar_U01()
    if u == 0:
        return float("inf")  # Teóricamente P(X=inf)=0
    return -math.log(u) / lam


# 3. NORMAL (continua) - Método: Transformada Inversa (Box-Muller)
def generar_normal(mu, sigma):
    if sigma < 0:
        raise ValueError("Sigma (desviación estándar) debe ser no negativa.")
    if sigma == 0:
        return mu

    u1, u2 = 0, 0
    while u1 == 0:
        u1 = generar_U01()  # Evita log(0)
    u2 = generar_U01()

    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mu + sigma * z0
