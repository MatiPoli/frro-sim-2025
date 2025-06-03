# inverse_transform_generators.py
import math
from scipy.stats import norm
from base_generator import generar_U01

# --- DISTRIBUCIONES CON TRANSFORMADA INVERSA ---

# 1. UNIFORME (continua) - Método: Transformada Inversa
def generar_uniforme(a, b):
    if a >= b: raise ValueError("El parámetro 'a' debe ser menor que 'b'.")
    u = generar_U01()
    return a + (b - a) * u

# 2. EXPONENCIAL (continua) - Método: Transformada Inversa
def generar_exponencial(lam):
    if lam <= 0: raise ValueError("El parámetro 'lam' (lambda) debe ser positivo.")
    u = generar_U01()
    if u == 0: return float('inf') # Teóricamente P(X=inf)=0
    return -math.log(u) / lam

# 3. NORMAL (continua) - Método: Transformada Inversa
def generar_normal(mu, sigma):
    if sigma < 0:
        raise ValueError("Sigma (desviación estándar) debe ser no negativa.")
    if sigma == 0:
        return mu

    u = generar_U01()

    # Calcular Z ~ N(0,1) usando la inversa de la FDA (función cuantil o ppf)
    # Esto es Φ⁻¹(u)
    z0 = norm.ppf(u)

    return mu + sigma * z0