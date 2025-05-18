# probability_functions.py
import math
from scipy.special import comb, gammaln # comb para C(n,k), gammaln para log(Gamma(k))

# Funciones de Densidad/Probabilidad (FDP/FP) necesarias para el método de rechazo
def fdp_gamma(x, k, theta): # Para Gamma
    if x < 0: return 0
    if x == 0:
        if k == 1: return 1.0/theta if theta > 0 else float('inf') # Exponencial
        if k > 1: return 0.0
        if k < 1: return float('inf') # Tiende a infinito
    try:
        log_gamma_k = gammaln(k)
        numerador = (k-1)*math.log(x) - (x/theta)
        denominador = log_gamma_k + k*math.log(theta)
        return math.exp(numerador - denominador)
    except (ValueError, OverflowError): # ej. log(negativo) o exp(muy grande)
        return 0.0

def pmf_pascal(k_val, r_exitos, p_exito): # k_val = número de fracasos
    if k_val < 0 or not isinstance(k_val, int): return 0
    if p_exito == 1: return 1.0 if k_val == 0 else 0.0
    if p_exito <= 0 or p_exito > 1: return 0.0 # p_exito debe estar en (0,1]
    try:
        # C(k+r-1, k) * p^r * (1-p)^k  o C(k+r-1, r-1)
        if k_val + r_exitos - 1 < r_exitos -1 : return 0 # Asegura que n >= k en C(n,k)
        coef_binomial = comb(k_val + r_exitos - 1, r_exitos - 1, exact=False) # exact=False para float
        term_p = p_exito ** r_exitos
        term_1_minus_p = (1 - p_exito) ** k_val
        return coef_binomial * term_p * term_1_minus_p
    except (ValueError, OverflowError):
        return 0.0

def pmf_binomial(k_val, n_ensayos, p_exito):
    if not (0 <= k_val <= n_ensayos and isinstance(k_val, int)): return 0
    if p_exito < 0 or p_exito > 1: return 0.0
    try:
        return comb(n_ensayos, k_val, exact=False) * (p_exito ** k_val) * ((1 - p_exito) ** (n_ensayos - k_val))
    except (ValueError, OverflowError):
        return 0.0

def pmf_hipergeometrica(k_val, N_pop, K_ex_pop, n_muestra):
    if not isinstance(k_val, int): return 0
    min_k = max(0, n_muestra - (N_pop - K_ex_pop))
    max_k = min(n_muestra, K_ex_pop)
    if not (min_k <= k_val <= max_k): return 0.0
    try:
        term1 = comb(K_ex_pop, k_val, exact=False)
        term2 = comb(N_pop - K_ex_pop, n_muestra - k_val, exact=False)
        denominador = comb(N_pop, n_muestra, exact=False)
        if denominador == 0: return 0.0 # Evitar división por cero si C(N,n) es 0 (improbable con params válidos)
        return (term1 * term2) / denominador
    except (ValueError, OverflowError):
        return 0.0

def pmf_poisson(k_val, lam):
    if k_val < 0 or not isinstance(k_val, int): return 0
    if lam < 0: return 0.0
    if lam == 0: return 1.0 if k_val == 0 else 0.0
    try:
        # Usar logaritmos para estabilidad con k! grande
        log_pmf = k_val * math.log(lam) - lam - gammaln(k_val + 1)
        return math.exp(log_pmf)
    except (ValueError, OverflowError):
        return 0.0