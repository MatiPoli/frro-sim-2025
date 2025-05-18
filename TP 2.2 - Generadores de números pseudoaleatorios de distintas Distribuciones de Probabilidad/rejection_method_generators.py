# rejection_method_generators.py
import random
import math
from base_generator import generar_U01
from probability_functions import (
    fdp_gamma, pmf_pascal, pmf_binomial,
    pmf_hipergeometrica, pmf_poisson
)

# --- DISTRIBUCIONES CON MÉTODO DE RECHAZO ---

# 4. GAMMA (continua) - Método: RECHAZO
def generar_gamma_rechazo(k, theta):
    """Genera Gamma(k, theta) usando el método de rechazo de Ahrens-Dieter (GS para k<1, adaptado para k>=1)."""
    if k <= 0 or theta <= 0:
        raise ValueError("k y theta deben ser positivos.")

    # Algoritmo para Gamma(k, 1)
    e_const = math.e
    b_gs = (e_const + k) / e_const
    while True:
        U1 = generar_U01()
        P = b_gs * U1
        if P <= 1:
            X = P ** (1/k)
            U2 = generar_U01()
            if U2 <= math.exp(-X):
                return X * theta # Escalar por theta
        else: # P > 1
            X = -math.log((b_gs - P) / k)
            U2 = generar_U01()
            if U2 <= X ** (k - 1):
                return X * theta # Escalar por theta

# 5. PASCAL (discreta) - Método: RECHAZO
def generar_pascal_rechazo(r_exitos, p_exito):
    if not isinstance(r_exitos, int) or r_exitos <= 0: raise ValueError("r_exitos entero > 0")
    if not (0 < p_exito <= 1): raise ValueError("p_exito en (0, 1]")
    if p_exito == 1.0: return 0

    media_k = r_exitos * (1 - p_exito) / p_exito
    if r_exitos > 1:
        modo_k = math.floor((r_exitos - 1) * (1 - p_exito) / p_exito)
    else:
        modo_k = 0
    modo_k = max(0, modo_k)

    max_f_k = pmf_pascal(modo_k, r_exitos, p_exito)
    
    k_max_estimado = 0
    temp_sum_p = 0
    limit_p = 0.9999
    k_iter = 0
    while temp_sum_p < limit_p and k_iter < (modo_k + 20 * r_exitos + 20):
        pmf_val = pmf_pascal(k_iter, r_exitos, p_exito)
        if pmf_val < 1e-12 and k_iter > modo_k + 5:
            break
        temp_sum_p += pmf_val
        k_iter += 1
    k_max_estimado = k_iter
    k_max_estimado = max(k_max_estimado, modo_k + 5, 10)

    if max_f_k == 0:
        max_f_k_temp = 0
        for k_test in range(k_max_estimado +1):
            current_pmf = pmf_pascal(k_test, r_exitos, p_exito)
            if current_pmf > max_f_k_temp:
                max_f_k_temp = current_pmf
        max_f_k = max_f_k_temp
        if max_f_k == 0: max_f_k = 1e-9

    while True:
        y_candidato = random.randint(0, k_max_estimado)
        u = generar_U01()
        f_y = pmf_pascal(y_candidato, r_exitos, p_exito)
        if u * max_f_k <= f_y:
            return y_candidato

# 6. BINOMIAL (discreta) - Método: RECHAZO
def generar_binomial_rechazo(n_ensayos, p_exito):
    if not isinstance(n_ensayos, int) or n_ensayos < 0: raise ValueError("n_ensayos entero >= 0")
    if not (0 <= p_exito <= 1): raise ValueError("p_exito en [0, 1]")
    if n_ensayos == 0: return 0

    modo = math.floor((n_ensayos + 1) * p_exito)
    modo = max(0, min(n_ensayos, modo))
    max_f_k = pmf_binomial(modo, n_ensayos, p_exito)

    if max_f_k == 0:
        if p_exito == 0: return 0
        if p_exito == 1: return n_ensayos
        max_f_k = 1e-9

    while True:
        y_candidato = random.randint(0, n_ensayos)
        u = generar_U01()
        f_y = pmf_binomial(y_candidato, n_ensayos, p_exito)
        if u * max_f_k <= f_y:
            return y_candidato

# 7. HIPERGEOMÉTRICA (discreta) - Método: RECHAZO
def generar_hipergeometrica_rechazo(N_pop, K_ex_pop, n_muestra):
    if not all(isinstance(x, int) for x in [N_pop, K_ex_pop, n_muestra]): raise ValueError("Params enteros")
    if not (0 <= K_ex_pop <= N_pop and 0 <= n_muestra <= N_pop): raise ValueError("Params inconsistentes")
    if n_muestra == 0: return 0

    k_min = max(0, n_muestra - (N_pop - K_ex_pop))
    k_max = min(n_muestra, K_ex_pop)

    if k_min > k_max:
        print(f"Advertencia Hiper: k_min {k_min} > k_max {k_max}. Params: N={N_pop}, K={K_ex_pop}, n={n_muestra}")
        return k_min 

    modo_aprox = math.floor((n_muestra + 1) * (K_ex_pop + 1) / (N_pop + 2))
    modo = max(k_min, min(k_max, modo_aprox))
    max_f_k = pmf_hipergeometrica(modo, N_pop, K_ex_pop, n_muestra)

    if max_f_k == 0:
        max_f_k_temp = 0
        for k_test in range(k_min, k_max + 1):
            current_pmf = pmf_hipergeometrica(k_test, N_pop, K_ex_pop, n_muestra)
            if current_pmf > max_f_k_temp:
                max_f_k_temp = current_pmf
        max_f_k = max_f_k_temp
        if max_f_k == 0:
            if k_min == k_max and pmf_hipergeometrica(k_min, N_pop,K_ex_pop,n_muestra) > 0:
                 max_f_k = pmf_hipergeometrica(k_min, N_pop,K_ex_pop,n_muestra)
            else:
                 max_f_k = 1e-9

    while True:
        if k_min > k_max :
            return k_min

        y_candidato = random.randint(k_min, k_max)
        u = generar_U01()
        f_y = pmf_hipergeometrica(y_candidato, N_pop, K_ex_pop, n_muestra)
        if u * max_f_k <= f_y:
            return y_candidato

# 8. POISSON (discreta) - Método: RECHAZO
def generar_poisson_rechazo(lam):
    if lam < 0: raise ValueError("lambda >= 0")
    if lam == 0: return 0

    modo = math.floor(lam)
    max_f_k = pmf_poisson(modo, lam)
    if lam > 0 and lam == modo :
         max_f_k = max(max_f_k, pmf_poisson(modo-1, lam))

    k_max_estimado = 0
    temp_sum_p = 0
    limit_p = 0.99999
    k_iter = 0
    iter_limit_k = math.ceil(modo + 10 * math.sqrt(lam) + 20 if lam >0 else 20)

    while temp_sum_p < limit_p and k_iter < iter_limit_k:
        pmf_val = pmf_poisson(k_iter, lam)
        if pmf_val < 1e-12 and k_iter > modo + 5 :
            break
        temp_sum_p += pmf_val
        k_iter += 1
    k_max_estimado = max(k_iter, modo + 5, 5)

    if max_f_k == 0 and lam > 0 :
        max_f_k_temp = 0
        for k_t in range(k_max_estimado +1):
            curr_pmf = pmf_poisson(k_t, lam)
            if curr_pmf > max_f_k_temp: max_f_k_temp = curr_pmf
        max_f_k = max_f_k_temp
        if max_f_k == 0: max_f_k = 1e-9

    while True:
        y_candidato = random.randint(0, k_max_estimado)
        u = generar_U01()
        f_y = pmf_poisson(y_candidato, lam)
        if u * max_f_k <= f_y:
            return y_candidato

# 9. EMPÍRICA DISCRETA (discreta) - Método: RECHAZO
def generar_empirica_discreta_rechazo(valores, probabilidades):
    if len(valores) != len(probabilidades): raise ValueError("Longitudes no coinciden")
    if not math.isclose(sum(probabilidades), 1.0, abs_tol=1e-9):
        print(f"Advertencia Empírica: Suma de probs es {sum(probabilidades)}")
    
    m = len(valores)
    if m == 0: raise ValueError("Listas vacías")

    max_p_i = 0.0
    if any(p > 0 for p in probabilidades):
        max_p_i = max(p for p in probabilidades if p > 0)
    
    if max_p_i == 0:
        if m == 1: return valores[0]
        print("Advertencia Empírica: Todas las probabilidades son <= 0. Devolviendo el primer valor.")
        return valores[0]

    while True:
        idx_candidato = random.randint(0, m - 1)
        valor_candidato = valores[idx_candidato]
        prob_candidato = probabilidades[idx_candidato]
        
        u = generar_U01()
        if u * max_p_i <= prob_candidato:
            return valor_candidato