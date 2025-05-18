import random
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform, expon, gamma, norm, nbinom, binom, hypergeom, poisson # Para FDP/FP teóricas en tests
from scipy.special import comb, gammaln # comb para C(n,k), gammaln para log(Gamma(k))

# -----------------------------------------------------------------------------
# GENERADOR BASE U(0,1)
# -----------------------------------------------------------------------------
def generar_U01():
    return random.random()

# -----------------------------------------------------------------------------
# GENERADORES DE DISTRIBUCIONES
# -----------------------------------------------------------------------------

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

# 3. NORMAL (continua) - Método: Transformada Inversa (Box-Muller)
def generar_normal(mu, sigma):
    if sigma < 0: raise ValueError("Sigma (desviación estándar) debe ser no negativa.")
    if sigma == 0: return mu

    u1, u2 = 0, 0
    while u1 == 0: u1 = generar_U01() # Evita log(0)
    u2 = generar_U01()

    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mu + sigma * z0

# --- DISTRIBUCIONES CON MÉTODO DE RECHAZO ---

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

# 4. GAMMA (continua) - Método: RECHAZO
# Para Gamma(k,theta), generamos Gamma(k,1) y luego multiplicamos por theta.
# Usaremos el algoritmo de rechazo de Ahrens y Dieter (1974) para Gamma(alpha,1) donde alpha=k.
# Este es un algoritmo específico, no uno genérico f(x) <= c*g(x) fácilmente derivable.
# Este algoritmo es eficiente para k > 0. Se divide en casos k < 1 y k >= 1.
def generar_gamma_rechazo(k, theta):
    """Genera Gamma(k, theta) usando el método de rechazo de Ahrens-Dieter (GS para k<1, adaptado para k>=1)."""
    if k <= 0 or theta <= 0:
        raise ValueError("k y theta deben ser positivos.")

    # Algoritmo para Gamma(k, 1)

    # Algoritmo GS (Ahrens y Dieter, 1974)
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

    # Estimar k_max (número de fracasos)
    media_k = r_exitos * (1 - p_exito) / p_exito
    # Para k_max_estimado, tomar media + ~5-7 desviaciones estándar
    # std_dev_k = math.sqrt(r_exitos * (1 - p_exito) / (p_exito**2))
    # k_max_estimado = math.ceil(media_k + 7 * std_dev_k if std_dev_k > 0 else media_k + 10*r_exitos )
    # k_max_estimado = max(k_max_estimado, r_exitos, 10) # Asegurar un mínimo razonable

    # Encontrar el modo para estimar max_f_k
    # Modo de los fracasos: floor((r-1)(1-p)/p) si r>1; 0 si r=1.
    if r_exitos > 1:
        modo_k = math.floor((r_exitos - 1) * (1 - p_exito) / p_exito)
    else: # r_exitos == 1 (Geométrica)
        modo_k = 0 # El valor más probable es 0 fracasos
    modo_k = max(0, modo_k) # Asegurar no negativo

    max_f_k = pmf_pascal(modo_k, r_exitos, p_exito)
    
    # k_max_estimado basado en donde la pmf es muy pequeña
    # Percentil 99.99%
    # Si nbinom.ppf no está disponible o para no usar scipy.stats aquí:
    k_max_estimado = 0
    temp_sum_p = 0
    limit_p = 0.9999
    k_iter = 0
    while temp_sum_p < limit_p and k_iter < (modo_k + 20 * r_exitos + 20): # Límite de iteraciones
        pmf_val = pmf_pascal(k_iter, r_exitos, p_exito)
        if pmf_val < 1e-12 and k_iter > modo_k + 5: # Si la PMF es muy baja y estamos lejos del modo
            break
        temp_sum_p += pmf_val
        k_iter += 1
    k_max_estimado = k_iter
    k_max_estimado = max(k_max_estimado, modo_k + 5, 10)


    if max_f_k == 0: # Si la PMF en el modo es 0 (improbable para params válidos)
        # Buscar el máximo iterando un poco.
        max_f_k_temp = 0
        for k_test in range(k_max_estimado +1):
            current_pmf = pmf_pascal(k_test, r_exitos, p_exito)
            if current_pmf > max_f_k_temp:
                max_f_k_temp = current_pmf
        max_f_k = max_f_k_temp
        if max_f_k == 0: max_f_k = 1e-9 # Fallback para evitar div por cero

    while True:
        y_candidato = random.randint(0, k_max_estimado) # Propuesta g(k) ~ U_discreta[0, k_max_estimado]
        u = generar_U01()
        f_y = pmf_pascal(y_candidato, r_exitos, p_exito)
        if u * max_f_k <= f_y: # Aceptar si u <= f(Y) / (c*g(Y)), con c*g(Y) = max_f_k
            return y_candidato

# 6. BINOMIAL (discreta) - Método: RECHAZO
def generar_binomial_rechazo(n_ensayos, p_exito):
    if not isinstance(n_ensayos, int) or n_ensayos < 0: raise ValueError("n_ensayos entero >= 0")
    if not (0 <= p_exito <= 1): raise ValueError("p_exito en [0, 1]")
    if n_ensayos == 0: return 0

    modo = math.floor((n_ensayos + 1) * p_exito)
    modo = max(0, min(n_ensayos, modo)) # Asegurar que el modo esté en [0, n_ensayos]
    max_f_k = pmf_binomial(modo, n_ensayos, p_exito)

    if max_f_k == 0: # Casos extremos p=0 o p=1, o n=0
        if p_exito == 0: return 0
        if p_exito == 1: return n_ensayos
        max_f_k = 1e-9 # Fallback

    # g(k) es U_discreta[0, n_ensayos]
    # c*g(k) = max_f_k (aproximadamente, si g(k) = 1/(n+1) y c = (n+1)max_f_k)
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

    if k_min > k_max: # No hay valores posibles, puede pasar si n_muestra es muy grande/pequeño
        # ej: N=10, K=2, n=5. k_min = max(0, 5-(10-2)=5-8=-3)=0. k_max=min(5,2)=2. Rango [0,2]
        # ej: N=10, K=8, n=5. k_min = max(0, 5-(10-8)=5-2=3)=3. k_max=min(5,8)=5. Rango [3,5]
        # Si k_min > k_max, significa que la situación es imposible.
        # Devuelve un valor fuera de rango o lanza error, o devuelve el límite más cercano.
        # Esto no debería pasar si los parámetros iniciales son lógicos para una hipergeométrica.
        # Si pasa, es probable que la PMF sea 0 en todo el rango.
        # Por seguridad, si k_min > k_max, devolvemos k_min (puede ser 0)
        # o el valor más probable que sería uno de los extremos si el rango se forzara.
        # Sin embargo, si pmf_hipergeométrica maneja bien estos casos, debería dar 0.
        print(f"Advertencia Hiper: k_min {k_min} > k_max {k_max}. Params: N={N_pop}, K={K_ex_pop}, n={n_muestra}")
        return k_min # O el valor que tenga probabilidad no nula, que podría ser ninguno.

    # Encontrar max_f_k. Modo de la hipergeométrica: floor((n+1)(K+1)/(N+2))
    modo_aprox = math.floor((n_muestra + 1) * (K_ex_pop + 1) / (N_pop + 2))
    modo = max(k_min, min(k_max, modo_aprox)) # Ajustar al rango válido
    max_f_k = pmf_hipergeometrica(modo, N_pop, K_ex_pop, n_muestra)

    if max_f_k == 0: # Si el modo da 0, buscar el máximo en el rango
        max_f_k_temp = 0
        for k_test in range(k_min, k_max + 1):
            current_pmf = pmf_hipergeometrica(k_test, N_pop, K_ex_pop, n_muestra)
            if current_pmf > max_f_k_temp:
                max_f_k_temp = current_pmf
        max_f_k = max_f_k_temp
        if max_f_k == 0: # Si sigue siendo 0, la distribución es 0 en todo el rango (o casi)
            if k_min == k_max and pmf_hipergeometrica(k_min, N_pop,K_ex_pop,n_muestra) > 0: # Caso de un solo punto
                 max_f_k = pmf_hipergeometrica(k_min, N_pop,K_ex_pop,n_muestra)
            else:
                 max_f_k = 1e-9 # Fallback

    # g(k) es U_discreta[k_min, k_max]
    while True:
        if k_min > k_max : # Seguridad adicional
            return k_min # No hay rango para generar

        y_candidato = random.randint(k_min, k_max)
        u = generar_U01()
        f_y = pmf_hipergeometrica(y_candidato, N_pop, K_ex_pop, n_muestra)
        if u * max_f_k <= f_y:
            return y_candidato

# 8. POISSON (discreta) - Método: RECHAZO
def generar_poisson_rechazo(lam):
    if lam < 0: raise ValueError("lambda >= 0")
    if lam == 0: return 0

    # Estimar k_max. Modo es floor(lambda).
    modo = math.floor(lam)
    max_f_k = pmf_poisson(modo, lam)
    if lam > 0 and lam == modo : # Si lambda es entero, pmf(lambda) y pmf(lambda-1) son máximos
         max_f_k = max(max_f_k, pmf_poisson(modo-1, lam))

    # k_max_estimado: donde la PMF sea muy pequeña
    k_max_estimado = 0
    temp_sum_p = 0
    limit_p = 0.99999
    k_iter = 0
    # Límite superior práctico para k_iter, por ejemplo, modo + 10*sqrt(lambda) + 10
    iter_limit_k = math.ceil(modo + 10 * math.sqrt(lam) + 20 if lam >0 else 20)

    while temp_sum_p < limit_p and k_iter < iter_limit_k:
        pmf_val = pmf_poisson(k_iter, lam)
        if pmf_val < 1e-12 and k_iter > modo + 5 : # PMF muy baja y lejos del modo
            break
        temp_sum_p += pmf_val
        k_iter += 1
    k_max_estimado = max(k_iter, modo + 5, 5) # Asegurar un rango mínimo

    if max_f_k == 0 and lam > 0 :
        max_f_k_temp = 0
        for k_t in range(k_max_estimado +1):
            curr_pmf = pmf_poisson(k_t, lam)
            if curr_pmf > max_f_k_temp: max_f_k_temp = curr_pmf
        max_f_k = max_f_k_temp
        if max_f_k == 0: max_f_k = 1e-9


    # g(k) es U_discreta[0, k_max_estimado]
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
        max_p_i = max(p for p in probabilidades if p > 0) # Máximo de las probabilidades positivas
    
    if max_p_i == 0: # Todas las probabilidades son 0 o negativas (lo cual es un error)
        # Si solo hay un valor, ese es. Si hay muchos y todas probs 0, es ambiguo.
        if m == 1: return valores[0]
        print("Advertencia Empírica: Todas las probabilidades son <= 0. Devolviendo el primer valor.")
        return valores[0] # O lanzar error.

    # g(v_i) es U_discreta sobre los índices [0, m-1]
    while True:
        idx_candidato = random.randint(0, m - 1)
        valor_candidato = valores[idx_candidato]
        prob_candidato = probabilidades[idx_candidato]
        
        u = generar_U01()
        # Aceptar si u <= f(Y) / (c*g(Y)) donde c*g(Y) = max_p_i
        if u * max_p_i <= prob_candidato:
            return valor_candidato

# -----------------------------------------------------------------------------
# FUNCIONES DE TESTEO (SIN CAMBIOS MAYORES, PERO SE ADAPTA A LA LÓGICA DE RECHAZO)
# -----------------------------------------------------------------------------
def testear_distribucion(nombre_dist, generador_func, params_dist, teor_media_func, teor_var_func,
                         scipy_dist_func_pdf_pmf, N_muestras=10000, es_discreta=False,
                         rango_grafica_teorica=None, usa_rechazo=False):
    print(f"\n--- Testeando Distribución: {nombre_dist} con parámetros {params_dist} {'(Rechazo)' if usa_rechazo else '(T.Inversa)'} ---")
    
    muestras = []
    intentos_totales = 0 # Solo para rechazo
    fallos_generacion_muestra = 0

    for i in range(N_muestras):
        muestra_generada = None
        intentos_muestra_actual = 0
        max_intentos_por_muestra = 1000 # Límite por si el rechazo es muy malo para una muestra específica

        while muestra_generada is None and intentos_muestra_actual < max_intentos_por_muestra:
            try:
                muestra_generada = generador_func(*params_dist)
                if usa_rechazo: intentos_totales += 1 # Contar el intento exitoso
            except NotImplementedError as nie: # Para Gamma no implementada
                print(f"OMITIDO Test {nombre_dist}: {nie}")
                return
            except Exception as e:
                print(f"Error generando muestra {i+1}/{N_muestras} para {nombre_dist}: {e}")
                fallos_generacion_muestra +=1
                if fallos_generacion_muestra > N_muestras / 10 : # Si >10% de las muestras fallan en general
                    print(f"Demasiados fallos generales ({fallos_generacion_muestra}) generando {nombre_dist}. Abortando test.")
                    return
                break # Salir del while interno para esta muestra, pasar a la siguiente.
            
            intentos_muestra_actual += 1
            if usa_rechazo and muestra_generada is None: # Si es rechazo y no se generó (porque u > f/cg)
                 intentos_totales += 1 # Contar el intento fallido (rechazado)

        if muestra_generada is not None:
            muestras.append(muestra_generada)
        elif intentos_muestra_actual >= max_intentos_por_muestra:
            print(f"  Advertencia: No se pudo generar la muestra {i+1} para {nombre_dist} después de {max_intentos_por_muestra} intentos.")

    if not muestras:
        print(f"No se generaron muestras válidas para {nombre_dist}. Abortando test.")
        return

    if usa_rechazo and intentos_totales > 0:
        tasa_aceptacion_aprox = len(muestras) / intentos_totales
        print(f"  Tasa de aceptación aprox. (Rechazo): {tasa_aceptacion_aprox:.4f} (Muestras={len(muestras)}, Intentos_totales={intentos_totales})")
    elif usa_rechazo:
        print(f"  No se registraron intentos para el método de rechazo (Muestras={len(muestras)}).")


    media_muestral = np.mean(muestras)
    var_muestral = np.var(muestras, ddof=1) if len(muestras) > 1 else 0

    # Capturar errores en cálculo de teóricas por si los params son problemáticos
    try:
        media_teorica = teor_media_func(*params_dist)
        var_teorica = teor_var_func(*params_dist)
        print(f"  Media Muestral: {media_muestral:.4f} | Media Teórica: {media_teorica:.4f}")
        print(f"  Varianza Muestral: {var_muestral:.4f} | Varianza Teórica: {var_teorica:.4f}")
    except Exception as e_teor:
        print(f"Error calculando momentos teóricos para {nombre_dist} {params_dist}: {e_teor}")
        media_teorica, var_teorica = float('nan'), float('nan')


    plt.figure(figsize=(10, 6))
    # ... (resto de la lógica de ploteo sin cambios significativos) ...
    if es_discreta:
        val_unicos, conteos = np.unique(muestras, return_counts=True)
        frecuencias_relativas = conteos / len(muestras)
        plt.bar(val_unicos, frecuencias_relativas, width=0.8 if len(val_unicos) > 1 else 0.1,
                alpha=0.7, label=f'Muestras (N={len(muestras)})', color='skyblue')
        if rango_grafica_teorica:
            x_teorico = np.arange(rango_grafica_teorica[0], rango_grafica_teorica[1] + 1)
        else:
            min_val_obs = min(val_unicos) if len(val_unicos)>0 else 0
            max_val_obs = max(val_unicos) if len(val_unicos)>0 else 1
            # Asegurar que el rango teórico cubra al menos el observado
            x_teorico = np.arange(min(min_val_obs,0), max_val_obs + 1)


        try:
            y_teorico_pmf = scipy_dist_func_pdf_pmf(x_teorico, *params_dist)
            plt.stem(x_teorico, y_teorico_pmf, linefmt='r-', markerfmt='ro', basefmt=" ",
                    label='PMF Teórica', use_line_collection=True)
            if len(x_teorico) < 20: plt.xticks(x_teorico)
            elif len(x_teorico) > 0 : # Ticks más espaciados si hay muchos valores
                 tick_step = max(1, len(x_teorico) // 10)
                 plt.xticks(x_teorico[::tick_step])

        except Exception as e_plot_teor:
            print(f"Error graficando PMF teórica para {nombre_dist} {params_dist}: {e_plot_teor}")

    else: # Continua
        plt.hist(muestras, bins='auto', density=True, alpha=0.7,
                 label=f'Muestras (N={len(muestras)})', color='skyblue')
        if rango_grafica_teorica:
            x_teorico = np.linspace(rango_grafica_teorica[0], rango_grafica_teorica[1], 200)
        else:
            min_val_obs = min(muestras) if len(muestras)>0 else 0
            max_val_obs = max(muestras) if len(muestras)>0 else 1
            # Extender un poco el rango para la gráfica teórica
            plot_min = min_val_obs - 0.1 * abs(max_val_obs - min_val_obs) if max_val_obs != min_val_obs else min_val_obs -1
            plot_max = max_val_obs + 0.1 * abs(max_val_obs - min_val_obs) if max_val_obs != min_val_obs else max_val_obs +1
            x_teorico = np.linspace(plot_min, plot_max, 200)
        try:
            y_teorico_fdp = scipy_dist_func_pdf_pmf(x_teorico, *params_dist)
            plt.plot(x_teorico, y_teorico_fdp, 'r-', lw=2, label='FDP Teórica')
        except Exception as e_plot_teor:
            print(f"Error graficando FDP teórica para {nombre_dist} {params_dist}: {e_plot_teor}")


    plt.title(f'Distribución {nombre_dist}{params_dist} {"(Rechazo)" if usa_rechazo else "(T.Inversa)"}')
    plt.xlabel('Valor')
    plt.ylabel('Densidad / Probabilidad')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# -----------------------------------------------------------------------------
# EJECUCIÓN DE TESTS
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    N_GLOBAL = 5000 # Reducido por si los rechazos son lentos

    # --- Test Uniforme (T. Inversa) ---
    a_unif, b_unif = 2, 10
    testear_distribucion( "Uniforme", generar_uniforme, (a_unif, b_unif),
        lambda a,b: (a+b)/2, lambda a,b: ((b-a)**2)/12,
        lambda x,a,b: uniform.pdf(x, loc=a, scale=b-a),
        N_muestras=N_GLOBAL, es_discreta=False, usa_rechazo=False,
        rango_grafica_teorica=(a_unif -1, b_unif +1))

    # --- Test Exponencial (T. Inversa) ---
    lam_exp = 0.5
    testear_distribucion("Exponencial", generar_exponencial, (lam_exp,),
        lambda l:1/l, lambda l:1/(l**2),
        lambda x,l: expon.pdf(x, scale=1/l),
        N_muestras=N_GLOBAL, es_discreta=False, usa_rechazo=False,
        rango_grafica_teorica=(0, expon.ppf(0.999, scale=1/lam_exp)))

    # --- Test Normal (T. Inversa - Box Muller) ---
    mu_norm, sigma_norm = 5, 2
    testear_distribucion("Normal", generar_normal, (mu_norm, sigma_norm),
        lambda mu,sig:mu, lambda mu,sig:sig**2,
        lambda x,mu,sig: norm.pdf(x, loc=mu, scale=sig),
        N_muestras=N_GLOBAL, es_discreta=False, usa_rechazo=False,
        rango_grafica_teorica=(norm.ppf(0.001, mu_norm, sigma_norm), norm.ppf(0.999, mu_norm, sigma_norm)))

    print("\n--- Iniciando tests para distribuciones con MÉTODO DE RECHAZO ---")

    # --- Test Gamma (RECHAZO) ---
    k_g, th_g = 0.7, 2.5 # k < 1
    testear_distribucion("Gamma (k<1)", generar_gamma_rechazo, (k_g, th_g),
        lambda k,t:k*t, lambda k,t:k*(t**2),
        lambda x,k,t: gamma.pdf(x, a=k, scale=t),
        N_muestras=N_GLOBAL, es_discreta=False, usa_rechazo=True,
        rango_grafica_teorica=(0, gamma.ppf(0.999, a=k_g, scale=th_g) if k_g > 0 else 10))


    # --- Test Pascal (RECHAZO) ---
    r_pasc, p_pasc = 5, 0.4
    testear_distribucion("Pascal", generar_pascal_rechazo, (r_pasc, p_pasc),
        lambda r,p: r*(1-p)/p, lambda r,p: r*(1-p)/(p**2),
        lambda k,r,p: nbinom.pmf(k, r, p), # Scipy nbinom: k=fracasos, n=exitos(r), p=prob_exito
        N_muestras=N_GLOBAL, es_discreta=True, usa_rechazo=True,
        rango_grafica_teorica=(0, int(nbinom.ppf(0.999, n=r_pasc, p=p_pasc)) + 5))

    # --- Test Binomial (RECHAZO) ---
    n_bin, p_bin = 25, 0.25
    testear_distribucion("Binomial", generar_binomial_rechazo, (n_bin, p_bin),
        lambda n,p: n*p, lambda n,p: n*p*(1-p),
        lambda k,n,p: binom.pmf(k, n, p),
        N_muestras=N_GLOBAL, es_discreta=True, usa_rechazo=True,
        rango_grafica_teorica=(0, n_bin))

    # --- Test Hipergeométrica (RECHAZO) ---
    N_h, K_h, n_h = 60, 15, 20
    testear_distribucion("Hipergeométrica", generar_hipergeometrica_rechazo, (N_h, K_h, n_h),
        lambda M,K,n_s: n_s*(K/M) if M>0 else 0,
        lambda M,K,n_s: n_s*(K/M)*(1-K/M)*((M-n_s)/(M-1)) if M>1 else 0,
        lambda k,M,n_K,N_n_s: hypergeom.pmf(k, M, n_K, N_n_s), # Scipy: M=N_total, n=K_exitos_pob, N=n_muestra
        N_muestras=N_GLOBAL, es_discreta=True, usa_rechazo=True,
        rango_grafica_teorica=(max(0, n_h-(N_h-K_h)), min(n_h, K_h)))

    # --- Test Poisson (RECHAZO) ---
    lam_pois = 8.0
    testear_distribucion("Poisson", generar_poisson_rechazo, (lam_pois,),
        lambda l:l, lambda l:l,
        lambda k,l: poisson.pmf(k, l),
        N_muestras=N_GLOBAL, es_discreta=True, usa_rechazo=True,
        rango_grafica_teorica=(0, int(poisson.ppf(0.9999, lam_pois)) + 5))

    # --- Test Empírica Discreta (RECHAZO) ---
    val_emp, prob_emp = [1,2,3,4,5,6], [0.1,0.1,0.3,0.2,0.15,0.15]
    def pmf_emp(k_val, v_list, p_list): # Helper para la teórica de la empírica
        res = []
        for kv in np.atleast_1d(k_val):
            try: idx = v_list.index(kv); res.append(p_list[idx])
            except ValueError: res.append(0)
        return np.array(res)
    testear_distribucion("Empírica Discreta", generar_empirica_discreta_rechazo, (val_emp, prob_emp),
        lambda v,p: np.sum(np.array(v)*np.array(p)),
        lambda v,p: np.sum( ((np.array(v) - np.sum(np.array(v)*np.array(p)))**2) * np.array(p) ),
        lambda k,v,p: pmf_emp(k,v,p),
        N_muestras=N_GLOBAL, es_discreta=True, usa_rechazo=True,
        rango_grafica_teorica=(min(val_emp),max(val_emp)))

    print("\n--- Todos los tests completados. Revisa las gráficas. ---")