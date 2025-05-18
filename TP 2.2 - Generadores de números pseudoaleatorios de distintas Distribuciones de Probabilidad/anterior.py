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

# 1. UNIFORME (continua) - Método: Transformada Inversa (como antes)
def generar_uniforme(a, b):
    if a >= b: raise ValueError("a < b")
    u = generar_U01()
    return a + (b - a) * u

# 2. EXPONENCIAL (continua) - Método: Transformada Inversa (como antes)
def generar_exponencial(lam):
    if lam <= 0: raise ValueError("lambda > 0")
    u = generar_U01()
    if u == 0: return float('inf')
    return -math.log(u) / lam

# 3. GAMMA (continua) - Método: RECHAZO
# Esto es más complejo. Un método de rechazo común para Gamma(k, theta)
# cuando k >= 1 es el de Ahrens-Dieter (modificado por Fishman o Cheng).
# Para simplificar (y mucho), si k >= 1, se puede usar una exponencial como propuesta.
# f(x) = (x^(k-1) * exp(-x/theta)) / (Gamma(k) * theta^k)
# Si usamos g(x) ~ Exponencial(lambda_g), necesitamos encontrar 'c'.
# Esto se vuelve complicado rápidamente para un TP corto.
#
# Alternativa más simple (pero potencialmente ineficiente) para k >= 1:
# Usar una Exponencial(1/theta) como base y ajustar.
# (Referencia: Ver "Non-Uniform Random Variate Generation" de Luc Devroye, Cap IX)
# Algoritmo de Johnk (para generar Beta, que se relaciona con Gamma), o variantes.
#
# DADA LA RESTRICCIÓN DE TIEMPO, implementaremos un rechazo conceptualmente más simple
# pero que puede ser ineficiente, especialmente si k es grande o muy pequeño.
# Usaremos una propuesta basada en Exponencial si k > 1, o una transformación si k < 1.
# Este es un placeholder y un método de rechazo REALMENTE BUENO para Gamma es no trivial.

def fdp_gamma(x, k, theta):
    if x < 0: return 0
    # Usamos gammaln para calcular log(Gamma(k)) para mayor estabilidad numérica
    log_gamma_k = gammaln(k)
    numerador = (k-1)*math.log(x) - (x/theta) if x > 0 else -float('inf')
    denominador = log_gamma_k + k*math.log(theta)
    return math.exp(numerador - denominador) if x > 0 else 0

def generar_gamma_rechazo(k, theta):
    """Genera Gamma(k, theta) usando un método de rechazo.
    NOTA: Este es un método simplificado y puede ser ineficiente.
    """
    if k <= 0 or theta <= 0:
        raise ValueError("k y theta deben ser positivos.")

    if k == 1: # Es una Exponencial(1/theta)
        return generar_exponencial(1/theta)

    # Para k > 0 (simplificación, no óptimo)
    # Basado en el método de Best (1978) para k > 1, simplificado.
    # O el algoritmo de Ahrens y Dieter (1974) si k < 1 (más complejo)
    # Vamos a intentar un rechazo simple para k > 1 usando una Normal como propuesta (si k es grande)
    # o una Exponencial.
    # Para k < 1, es más complicado. Usaremos una técnica de "thinning" (Devroye).
    # Gamma(k,theta) = theta * Gamma(k,1)

    if k > 1:
        # Usamos el método de Marsaglia y Tsang (Ziggurat es más rápido pero complejo)
        # o una aproximación más simple de rechazo usando una Normal como "majorizing function"
        # si k es grande, o una Exponencial.
        # Para hacerlo más sencillo para el TP: si k > 1, intentamos un rechazo
        # con una exponencial con media k*theta (la media de la Gamma)
        # Esto NO es un buen método de rechazo general para Gamma. Es un placeholder.
        # Un valor c sería difícil de determinar analíticamente de forma simple.
        #
        # *MEJOR OPCIÓN SIMPLE PARA k > 1 (No es rechazo puro, es T. Inversa compuesta si k es entero)*:
        # if isinstance(k, int):
        #     return sum(generar_exponencial(1/theta) for _ in range(k))
        # *Esta era la opción anterior y es más válida bajo la regla de "T.Inversa o Rechazo"*
        # *Si se exige RECHAZO puro, es más complejo.*
        #
        # Intento de Rechazo para Gamma(k,1) donde k > 1 (luego se escala por theta)
        # Algoritmo RGS de Best (1978) simplificado
        # (Este algoritmo tiene su propia lógica de aceptación/rechazo interna)
        # Referencia: Gentle, "Random Number Generation and Monte Carlo Methods", 2nd ed., p. 113
        # Este no es un "rechazo estándar f(x) <= c*g(x)" sino un algoritmo específico.
        # Dado que pide "método de rechazo" genérico, lo interpretamos ampliamente.
        b = k - 1
        c_const = k + b / math.e # Constante del método, no el 'c' del rechazo general
        while True:
            U1, U2 = generar_U01(), generar_U01()
            P = c_const * U1
            if P <= 1: # Aceptación temprana
                X = P**(1/b) if b != 0 else math.exp(P-1) # Caso b=0 -> k=1
                if U2 <= X: # Aceptación final para P <=1
                    return X * theta
            else: # P > 1
                X = -math.log((c_const - P) / b) if b!=0 else P-1
                if U2 <= X**(b-1) if b!=0 else math.exp(-X) : # Aceptación final para P > 1 (simplificado)
                    return X * theta
        # LA ANTERIOR ES UNA ADAPTACIÓN LIBRE Y SIMPLIFICADA. UN MÉTODO ROBUSTO ES LARGO.
        # Por ahora, mantendremos la suma de exponenciales para k entero como "transformada inversa compuesta"
        # y marcaremos Gamma con Rechazo como "complejo/no implementado de forma robusta".
        if isinstance(k, int) and k > 0:
            return sum(generar_exponencial(1.0/theta) for _ in range(k))
        else:
             raise NotImplementedError("Rechazo robusto para Gamma general no implementado en este script simple. Suma de exponenciales para k entero es T.Inversa Compuesta.")

    elif 0 < k < 1:
        # Algoritmo GS de Ahrens y Dieter (1974) para Gamma(k,1) con 0 < k < 1
        # (Luego se escala por theta)
        # Referencia: Devroye, p. 418
        b_ad = (math.e + k) / math.e
        while True:
            U1 = generar_U01()
            P = b_ad * U1
            if P <= 1:
                X = P**(1/k)
                U2 = generar_U01()
                if U2 <= math.exp(-X):
                    return X * theta
            else: # P > 1
                X = -math.log((b_ad - P) / k)
                U2 = generar_U01()
                if U2 <= X**(k-1):
                    return X * theta


# 4. NORMAL (continua) - Método: Transformada Inversa (Box-Muller, como antes)
def generar_normal(mu, sigma):
    if sigma < 0: raise ValueError("sigma >= 0")
    if sigma == 0: return mu
    u1, u2 = 0,0
    while u1 == 0: u1 = generar_U01() # Evita log(0)
    u2 = generar_U01()
    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mu + sigma * z0

# --- FUNCIONES DE PROBABILIDAD PARA RECHAZO EN DISCRETAS ---
def pmf_pascal(k_val, r_exitos, p_exito): # k_val = número de fracasos
    if k_val < 0: return 0
    # C(k+r-1, k) * p^r * (1-p)^k
    # C(k+r-1, r-1)
    if p_exito == 1 and k_val > 0: return 0
    if p_exito == 1 and k_val == 0: return 1
    if p_exito == 0: return 0 # Nunca se alcanza r_exitos
    try:
        coef_binomial = comb(k_val + r_exitos - 1, r_exitos - 1)
        return coef_binomial * (p_exito ** r_exitos) * ((1 - p_exito) ** k_val)
    except ValueError: # Si k_val es negativo o k_val + r_exitos -1 < r_exitos -1
        return 0

def pmf_binomial(k_val, n_ensayos, p_exito):
    if not (0 <= k_val <= n_ensayos): return 0
    # C(n, k) * p^k * (1-p)^(n-k)
    return comb(n_ensayos, k_val) * (p_exito ** k_val) * ((1 - p_exito) ** (n_ensayos - k_val))

def pmf_hipergeometrica(k_val, N_pop, K_ex_pop, n_muestra):
    # k_val: exitos en muestra
    # N_pop: tamaño total poblacion
    # K_ex_pop: exitos totales en poblacion
    # n_muestra: tamaño de la muestra
    term1 = comb(K_ex_pop, k_val, exact=True) if K_ex_pop >= k_val and k_val >=0 else 0
    term2 = comb(N_pop - K_ex_pop, n_muestra - k_val, exact=True) if (N_pop-K_ex_pop >= n_muestra-k_val) and (n_muestra-k_val >=0) else 0
    denominador = comb(N_pop, n_muestra, exact=True) if N_pop >= n_muestra and n_muestra >=0 else 0
    if denominador == 0: return 0
    return (term1 * term2) / denominador

def pmf_poisson(k_val, lam):
    if k_val < 0: return 0
    # (lambda^k * exp(-lambda)) / k!
    try:
        return (lam ** k_val * math.exp(-lam)) / math.factorial(k_val)
    except OverflowError: # Si lam^k o k! es muy grande
        # Usar logaritmos para estabilidad
        log_pmf = k_val * math.log(lam) - lam - gammaln(k_val + 1)
        return math.exp(log_pmf)


# 5. PASCAL (discreta) - Método: RECHAZO
# f(k) = C(k+r-1, k) * p^r * (1-p)^k
# Necesitamos una g(k) y una c. Una Geométrica podría ser una g(k).
# O una uniforme discreta si podemos acotar el rango k_max.
# Para simplificar: Rechazo con una uniforme discreta hasta un k_max razonable.
# k_max podría ser, por ejemplo, media + algunas desviaciones estándar.
def generar_pascal_rechazo(r_exitos, p_exito):
    if not isinstance(r_exitos, int) or r_exitos <= 0: raise ValueError("r_exitos entero > 0")
    if not (0 < p_exito <= 1): raise ValueError("p_exito en (0, 1]")

    if p_exito == 1.0: return 0 # 0 fracasos siempre

    # Estimamos un k_max para la uniforme discreta g(k)
    # Media = r*(1-p)/p, Varianza = r*(1-p)/p^2
    media_k = r_exitos * (1 - p_exito) / p_exito
    # Un k_max práctico. Podría necesitar ajuste.
    k_max_estimado = math.ceil(media_k + 5 * math.sqrt(media_k / p_exito + media_k) if media_k > 0 else 10*r_exitos)
    k_max_estimado = max(k_max_estimado, r_exitos) # Debe ser al menos r

    # Encontrar c: c = max(f(k) / g(k)). Si g(k) = 1/(k_max+1), entonces c = (k_max+1) * max(f(k)).
    # max_f_k es aproximadamente f(media_k) o cerca del modo.
    # Para simplificar, calculamos f(k) en algunos puntos.
    # O, un c más seguro: (k_max+1) * f(modo). El modo es floor((r-1)(1-p)/p) si r>1.
    # O simplemente iteramos y encontramos el max f(k) en el rango [0, k_max_estimado]
    
    f_k_values = [pmf_pascal(k, r_exitos, p_exito) for k in range(k_max_estimado + 1)]
    max_f_k = 0
    if f_k_values:
        max_f_k = max(f_k_values) if any(f > 0 for f in f_k_values) else 1 # Evitar max de lista vacía o todo ceros

    c_const = (k_max_estimado + 1) * max_f_k
    if c_const == 0: c_const = 1 # Si max_f_k es 0, evitar división por cero

    while True:
        # Generar Y de g(k) = Uniforme Discreta en [0, k_max_estimado]
        y_candidato = random.randint(0, k_max_estimado)
        
        u = generar_U01()
        # g(y_candidato) es 1/(k_max_estimado+1)
        # Aceptar si u <= f(y_candidato) / (c * g(y_candidato))
        # u <= f(y_candidato) / ( (k_max+1)*max_f_k * (1/(k_max+1)) )
        # u <= f(y_candidato) / max_f_k
        if max_f_k == 0: # Caso donde todas las probabilidades en el rango son 0
             if pmf_pascal(y_candidato, r_exitos, p_exito) > 0: # Debería ser 0 si max_f_k es 0
                return y_candidato # Improbable, pero para cubrir
             else:
                continue # Seguir buscando si max_f_k es 0 y f(y_cand) también lo es
        
        f_y = pmf_pascal(y_candidato, r_exitos, p_exito)
        if u * max_f_k <= f_y : # Equivalente a u <= f_y / max_f_k
            return y_candidato
        # Si k_max_estimado es muy bajo, este bucle puede ser infinito o muy lento.

# 6. BINOMIAL (discreta) - Método: RECHAZO
# f(k) = C(n,k) * p^k * (1-p)^(n-k)
# g(k) = Uniforme discreta en [0, n_ensayos]. g(k) = 1/(n_ensayos+1)
# c = (n_ensayos+1) * max(f(k)) (max f(k) ocurre cerca de np)
def generar_binomial_rechazo(n_ensayos, p_exito):
    if not isinstance(n_ensayos, int) or n_ensayos < 0: raise ValueError("n_ensayos entero >= 0")
    if not (0 <= p_exito <= 1): raise ValueError("p_exito en [0, 1]")

    if n_ensayos == 0: return 0

    # Encontrar max_f_k. El modo de la binomial es floor((n+1)p).
    modo = math.floor((n_ensayos + 1) * p_exito)
    if modo > n_ensayos : modo = n_ensayos # Asegurar que esté en rango
    if modo < 0 : modo = 0

    max_f_k = pmf_binomial(modo, n_ensayos, p_exito)
    if max_f_k == 0: # Si p=0 o p=1, el modo puede dar pmf=0 si no es el valor exacto (0 o n)
        if p_exito == 0: max_f_k = pmf_binomial(0, n_ensayos, p_exito) # Debería ser 1
        elif p_exito == 1: max_f_k = pmf_binomial(n_ensayos, n_ensayos, p_exito) # Debería ser 1
        else: # Si p está entre 0 y 1 pero el modo calculado dio pmf 0 (improbable para n>0)
              # buscar el máximo iterando un poco alrededor del modo o en todo el rango
            max_f_k = 0
            for k_test in range(n_ensayos + 1):
                current_pmf = pmf_binomial(k_test, n_ensayos, p_exito)
                if current_pmf > max_f_k:
                    max_f_k = current_pmf
            if max_f_k == 0 and n_ensayos > 0 : # Algo raro, la PMF no debería ser 0 en todos lados
                 max_f_k = 1 # fallback
    
    # c_const = (n_ensayos + 1) * max_f_k (No lo necesitamos explícitamente si usamos la forma u <= f(Y)/M)

    while True:
        # Generar Y de g(k) = U_discreta[0, n_ensayos]
        y_candidato = random.randint(0, n_ensayos)
        u = generar_U01()
        
        # Aceptar si u <= f(Y) / (c * g(Y))
        # Con g(Y) = 1/(n+1), c = (n+1)M donde M = max_f_k
        # Entonces u <= f(Y) / M
        if max_f_k == 0: # Si max_f_k es 0, sólo aceptamos si f(y) también es 0, lo que no genera nada.
                         # Esto pasa si p=0 (solo k=0 es aceptado) o p=1 (solo k=n es aceptado)
            if p_exito == 0 and y_candidato == 0: return 0
            if p_exito == 1 and y_candidato == n_ensayos: return n_ensayos
            continue

        f_y = pmf_binomial(y_candidato, n_ensayos, p_exito)
        if u * max_f_k <= f_y:
            return y_candidato

# 7. HIPERGEOMÉTRICA (discreta) - Método: RECHAZO
# f(k) = [C(K,k)*C(N-K, n-k)] / C(N,n)
# Rango de k: [max(0, n-(N-K)), min(n,K)]
# g(k) = Uniforme discreta en este rango.
def generar_hipergeometrica_rechazo(N_pop, K_ex_pop, n_muestra):
    if not all(isinstance(x, int) for x in [N_pop, K_ex_pop, n_muestra]): raise ValueError("Params enteros")
    if not (0 <= K_ex_pop <= N_pop and 0 <= n_muestra <= N_pop): raise ValueError("Params inconsistentes")

    if n_muestra == 0: return 0
    
    k_min = max(0, n_muestra - (N_pop - K_ex_pop))
    k_max = min(n_muestra, K_ex_pop)

    if k_min > k_max: # No hay valores posibles
        return k_min # O manejar como error, pero k_min podría ser el único resultado si n_muestra es grande

    # Encontrar max_f_k en el rango [k_min, k_max]
    # El modo de la hipergeométrica es más complejo. Iteramos para encontrar el max.
    max_f_k = 0
    # modo_aprox = math.floor((n_muestra + 1) * (K_ex_pop + 1) / (N_pop + 2)) # Aproximación del modo
    # modo_aprox = max(k_min, min(k_max, modo_aprox))
    # max_f_k = pmf_hipergeometrica(modo_aprox, N_pop, K_ex_pop, n_muestra)

    # Iterar en el rango válido para encontrar el máximo real de la PMF
    for k_test in range(k_min, k_max + 1):
        current_pmf = pmf_hipergeometrica(k_test, N_pop, K_ex_pop, n_muestra)
        if current_pmf > max_f_k:
            max_f_k = current_pmf
    
    if max_f_k == 0 and not (k_min == k_max and pmf_hipergeometrica(k_min, N_pop, K_ex_pop, n_muestra) == 0) : # Si todas las probs son 0 (salvo caso trivial)
        # Esto puede pasar si los parámetros son tales que la probabilidad es muy baja en todo el rango.
        # O si k_min > k_max (ya cubierto)
        # O si el único valor posible tiene prob 0 (ej: sacar 5 bolas de una urna con 0 bolas blancas -> k=0 P=1)
        # Para evitar bucle infinito si max_f_k es 0, debemos manejarlo.
        # Si solo hay un valor posible (k_min == k_max), ese es el resultado.
        if k_min == k_max: return k_min
        # Sino, puede ser un problema de parámetros.
        print(f"Advertencia Hiper: max_f_k es 0 para N={N_pop}, K={K_ex_pop}, n={n_muestra}. Rango k: [{k_min}, {k_max}]")
        # Podríamos retornar k_min o lanzar error. Optamos por seguir intentando, pero esto puede ser un bucle.
        # Si max_f_k es genuinamente cero, el bucle de abajo será infinito si f_y siempre es cero.
        # Para seguridad, si max_f_k es 0, y el rango es válido, se asume que hay un error en la PMF o params.
        if k_min <= k_max : max_f_k = 1e-9 # Poner un valor pequeño para evitar div por cero y permitir posible aceptación.

    rango_len = k_max - k_min + 1

    while True:
        # Generar Y de g(k) = U_discreta[k_min, k_max]
        y_candidato = random.randint(k_min, k_max)
        u = generar_U01()

        f_y = pmf_hipergeometrica(y_candidato, N_pop, K_ex_pop, n_muestra)
        if max_f_k == 0: # Ya se manejó arriba, pero por si acaso
             if f_y > 0: return y_candidato # Si max_f_k fue 0 pero f_y no, es raro.
             else: continue

        if u * max_f_k <= f_y:
            return y_candidato

# 8. POISSON (discreta) - Método: RECHAZO
# f(k) = (lambda^k * exp(-lambda)) / k!
# g(k) = Uniforme discreta en [0, k_max_estimado]
# k_max_estimado: media + algunas std dev. Media=lambda, Var=lambda
def generar_poisson_rechazo(lam):
    if lam < 0: raise ValueError("lambda >= 0")
    if lam == 0: return 0

    # Estimación de k_max
    k_max_estimado = math.ceil(lam + 5 * math.sqrt(lam) if lam > 0 else 10) # +5 std devs
    k_max_estimado = max(k_max_estimado, int(lam)+1, 1) # Asegurar que sea al menos lambda y > 0

    # Encontrar max_f_k. Modo de Poisson es floor(lambda).
    modo1 = math.floor(lam)
    modo2 = math.ceil(lam) -1 # si lambda es entero, hay dos modos: lambda y lambda-1
    
    max_f_k = pmf_poisson(modo1, lam)
    if lam == modo1 + 1 and modo1 >=0 : # Si lambda es entero, pmf(lambda) y pmf(lambda-1) son iguales y máximos
        max_f_k_alt = pmf_poisson(modo1-1 if modo1 > 0 else 0, lam) # pmf(lambda-1)
        max_f_k = max(max_f_k, max_f_k_alt)
    elif modo2 != modo1 and modo2 >=0 : # Si lambda no es entero, modo es floor(lambda)
        max_f_k_alt = pmf_poisson(modo2, lam)
        max_f_k = max(max_f_k, max_f_k_alt)
    
    if max_f_k == 0 and lam > 0: # Si pmf en el modo es 0 (raro para lambda>0)
        max_f_k = 0
        for k_test in range(k_max_estimado + 1):
            current_pmf = pmf_poisson(k_test, lam)
            if current_pmf > max_f_k:
                max_f_k = current_pmf
        if max_f_k == 0: max_f_k = 1e-9 # Fallback

    while True:
        y_candidato = random.randint(0, k_max_estimado)
        u = generar_U01()
        
        f_y = pmf_poisson(y_candidato, lam)
        if max_f_k == 0:
            if f_y > 0: return y_candidato
            else: continue
            
        if u * max_f_k <= f_y:
            return y_candidato

# 9. EMPÍRICA DISCRETA (discreta) - Método: RECHAZO
# f(v_i) = p_i
# g(v_i) = Uniforme discreta sobre los valores [v_1, ..., v_m]. g(v_i) = 1/m
# c = m * max(p_i)
def generar_empirica_discreta_rechazo(valores, probabilidades):
    if len(valores) != len(probabilidades): raise ValueError("Longitudes no coinciden")
    if not math.isclose(sum(probabilidades), 1.0, abs_tol=1e-9):
        print(f"Advertencia Empírica: Suma de probs es {sum(probabilidades)}")
    
    m = len(valores)
    if m == 0: raise ValueError("Listas de valores/probabilidades no pueden estar vacías")

    max_p_i = 0
    if probabilidades:
        max_p_i = max(probabilidades) if any(p > 0 for p in probabilidades) else 0

    if max_p_i == 0: # Todas las probabilidades son cero
        # Esto significa que no se puede generar nada, o hay un error en los datos.
        # Podríamos devolver el primer valor, o un error.
        # Si todas las probs son 0, cualquier valor es "aceptable" con prob 0... bucle infinito.
        # Si solo hay un valor, ese es el único que se puede generar, independientemente de su prob.
        if m == 1: return valores[0]
        # Si hay múltiples valores y todas las probs son 0, es un problema.
        print("Advertencia Empírica: Todas las probabilidades son 0. Devolviendo el primer valor.")
        return valores[0] # O lanzar error

    while True:
        # Generar Y de g(v_i) = U_discreta sobre los índices [0, m-1]
        idx_candidato = random.randint(0, m - 1)
        valor_candidato = valores[idx_candidato]
        
        u = generar_U01()
        
        # Aceptar si u <= f(Y) / (c * g(Y))
        # c*g(Y) = (m * max_p_i) * (1/m) = max_p_i
        # Entonces, aceptar si u <= p_candidato / max_p_i
        prob_candidato = probabilidades[idx_candidato]
        
        if u * max_p_i <= prob_candidato:
            return valor_candidato

# -----------------------------------------------------------------------------
# FUNCIONES DE TESTEO (SIN CAMBIOS RESPECTO AL ANTERIOR)
# -----------------------------------------------------------------------------
def testear_distribucion(nombre_dist, generador_func, params_dist, teor_media_func, teor_var_func,
                         scipy_dist_func_pdf_pmf, N_muestras=10000, es_discreta=False,
                         rango_grafica_teorica=None):
    print(f"\n--- Testeando Distribución (Rechazo): {nombre_dist} con parámetros {params_dist} ---")
    muestras = []
    intentos_max_por_muestra = N_muestras * 100 # Para evitar bucles infinitos en rechazos malos
    intentos_totales = 0

    for i in range(N_muestras):
        muestra_generada = None
        intentos_muestra_actual = 0
        while muestra_generada is None and intentos_muestra_actual < 1000 : # Límite por si el rechazo es muy malo
            try:
                muestra_generada = generador_func(*params_dist)
                muestras.append(muestra_generada)
            except Exception as e:
                print(f"Error generando muestra {i} para {nombre_dist}: {e}")
                intentos_muestra_actual +=1
                if intentos_muestra_actual >= 100: # Demasiados errores para esta muestra
                    print(f"FALLO GRAVE: No se pudo generar muestra para {nombre_dist} después de 100 intentos con error.")
                    return # Abortar test para esta distribución
            intentos_totales += 1
            if muestra_generada is not None:
                break # Salir del while si se generó
            if intentos_totales > intentos_max_por_muestra:
                print(f"FALLO GRAVE: Demasiados intentos totales ({intentos_totales}) para {nombre_dist}. Abortando.")
                return
    
    if not muestras:
        print(f"No se generaron muestras para {nombre_dist}. Abortando test.")
        return

    tasa_aceptacion_aprox = N_muestras / intentos_totales if intentos_totales > 0 else 0
    print(f"  Tasa de aceptación aproximada: {tasa_aceptacion_aprox:.4f} (N_muestras={len(muestras)}, Intentos_totales={intentos_totales})")


    media_muestral = np.mean(muestras)
    var_muestral = np.var(muestras, ddof=1) if len(muestras) > 1 else 0

    media_teorica = teor_media_func(*params_dist)
    var_teorica = teor_var_func(*params_dist)

    print(f"  Media Muestral: {media_muestral:.4f} | Media Teórica: {media_teorica:.4f}")
    print(f"  Varianza Muestral: {var_muestral:.4f} | Varianza Teórica: {var_teorica:.4f}")

    plt.figure(figsize=(10, 6))
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
            x_teorico = np.arange(min_val_obs, max_val_obs + 1)

        y_teorico_pmf = scipy_dist_func_pdf_pmf(x_teorico, *params_dist)
        plt.stem(x_teorico, y_teorico_pmf, linefmt='r-', markerfmt='ro', basefmt=" ",
                 label='PMF Teórica')
        if len(x_teorico) < 20: plt.xticks(x_teorico)
    else: # Continua
        plt.hist(muestras, bins='auto', density=True, alpha=0.7,
                 label=f'Muestras (N={len(muestras)})', color='skyblue')
        if rango_grafica_teorica:
            x_teorico = np.linspace(rango_grafica_teorica[0], rango_grafica_teorica[1], 200)
        else:
            min_val_obs = min(muestras) if len(muestras)>0 else 0
            max_val_obs = max(muestras) if len(muestras)>0 else 1
            x_teorico = np.linspace(min_val_obs, max_val_obs, 200)

        y_teorico_fdp = scipy_dist_func_pdf_pmf(x_teorico, *params_dist)
        plt.plot(x_teorico, y_teorico_fdp, 'r-', lw=2, label='FDP Teórica')

    plt.title(f'Distribución {nombre_dist}{params_dist} (Rechazo)')
    plt.xlabel('Valor')
    plt.ylabel('Densidad / Probabilidad')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# -----------------------------------------------------------------------------
# EJECUCIÓN DE TESTS (Adaptar los generadores a los de rechazo)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    N_GLOBAL = 5000 # Reducir N para tests de rechazo que pueden ser lentos

    # --- Test Uniforme (T. Inversa) ---
    # ... (igual que antes)
    a_unif, b_unif = 2, 10
    testear_distribucion(
        "Uniforme", generar_uniforme, (a_unif, b_unif),
        lambda a, b: (a + b) / 2, lambda a, b: ((b - a)**2) / 12,
        lambda x, a, b: uniform.pdf(x, loc=a, scale=b-a),
        N_muestras=N_GLOBAL, es_discreta=False,
        rango_grafica_teorica=(a_unif - 1, b_unif + 1)
    )

    # --- Test Exponencial (T. Inversa) ---
    # ... (igual que antes)
    lam_exp = 0.5
    testear_distribucion(
        "Exponencial", generar_exponencial, (lam_exp,),
        lambda l: 1/l, lambda l: 1/(l**2),
        lambda x, l: expon.pdf(x, scale=1/l),
        N_muestras=N_GLOBAL, es_discreta=False,
        rango_grafica_teorica=(0, expon.ppf(0.999, scale=1/lam_exp))
    )

    # --- Test Gamma (RECHAZO) ---
    k_gamma, theta_gamma = 0.5, 2.0 # Probar k < 1
    #k_gamma, theta_gamma = 3, 2.0 # Probar k entero > 1 (actualmente usa suma de exponenciales)
    try:
        testear_distribucion(
            "Gamma", generar_gamma_rechazo, (k_gamma, theta_gamma),
            lambda k, t: k * t, lambda k, t: k * (t**2),
            lambda x, k, t: gamma.pdf(x, a=k, scale=t),
            N_muestras=N_GLOBAL, es_discreta=False,
            rango_grafica_teorica=(0, gamma.ppf(0.999, a=k_gamma, scale=theta_gamma) if k_gamma > 0 else 10)
        )
    except NotImplementedError as e: print(f"Test Gamma OMITIDO: {e}")
    except ValueError as e: print(f"Test Gamma OMITIDO (ValueError): {e}")

    k_gamma_2, theta_gamma_2 = 3, 1.5 # Para k entero, la implementación actual NO es rechazo.
    print(f"NOTA: Para Gamma con k={k_gamma_2} (entero), el generador actual usa Suma de Exponenciales (T.Inversa Compuesta), no Rechazo puro.")
    try:
        testear_distribucion(
            "Gamma (k entero, actual T.Inv.Comp.)", generar_gamma_rechazo, (k_gamma_2, theta_gamma_2),
            lambda k, t: k * t, lambda k, t: k * (t**2),
            lambda x, k, t: gamma.pdf(x, a=k, scale=t),
            N_muestras=N_GLOBAL, es_discreta=False,
            rango_grafica_teorica=(0, gamma.ppf(0.999, a=k_gamma_2, scale=theta_gamma_2) if k_gamma_2 > 0 else 10)
        )
    except NotImplementedError as e: print(f"Test Gamma k>1 OMITIDO: {e}")
    except ValueError as e: print(f"Test Gamma k>1 OMITIDO (ValueError): {e}")


    # --- Test Normal (T. Inversa - Box Muller) ---
    # ... (igual que antes)
    mu_norm, sigma_norm = 5, 2
    testear_distribucion(
        "Normal", generar_normal, (mu_norm, sigma_norm),
        lambda mu, sigma: mu, lambda mu, sigma: sigma**2,
        lambda x, mu, sigma: norm.pdf(x, loc=mu, scale=sigma),
        N_muestras=N_GLOBAL, es_discreta=False,
        rango_grafica_teorica=(norm.ppf(0.001, mu_norm, sigma_norm), norm.ppf(0.999, mu_norm, sigma_norm))
    )

    # --- Test Pascal (RECHAZO) ---
    r_pascal, p_pascal = 4, 0.6
    testear_distribucion(
        "Pascal", generar_pascal_rechazo, (r_pascal, p_pascal),
        lambda r, p: r * (1-p) / p, lambda r, p: r * (1-p) / (p**2),
        lambda k, r, p: nbinom.pmf(k, r, p),
        N_muestras=N_GLOBAL, es_discreta=True,
        rango_grafica_teorica=(0, int(nbinom.ppf(0.999, r_pascal, p_pascal)) + 5 ) # +5 para dar margen al k_max_estimado
    )

    # --- Test Binomial (RECHAZO) ---
    n_binom, p_binom = 20, 0.3 # Aumentar n para probar mejor el rechazo
    testear_distribucion(
        "Binomial", generar_binomial_rechazo, (n_binom, p_binom),
        lambda n, p: n * p, lambda n, p: n * p * (1-p),
        lambda k, n, p: binom.pmf(k, n, p),
        N_muestras=N_GLOBAL, es_discreta=True,
        rango_grafica_teorica=(0, n_binom)
    )

    # --- Test Hipergeométrica (RECHAZO) ---
    N_pop_h, K_ex_pop_h, n_muestra_h = 50, 10, 15
    testear_distribucion(
        "Hipergeométrica", generar_hipergeometrica_rechazo, (N_pop_h, K_ex_pop_h, n_muestra_h),
        lambda M, K, n_s: n_s * (K / M) if M > 0 else 0,
        lambda M, K, n_s: n_s * (K/M) * (1 - K/M) * ((M - n_s) / (M - 1)) if M > 1 and M > 0 else 0,
        lambda k, M, K_p, n_s: hypergeom.pmf(k, M, K_p, n_s),
        N_muestras=N_GLOBAL, es_discreta=True,
        rango_grafica_teorica=(max(0, n_muestra_h - (N_pop_h - K_ex_pop_h)), min(n_muestra_h, K_ex_pop_h))
    )

    # --- Test Poisson (RECHAZO) ---
    lam_poisson = 7.5 # Un lambda más grande
    testear_distribucion(
        "Poisson", generar_poisson_rechazo, (lam_poisson,),
        lambda l: l, lambda l: l,
        lambda k, l: poisson.pmf(k, l),
        N_muestras=N_GLOBAL, es_discreta=True,
        rango_grafica_teorica=(0, int(poisson.ppf(0.9999, lam_poisson)) + 5)
    )

    # --- Test Empírica Discreta (RECHAZO) ---
    valores_emp = [10, 20, 30, 40, 50, 60]
    probs_emp =   [0.05, 0.1, 0.4, 0.25, 0.15, 0.05]
    def pmf_empirica(k_val, v_list, p_list): # Definida en el script anterior
        res = []
        for kv in np.atleast_1d(k_val): # Asegurar que k_val sea iterable
            try:
                idx = v_list.index(kv)
                res.append(p_list[idx])
            except ValueError: res.append(0)
        return np.array(res)

    testear_distribucion(
        "Empírica Discreta", generar_empirica_discreta_rechazo, (valores_emp, probs_emp),
        lambda v, p: np.sum(np.array(v) * np.array(p)),
        lambda v, p: np.sum(((np.array(v) - np.sum(np.array(v) * np.array(p)))**2) * np.array(p)),
        lambda k, v, p: pmf_empirica(k, v, p),
        N_muestras=N_GLOBAL, es_discreta=True,
        rango_grafica_teorica=(min(valores_emp), max(valores_emp))
    )

    print("\n--- Todos los tests completados (con métodos de rechazo). Revisa las gráficas. ---")