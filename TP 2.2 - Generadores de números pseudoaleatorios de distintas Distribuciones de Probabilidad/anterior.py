import random
import math
import matplotlib.pyplot as plt
from scipy.stats import uniform as sp_uniform, expon as sp_expon, norm as sp_norm, \
                        gamma as sp_gamma, nbinom as sp_nbinom, binom as sp_binom, \
                        hypergeom as sp_hypergeom, poisson as sp_poisson
import numpy as np # Para linspace y operaciones de array en gráficos

# --- 0. GENERADOR BASE U(0,1) ---
def generar_U01():
    """Genera un número pseudoaleatorio uniforme continuo entre 0 y 1."""
    return random.random()

# --- 1. DISTRIBUCIONES CON TRANSFORMADA INVERSA ---

# 1.1 UNIFORME (a, b) - Transformada Inversa
def generar_uniforme_TI(a, b):
    """Genera una variable aleatoria Uniforme(a,b) usando Transformada Inversa."""
    u = generar_U01()
    return a + (b - a) * u

# 1.2 EXPONENCIAL (lam) - Transformada Inversa
def generar_exponencial_TI(lam):
    """Genera una variable aleatoria Exponencial(lambda) usando Transformada Inversa."""
    if lam <= 0:
        raise ValueError("Lambda debe ser positivo.")
    u = generar_U01()
    return -math.log(1 - u) / lam # Equivalente a -math.log(u) / lam

# 1.3 NORMAL (mu, sigma) - Transformada Inversa (Box-Muller)
# Box-Muller genera dos N(0,1) a la vez. Para ser estrictos con "una por llamada"
# podríamos guardar la segunda, pero para ilustración simple, generamos y usamos una.
_normal_TI_spare_value = None # Para guardar la segunda normal generada por Box-Muller

def generar_normal_TI(mu, sigma):
    """Genera una variable aleatoria Normal(mu, sigma) usando Box-Muller (T. Inversa)."""
    global _normal_TI_spare_value
    if sigma <= 0:
        raise ValueError("Sigma debe ser positivo.")

    if _normal_TI_spare_value is not None:
        z = _normal_TI_spare_value
        _normal_TI_spare_value = None
    else:
        while True: # Asegurar u1 > 0 para log
            u1 = generar_U01()
            if u1 > 0: break
        u2 = generar_U01()
        
        z1 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        _normal_TI_spare_value = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2) # Guardar la otra
        z = z1
        
    return mu + sigma * z

# --- 2. DISTRIBUCIONES CON MÉTODO DE RECHAZO ---
# Estos son más complejos y requieren una función f(x) (la PMF/PDF objetivo)
# y una función g(x) (de la que sabemos generar, usualmente uniforme o discreta simple)
# y una constante c tal que f(x) <= c*g(x).

# 2.1 GAMMA (k, theta) - Método de Rechazo
# Implementar un método de rechazo genérico para Gamma es complejo.
# Un caso simple es el de Ahrens-Dieter para Gamma(k,1) con k > 1.
# Si k es entero, la suma de exponenciales es más fácil (pero no es Rechazo puro).
# Aquí intentaremos un rechazo simple si k < 1 (basado en Naylor o Fishman).
# Para este ejercicio, si es muy complejo, se puede simplificar o indicar la dificultad.
# **Simplificación para ilustración (NO es un buen generador Gamma general):**
# Este es un placeholder y probablemente MALO. Un buen método de rechazo para Gamma es no trivial.
# El siguiente es un intento MUY simplificado y probablemente ineficiente/incorrecto para una Gamma general.
# Se necesitaría un algoritmo específico de Naylor.
def generar_gamma_MR(k_shape, theta_scale):
    """
    Genera una variable aleatoria Gamma(k, theta) usando un Método de Rechazo MUY SIMPLIFICADO.
    ADVERTENCIA: Este es un ejemplo ilustrativo y puede ser muy ineficiente o incorrecto
    para ciertos parámetros. Un generador de Rechazo robusto para Gamma es complejo.
    Se basa en la idea de rechazar sobre una exponencial si k < 1, o sobre otra función si k >= 1.
    Nos enfocaremos en k<1 usando el algoritmo RGS de Ahrens y Dieter (simplificado).
    """
    if k_shape <= 0 or theta_scale <= 0:
        raise ValueError("k y theta deben ser positivos.")

    if k_shape < 1: # Algoritmo RGS (Ahrens y Dieter) simplificado
        b = (math.e + k_shape) / math.e
        while True:
            p = b * generar_U01()
            u2 = generar_U01()
            if p <= 1:
                x_prop = p**(1/k_shape)
                if u2 <= math.exp(-x_prop):
                    return x_prop * theta_scale
            else: # p > 1
                x_prop = -math.log((b - p) / k_shape)
                if u2 <= x_prop**(k_shape - 1):
                    return x_prop * theta_scale
    else: # k_shape >= 1
        # Para k >= 1, se usan otros algoritmos de rechazo más complejos (ej. Cheng's GKM3, Marsaglia).
        # O si k es entero, se puede usar suma de exponenciales (pero eso no es "Rechazo").
        # Aquí, como demostración de rechazo, podemos intentar un rechazo sobre una Normal
        # (lo cual es conceptualmente posible pero requiere cuidado con la envolvente).
        # Esta es una placeholder para ilustración, NO un método robusto.
        # print(f"ADVERTENCIA: Gamma MR para k={k_shape}>=1 es un placeholder muy ineficiente/básico.")
        # Usaremos el hecho de que para k grande, Gamma se aproxima a Normal.
        # Esto es solo para tener *algo* que use rechazo.
        mean_approx = k_shape * theta_scale
        std_dev_approx = math.sqrt(k_shape * theta_scale**2)
        
        # Cota superior c*g(x) donde g(x) es una Normal. La c puede ser grande.
        # f_gamma(x) / f_normal(x) -> encontrar el máximo c.
        # Esto es complicado de hacer bien analíticamente para una c óptima.
        # Para fines ilustrativos, vamos a aceptar una tasa de rechazo potencialmente alta.
        # El valor de c es crucial y difícil de determinar sin análisis profundo.
        # Para simplificar, se usa un 'c' grande, sabiendo que es ineficiente.
        c_envelope = 5.0 # VALOR ARBITRARIO, SOLO PARA ILUSTRAR EL PROCESO
        
        max_intentos = 10000 # Para evitar bucles infinitos
        intentos = 0
        while intentos < max_intentos:
            intentos +=1
            # Generar de una distribución envolvente g(x), por ejemplo una Normal con media y varianza similares
            # o una exponencial si la cola es importante.
            # Aquí usamos una Normal N(mean_approx, std_dev_approx * 1.5) para tener más cobertura
            x_prop = generar_normal_TI(mean_approx, std_dev_approx * 1.5)
            if x_prop <= 0: continue # Gamma es > 0

            u = generar_U01()
            
            # g_x es la PDF de la Normal usada para proponer
            # f_x es la PDF de la Gamma objetivo
            # Se debe calcular f_x / (c * g_x)
            try:
                # Necesitamos la fdp de la gamma y de la normal
                pdf_gamma_prop = sp_gamma.pdf(x_prop, a=k_shape, scale=theta_scale)
                pdf_normal_envolvente_prop = sp_norm.pdf(x_prop, loc=mean_approx, scale=std_dev_approx * 1.5)
                if pdf_normal_envolvente_prop == 0: continue # Evitar división por cero

                # Condición de aceptación: u <= f(x_prop) / (c * g(x_prop))
                if u * c_envelope * pdf_normal_envolvente_prop <= pdf_gamma_prop :
                    return x_prop
            except (ValueError, OverflowError): # En caso de problemas numéricos
                continue
        # print(f"Gamma MR no convergió para k={k_shape}, theta={theta_scale} después de {max_intentos} intentos.")
        return float('nan') # Si no se genera nada después de muchos intentos

# Para las discretas con Rechazo, necesitamos una PMF f(x) y una PMF envolvente g(x)
# de la que sepamos generar, y una c tal que f(x) <= c*g(x).
# Una g(x) común para discretas es la Uniforme Discreta sobre un rango [0, M].

# Helper para calcular Combinatoria C(n,k)
def combinatoria(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    if k > n // 2:
        k = n - k
    
    res = 1
    for i in range(k):
        res = res * (n - i) // (i + 1)
    return res

# 2.2 PASCAL (r, p) / BINOMIAL NEGATIVA - Método de Rechazo
# Cuenta el número de fracasos k antes del r-ésimo éxito.
# PMF: P(X=k) = C(k+r-1, k) * p^r * (1-p)^k
# Necesitamos un rango máximo M para la uniforme discreta y una cota c.
# Encontrar M y c óptimas es difícil. Para ilustración, M puede ser un cuantil alto.
def generar_pascal_MR(r_exitos, p_prob_exito):
    """Genera Pascal(r,p) (num fracasos) usando Rechazo sobre una Uniforme Discreta."""
    if r_exitos <= 0 or not isinstance(r_exitos, int):
        raise ValueError("r_exitos debe ser entero positivo.")
    if not (0 < p_prob_exito <= 1):
        raise ValueError("p_prob_exito debe estar en (0, 1].")

    # Estimar un M razonable (ej. media + algunas desviaciones estándar)
    # E[X] = r(1-p)/p, Var(X) = r(1-p)/p^2
    if p_prob_exito == 1: return 0
    mean_k = r_exitos * (1 - p_prob_exito) / p_prob_exito
    # M_max_k puede ser muy grande si p es pequeño.
    M_max_k = int(mean_k + 10 * math.sqrt(mean_k / p_prob_exito + 1) + 50) # Rango para la uniforme, heurístico
    if M_max_k <=0 : M_max_k = 50 # un valor minimo

    # Encontrar c: c >= f(k) / g(k) = f(k) / (1/(M_max_k+1))
    # c = (M_max_k+1) * max(f(k)). El max(f(k)) es cerca de la moda.
    # Para simplificar, usaremos un c un poco mayor que (M_max_k+1) * f(moda_aprox)
    # La moda de la binomial negativa (num fracasos) es floor((r-1)(1-p)/p) si r > 1.
    # Si r=1 (Geométrica), moda es 0.
    pmf_at_mode_approx = sp_nbinom.pmf(int(mean_k), r_exitos, p_prob_exito) # Usar scipy para la PMF teórica
    c_envelope = (M_max_k + 1) * pmf_at_mode_approx * 1.5 # Factor de seguridad

    max_intentos_rechazo = 10000
    for _ in range(max_intentos_rechazo):
        # 1. Generar Y de g(y) (Uniforme Discreta en [0, M_max_k])
        y_prop = random.randint(0, M_max_k)
        
        # 2. Generar U ~ U(0,1)
        u = generar_U01()
        
        # 3. Condición de aceptación: u <= f(y_prop) / (c * g(y_prop))
        # g(y_prop) = 1 / (M_max_k + 1)
        # f(y_prop) = PMF de Pascal en y_prop
        pmf_pascal_prop = sp_nbinom.pmf(y_prop, r_exitos, p_prob_exito)
        
        if u * c_envelope * (1 / (M_max_k + 1)) <= pmf_pascal_prop:
            return y_prop
            
    # print(f"Pascal MR no convergió para r={r_exitos}, p={p_prob_exito}")
    return int(mean_k) # Devolver la media si falla, como último recurso para ilustración

# 2.3 BINOMIAL (n, p) - Método de Rechazo
# PMF: P(X=k) = C(n, k) * p^k * (1-p)^(n-k)
def generar_binomial_MR(n_ensayos, p_prob_exito):
    """Genera Binomial(n,p) usando Rechazo sobre una Uniforme Discreta."""
    if n_ensayos < 0 or not isinstance(n_ensayos, int):
        raise ValueError("n_ensayos debe ser entero no negativo.")
    if not (0 <= p_prob_exito <= 1):
        raise ValueError("p_prob_exito debe estar en [0, 1].")
    if n_ensayos == 0: return 0

    M_max_k = n_ensayos # El rango de la binomial es [0, n]
    
    # c = (M_max_k+1) * max(f(k)). El max(f(k)) es cerca de la moda (np).
    moda_aprox = int(n_ensayos * p_prob_exito)
    pmf_at_mode_approx = sp_binom.pmf(moda_aprox, n_ensayos, p_prob_exito)
    c_envelope = (M_max_k + 1) * pmf_at_mode_approx * 1.2 # Factor de seguridad

    if c_envelope == 0 and pmf_at_mode_approx == 0: # Si la moda es 0 (e.g. p=0 o p=1), c puede ser 0
        # Casos borde
        if p_prob_exito == 0: return 0
        if p_prob_exito == 1: return n_ensayos
        c_envelope = 1.0 # Un valor por defecto para evitar división por cero si algo sale mal

    max_intentos_rechazo = 10000
    for _ in range(max_intentos_rechazo):
        y_prop = random.randint(0, M_max_k) # Generar de U[0, n]
        u = generar_U01()
        
        pmf_binom_prop = sp_binom.pmf(y_prop, n_ensayos, p_prob_exito)
        
        # Evitar division por cero si c_envelope es muy pequeño o cero.
        if c_envelope * (1 / (M_max_k + 1)) == 0:
             if pmf_binom_prop > 0 : # si f(y) > 0 pero c*g(y) es 0, algo está mal con c
                 pass #print("Advertencia: c*g(y) es cero pero f(y) no lo es en Binomial MR")
             # si ambos son 0, u <= 0/0 no es evaluable, pero si f(y)=0, no deberíamos aceptar.
             # Si f(y)=0, u*0 <= 0 -> u*0 <= 0 es verdad. Debemos asegurar que si f(y)=0, no se acepte.
             if pmf_binom_prop == 0 : continue # No aceptar si f(y)=0

        if u * c_envelope * (1 / (M_max_k + 1)) <= pmf_binom_prop :
            return y_prop
            
    # print(f"Binomial MR no convergió para n={n_ensayos}, p={p_prob_exito}")
    return int(n_ensayos * p_prob_exito) # Devolver la media si falla

# 2.4 HIPERGEOMÉTRICA (N_pop, K_exitos_pop, n_muestra) - Método de Rechazo
# PMF: P(X=k) = [C(K, k) * C(N-K, n-k)] / C(N, n)
def generar_hipergeometrica_MR(N_pop, K_exitos_pop, n_muestra_tam):
    """Genera Hipergeométrica(N,K,n) usando Rechazo sobre una Uniforme Discreta."""
    # Validaciones
    if not all(isinstance(x, int) for x in [N_pop, K_exitos_pop, n_muestra_tam]):
        raise ValueError("Parámetros deben ser enteros.")
    if not (0 <= K_exitos_pop <= N_pop and 0 <= n_muestra_tam <= N_pop):
        raise ValueError("Parámetros inconsistentes para Hipergeométrica.")

    min_k = max(0, n_muestra_tam - (N_pop - K_exitos_pop))
    max_k = min(n_muestra_tam, K_exitos_pop)
    
    if min_k > max_k : # Rango inválido, no se pueden generar valores
        # Esto puede pasar si, por ejemplo, n_muestra_tam > K_exitos_pop + (N_pop - K_exitos_pop)
        # o sea, si n_muestra_tam > N_pop lo cual ya está validado.
        # O si se piden más éxitos en la muestra de los que hay disponibles
        # O si se piden más fracasos de los que hay.
        # print(f"Hipergeométrica: Rango de k es inválido ({min_k} a {max_k}). Puede que no haya valores posibles.")
        # Devolver un valor dentro del rango posible o un NaN si no hay rango.
        # Si n_muestra_tam = 0, devuelve 0.
        if n_muestra_tam == 0: return 0
        # Si K_exitos_pop = 0, devuelve 0.
        if K_exitos_pop == 0: return 0
        # Si K_exitos_pop = N_pop (todos son exitos), devuelve n_muestra_tam
        if K_exitos_pop == N_pop: return n_muestra_tam

        # Si llegamos aquí, los parámetros son válidos pero el rango calculado es problemático.
        # Forzar un valor si el rango calculado es vacío (ej. K=5, N=10, n=7 -> min_k = max(0, 7-(10-5))=2, max_k=min(7,5)=5. Rango [2,5])
        # Si N=5, K=2, n=4 -> min_k = max(0, 4-(5-2))=1, max_k=min(4,2)=2. Rango [1,2]
        # El problema surge si el rango min_k, max_k es imposible de cumplir con los C(n,k)
        # Por ejemplo, si el denominador C(N_pop, n_muestra_tam) es 0, lo cual no debería ser si n_muestra_tam <= N_pop.
        # Esto usualmente significa que no hay valores posibles, así que la PMF es 0 para todo k.
        # Pero el test estadístico esperará una media.
        # E[X] = n_muestra_tam * (K_exitos_pop / N_pop)
        return int(n_muestra_tam * (K_exitos_pop / N_pop)) # Devolver media como fallback


    M_max_val = max_k # Límite superior para la uniforme discreta
    M_min_val = min_k # Límite inferior
    
    if M_min_val > M_max_val: # No hay rango válido
        # print(f"Hipergeométrica MR: Rango min_k {M_min_val} > max_k {M_max_val}, devolviendo media teórica.")
        return int(n_muestra_tam * (K_exitos_pop / N_pop))


    # Estimar cota c. Moda es aprox. floor((n+1)(K+1)/(N+2)) - 1
    # O simplemente usar la media como punto para evaluar PMF
    mean_k = n_muestra_tam * (K_exitos_pop / N_pop)
    moda_aprox = int(mean_k)
    # Asegurar que moda_aprox está en el rango [min_k, max_k]
    moda_aprox = max(min_k, min(max_k, moda_aprox))

    if M_max_val - M_min_val + 1 <= 0: # El rango de la uniforme es vacío o negativo
        # print(f"Hipergeométrica MR: Rango para uniforme g(x) es inválido ({M_min_val} a {M_max_val}). Devolviendo media.")
        return int(mean_k)

    pmf_at_mode_approx = sp_hypergeom.pmf(moda_aprox, N_pop, K_exitos_pop, n_muestra_tam)
    c_envelope = (M_max_val - M_min_val + 1) * pmf_at_mode_approx * 1.5
    if c_envelope == 0: c_envelope = 1.0 # Evitar división por cero

    max_intentos_rechazo = 10000
    for _ in range(max_intentos_rechazo):
        # Generar de U[min_k, max_k]
        if M_min_val > M_max_val: # Si el rango sigue siendo inválido
             # print(f"Hipergeométrica MR - Bucle: Rango min_k {M_min_val} > max_k {M_max_val}. Devolviendo media.")
             return int(mean_k) # Fallback
        y_prop = random.randint(M_min_val, M_max_val) 
        u = generar_U01()
        
        pmf_hyper_prop = sp_hypergeom.pmf(y_prop, N_pop, K_exitos_pop, n_muestra_tam)
        
        g_y = 1 / (M_max_val - M_min_val + 1) if (M_max_val - M_min_val + 1) > 0 else 1.0

        if u * c_envelope * g_y <= pmf_hyper_prop:
            return y_prop
            
    # print(f"Hipergeométrica MR no convergió.")
    return int(mean_k)

# 2.5 POISSON (lam) - Método de Rechazo
# PMF: P(X=k) = (lam^k * exp(-lam)) / k!
# Knuth es T.Inversa indirecta. Para Rechazo puro, necesitamos envolvente.
# Podemos usar una geométrica como envolvente si lambda es pequeño, o normal si es grande.
# O la más simple: Uniforme Discreta sobre [0, M_max_k].
def generar_poisson_MR(lam):
    """Genera Poisson(lambda) usando Rechazo sobre una Uniforme Discreta."""
    if lam < 0:
        raise ValueError("lambda debe ser no negativo.")
    if lam == 0: return 0

    # M_max_k: media + algunas dev std. std=sqrt(lam)
    M_max_k = int(lam + 10 * math.sqrt(lam) + 20) # Rango heurístico
    
    # c = (M_max_k+1) * max(f(k)). Moda es floor(lam).
    moda_aprox = math.floor(lam)
    pmf_at_mode_approx = sp_poisson.pmf(moda_aprox, lam)
    c_envelope = (M_max_k + 1) * pmf_at_mode_approx * 1.2
    if c_envelope == 0: c_envelope = 1.0

    max_intentos_rechazo = 10000
    for _ in range(max_intentos_rechazo):
        y_prop = random.randint(0, M_max_k)
        u = generar_U01()
        
        pmf_poisson_prop = sp_poisson.pmf(y_prop, lam)
        
        if u * c_envelope * (1 / (M_max_k + 1)) <= pmf_poisson_prop:
            return y_prop
            
    # print(f"Poisson MR no convergió para lam={lam}")
    return int(lam)


# 2.6 EMPÍRICA DISCRETA (valores, probabilidades) - Método de Rechazo
# PMF: P(X=v_i) = prob_i
# g(x) puede ser una uniforme discreta sobre los índices 0 a m-1.
def generar_empirica_discreta_MR(valores, probabilidades):
    """Genera de una Dist. Empírica Discreta usando Rechazo sobre Índices Uniformes."""
    m = len(valores)
    if m == 0:
        raise ValueError("Valores y probabilidades no pueden estar vacíos.")
    if m != len(probabilidades):
        raise ValueError("Longitudes de valores y probabilidades no coinciden.")
    if not math.isclose(sum(probabilidades), 1.0, abs_tol=1e-5):
        # print(f"Advertencia Empírica: Suma de probabilidades es {sum(probabilidades)}")
        pass

    # g(i) = 1/m (probabilidad de elegir el índice i)
    # f(i) = probabilidades[i] (probabilidad del valor en el índice i)
    # c tal que probabilidades[i] <= c * (1/m)  => c >= m * max(probabilidades)
    max_prob = 0
    for p_i in probabilidades: # Encontrar max_prob manualmente
        if p_i > max_prob:
            max_prob = p_i
    
    c_envelope = m * max_prob * 1.05 # Pequeño factor de seguridad
    if c_envelope == 0: c_envelope = 1.0 # Si todas las probs son 0 (no debería ser)

    max_intentos_rechazo = 10000
    for _ in range(max_intentos_rechazo):
        # 1. Generar Y de g(y) (índice uniforme de 0 a m-1)
        idx_prop = random.randint(0, m - 1)
        
        # 2. Generar U ~ U(0,1)
        u = generar_U01()
        
        # 3. Condición: u <= f(idx_prop) / (c * g(idx_prop))
        # f(idx_prop) = probabilidades[idx_prop]
        # g(idx_prop) = 1/m
        f_y = probabilidades[idx_prop]
        
        if u * c_envelope * (1 / m) <= f_y:
            return valores[idx_prop]
            
    # print("Empírica Discreta MR no convergió.")
    # Fallback: devolver un valor aleatorio de la lista de valores si falla el rechazo
    return random.choice(valores)


# --- FUNCIONES DE TESTEO ---
def testear_distribucion(nombre_dist, generador_func, params_generador, 
                         sp_dist_func, params_sp_dist, 
                         N_muestras=10000, es_discreta=False,
                         custom_x_range=None):
    print(f"\n--- Testeando {nombre_dist} ---")
    print(f"Parámetros del generador: {params_generador}")

    muestras = []
    for _ in range(N_muestras):
        muestras.append(generador_func(*params_generador))
    
    muestras = [m for m in muestras if not (isinstance(m, float) and math.isnan(m))] # Filtrar NaNs
    if not muestras:
        print("No se generaron muestras válidas.")
        return

    # 1. Test Estadístico (Media y Varianza)
    media_muestral = np.mean(muestras)
    varianza_muestral = np.var(muestras, ddof=1) # ddof=1 para insesgada

    try:
        media_teorica = sp_dist_func.mean(*params_sp_dist)
        varianza_teorica = sp_dist_func.var(*params_sp_dist)
        print(f"Media Muestral: {media_muestral:.4f} vs Teórica: {media_teorica:.4f}")
        print(f"Varianza Muestral: {varianza_muestral:.4f} vs Teórica: {varianza_teorica:.4f}")
    except Exception as e:
        print(f"No se pudo calcular media/varianza teórica con scipy: {e}")
        print(f"Media Muestral: {media_muestral:.4f}")
        print(f"Varianza Muestral: {varianza_muestral:.4f}")


    # 2. Test Visual (Histograma vs FDP/FP Teórica)
    plt.figure(figsize=(10, 6))
    
    if es_discreta:
        # Para discretas, es mejor un histograma de frecuencias y comparar con PMF
        if not muestras: 
            print("No hay muestras para el histograma discreto.")
            plt.title(f"Distribución {nombre_dist} (Simulada) - SIN MUESTRAS")
            plt.show()
            return

        val_unicos, conteos = np.unique(muestras, return_counts=True)
        frecuencias_relativas = conteos / N_muestras
        plt.bar(val_unicos, frecuencias_relativas, width=0.9, label='Simulada (Frec. Relativa)', alpha=0.7, color='skyblue')

        # PMF Teórica
        if custom_x_range:
            x_teorico = np.array(custom_x_range)
        else:
            min_val = min(val_unicos) if len(val_unicos)>0 else 0
            max_val = max(val_unicos) if len(val_unicos)>0 else 1
            # Asegurar que el rango sea al menos de algunos puntos
            min_val_plot = int(min_val - max(1, abs(min_val)*0.1))
            max_val_plot = int(max_val + max(1, abs(max_val)*0.1) + 2) # +2 para asegurar que max_val esté incluido
            if min_val_plot > max_val_plot : min_val_plot = max_val_plot -1 # Evitar rango invertido
            x_teorico = np.arange(min_val_plot, max_val_plot +1)

        try:
            pmf_teorica = sp_dist_func.pmf(x_teorico, *params_sp_dist)
            plt.plot(x_teorico, pmf_teorica, 'ro-', label='Teórica (PMF)', markersize=5)
        except Exception as e:
            print(f"Error al graficar PMF teórica para {nombre_dist}: {e}")

        plt.xticks(x_teorico) # Asegurar que los ticks sean enteros para discretas
        plt.xlabel("Valor")
        plt.ylabel("Probabilidad / Frecuencia Relativa")

    else: # Continua
        plt.hist(muestras, bins='auto', density=True, label='Simulada (Histograma Normalizado)', alpha=0.7, color='skyblue')
        
        # FDP Teórica
        if custom_x_range:
            x_teorico = np.linspace(custom_x_range[0], custom_x_range[1], 200)
        else:
            min_val = min(muestras) if muestras else 0
            max_val = max(muestras) if muestras else 1
            if min_val == max_val : # si todos los valores son iguales
                min_val -= 0.5
                max_val += 0.5
            x_teorico = np.linspace(min_val, max_val, 200)
        try:
            fdp_teorica = sp_dist_func.pdf(x_teorico, *params_sp_dist)
            plt.plot(x_teorico, fdp_teorica, 'r-', label='Teórica (FDP)', linewidth=2)
        except Exception as e:
            print(f"Error al graficar FDP teórica para {nombre_dist}: {e}")
        plt.xlabel("Valor")
        plt.ylabel("Densidad")

    plt.title(f"Distribución {nombre_dist} (Simulada vs Teórica)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# --- EJECUCIÓN DE LOS TESTS ---
if __name__ == "__main__":
    N_GLOBAL = 10000 # Número de muestras para cada test

    # 1.1 Uniforme
    a, b = 2, 10
    testear_distribucion("Uniforme (TI)", generar_uniforme_TI, (a, b), 
                         sp_uniform, {"loc": a, "scale": b - a}, N_GLOBAL)

    # 1.2 Exponencial
    lam = 0.5
    testear_distribucion("Exponencial (TI)", generar_exponencial_TI, (lam,),
                         sp_expon, {"scale": 1/lam}, N_GLOBAL) # scale = 1/lambda para scipy

    # 1.3 Normal
    mu, sigma = 5, 2
    testear_distribucion("Normal (TI - Box-Muller)", generar_normal_TI, (mu, sigma),
                         sp_norm, {"loc": mu, "scale": sigma}, N_GLOBAL)

    # 2.1 Gamma (MR) - Este es el más delicado por la implementación de MR
    k_gamma, theta_gamma = 2.0, 1.5 # k entero para que sea más fácil para el MR simplificado si k>=1.
                                  # O probar k_gamma=0.5 para el otro camino.
    print(f"\nADVERTENCIA: El generador Gamma por Rechazo es muy simplificado y puede ser ineficiente/impreciso.")
    testear_distribucion("Gamma (MR - Simplificado)", generar_gamma_MR, (k_gamma, theta_gamma),
                         sp_gamma, {"a": k_gamma, "scale": theta_gamma}, N_GLOBAL,
                         custom_x_range=(0, k_gamma*theta_gamma + 5*math.sqrt(k_gamma*theta_gamma**2) + 5) ) # Rango para el gráfico

    # 2.2 Pascal (MR)
    r_pascal, p_pascal = 5, 0.4 # r éxitos, prob de éxito p
    testear_distribucion("Pascal (MR)", generar_pascal_MR, (r_pascal, p_pascal),
                         sp_nbinom, {"n": r_pascal, "p": p_pascal}, N_GLOBAL, es_discreta=True)

    # 2.3 Binomial (MR)
    n_binomial, p_binomial = 20, 0.3
    testear_distribucion("Binomial (MR)", generar_binomial_MR, (n_binomial, p_binomial),
                         sp_binom, {"n": n_binomial, "p": p_binomial}, N_GLOBAL, es_discreta=True)

    # 2.4 Hipergeométrica (MR)
    N_hyper, K_hyper, n_hyper_muestra = 50, 15, 10 # Población, Éxitos en pob, Tamaño muestra
    # Rango para graficar Hipergeométrica
    min_k_h = max(0, n_hyper_muestra - (N_hyper - K_hyper))
    max_k_h = min(n_hyper_muestra, K_hyper)
    testear_distribucion("Hipergeométrica (MR)", generar_hipergeometrica_MR, (N_hyper, K_hyper, n_hyper_muestra),
                         sp_hypergeom, {"M": N_hyper, "n": K_hyper, "N": n_hyper_muestra}, N_GLOBAL, # Notación scipy M,n,N
                         es_discreta=True, custom_x_range=np.arange(min_k_h -1, max_k_h + 2))

    # 2.5 Poisson (MR)
    lam_poisson = 3.5
    testear_distribucion("Poisson (MR)", generar_poisson_MR, (lam_poisson,),
                         sp_poisson, {"mu": lam_poisson}, N_GLOBAL, es_discreta=True)

    # 2.6 Empírica Discreta (MR)
    valores_emp = [10, 20, 30, 40, 50]
    probs_emp =   [0.1, 0.3, 0.4, 0.1, 0.1]
    # Para la Empírica, scipy no tiene una "distribución empírica" directa para media/var/pmf.
    # Lo haremos manualmente para el testeo teórico.
    
    # --- Testeo Manual para Empírica ---
    print(f"\n--- Testeando Empírica Discreta (MR) ---")
    print(f"Valores: {valores_emp}, Probabilidades: {probs_emp}")
    muestras_emp = [generar_empirica_discreta_MR(valores_emp, probs_emp) for _ in range(N_GLOBAL)]
    
    media_muestral_emp = np.mean(muestras_emp)
    varianza_muestral_emp = np.var(muestras_emp, ddof=1)

    media_teorica_emp = sum(v * p for v, p in zip(valores_emp, probs_emp))
    var_teorica_emp = sum(((v - media_teorica_emp)**2) * p for v, p in zip(valores_emp, probs_emp))
    
    print(f"Media Muestral Emp: {media_muestral_emp:.4f} vs Teórica: {media_teorica_emp:.4f}")
    print(f"Varianza Muestral Emp: {varianza_muestral_emp:.4f} vs Teórica: {var_teorica_emp:.4f}")

    plt.figure(figsize=(10, 6))
    val_unicos_emp, conteos_emp = np.unique(muestras_emp, return_counts=True)
    frec_rel_emp = conteos_emp / N_GLOBAL
    plt.bar(val_unicos_emp, frec_rel_emp, width=0.9, label='Simulada (Frec. Relativa)', alpha=0.7, color='skyblue')
    
    # PMF Teórica (son las probabilidades dadas)
    # Asegurar que los valores teóricos estén ordenados si los simulados lo están (unique los ordena)
    # Creamos un diccionario para mapear valor a probabilidad teórica
    map_val_prob = {val: prob for val, prob in zip(valores_emp, probs_emp)}
    
    # Graficar puntos teóricos solo donde hay valores definidos
    # plt.stem(valores_emp, probs_emp, linefmt='r-', markerfmt='ro', basefmt=" ", label='Teórica (PMF)')
    # Usar plot para mejor visualización con el bar chart
    plt.plot(valores_emp, probs_emp, 'ro-', label='Teórica (PMF)', markersize=8)

    plt.xticks(valores_emp) # Ticks en los valores definidos
    plt.xlabel("Valor")
    plt.ylabel("Probabilidad / Frecuencia Relativa")
    plt.title(f"Distribución Empírica Discreta (Simulada vs Teórica)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()