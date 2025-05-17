import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats # Para FDP/FP teóricas y algunas medias/varianzas
import os

if not os.path.exists("graficas"):
    os.makedirs("graficas")

# --- 0. GENERADOR BASE U(0,1) ---
def generar_U01():
    """Genera un número pseudoaleatorio uniforme continuo entre 0 y 1."""
    return random.random()

# --- 1. DISTRIBUCIÓN UNIFORME (continua) ---
def generar_uniforme(a, b):
    """Genera un valor de una distribución Uniforme(a, b)."""
    if a >= b:
        raise ValueError("El parámetro 'a' debe ser menor que 'b'.")
    u = generar_U01()
    return a + (b - a) * u

def teorica_uniforme(params):
    a, b = params
    media = (a + b) / 2
    varianza = ((b - a)**2) / 12
    fdp = lambda x_vals: stats.uniform.pdf(x_vals, loc=a, scale=b-a)
    return media, varianza, fdp, None # No FP para continuas

# --- 2. DISTRIBUCIÓN EXPONENCIAL (continua) ---
def generar_exponencial(lam):
    """Genera un valor de una distribución Exponencial(lambda)."""
    if lam <= 0:
        raise ValueError("Lambda (lam) debe ser positivo.")
    u = generar_U01()
    # Se usa -ln(u) en lugar de -ln(1-u) ya que si u ~ U(0,1), entonces 1-u ~ U(0,1)
    return -math.log(u) / lam

def teorica_exponencial(params):
    lam = params[0]
    media = 1 / lam
    varianza = 1 / (lam**2)
    fdp = lambda x_vals: stats.expon.pdf(x_vals, scale=1/lam)
    return media, varianza, fdp, None

# --- 3. DISTRIBUCIÓN GAMMA (continua) ---
# Implementación para k (alpha) entero, sumando k exponenciales
def generar_gamma_k_entero(k_int, theta):
    """
    Genera un valor de una distribución Gamma(k, theta) donde k es un entero positivo.
    k (alpha): parámetro de forma.
    theta (beta): parámetro de escala.
    """
    if not isinstance(k_int, int) or k_int <= 0:
        raise ValueError("k_int debe ser un entero positivo para este método.")
    if theta <= 0:
        raise ValueError("Theta debe ser positivo.")
    
    lam_exp = 1.0 / theta # Tasa para las exponenciales subyacentes
    suma_exponenciales = 0
    for _ in range(k_int):
        suma_exponenciales += generar_exponencial(lam_exp)
    return suma_exponenciales

def teorica_gamma(params):
    k, theta = params
    media = k * theta
    varianza = k * (theta**2)
    # Usamos scipy.stats.gamma donde 'a' es k (forma) y 'scale' es theta (escala)
    fdp = lambda x_vals: stats.gamma.pdf(x_vals, a=k, scale=theta)
    return media, varianza, fdp, None

# --- 4. DISTRIBUCIÓN NORMAL (continua) ---
# Usando el método de Box-Muller (genera una muestra por llamada)
# Se puede optimizar para generar dos a la vez si se guardara el segundo valor.
_z1_guardado = None # Para guardar el segundo valor de Box-Muller

def generar_normal_box_muller(mu, sigma):
    """Genera un valor de una distribución Normal(mu, sigma) usando Box-Muller."""
    global _z1_guardado
    if sigma < 0:
        raise ValueError("Sigma debe ser no negativo.")

    if _z1_guardado is not None:
        z0 = _z1_guardado
        _z1_guardado = None
    else:
        u1 = generar_U01()
        u2 = generar_U01()
        # Asegurar que u1 no sea 0 para evitar math.log(0)
        while u1 == 0: u1 = generar_U01()

        R = math.sqrt(-2.0 * math.log(u1))
        theta_angle = 2.0 * math.pi * u2
        z0 = R * math.cos(theta_angle)
        _z1_guardado = R * math.sin(theta_angle) # Guardar para la próxima llamada

    return mu + sigma * z0

def teorica_normal(params):
    mu, sigma = params
    media = mu
    varianza = sigma**2
    fdp = lambda x_vals: stats.norm.pdf(x_vals, loc=mu, scale=sigma)
    return media, varianza, fdp, None

# --- 5. DISTRIBUCIÓN PASCAL (Binomial Negativa) (discreta) ---
# Definición: Número de FRACASOS antes de 'r' éxitos.
def generar_geometrica_fracasos(p_exito):
    """Genera el número de fracasos antes del primer éxito en Bernoulli(p_exito)."""
    if not (0 < p_exito <= 1):
        raise ValueError("p_exito debe estar en (0, 1].")
    if p_exito == 1.0: return 0 # Éxito seguro, 0 fracasos
    
    u = generar_U01()
    # Para evitar math.log(0) si u es muy cercano a 1 y p_exito es muy pequeño,
    # o si u es muy cercano a 0 y p_exito es muy cercano a 1.
    # En la práctica, random.random() no devuelve 0.0 exacto típicamente.
    # Si u es 1.0, log(u) es 0, entonces floor(0 / log(1-p)) = 0, lo cual es correcto.
    return math.floor(math.log(u) / math.log(1.0 - p_exito))

def generar_pascal(r_exitos, p_exito):
    """
    Genera un valor de una distribución Pascal(r, p) (Binomial Negativa).
    Cuenta el número de FRACASOS antes de 'r_exitos'.
    """
    if not isinstance(r_exitos, int) or r_exitos <= 0:
        raise ValueError("r_exitos debe ser un entero positivo.")
    
    num_fracasos_total = 0
    for _ in range(r_exitos):
        num_fracasos_total += generar_geometrica_fracasos(p_exito)
    return num_fracasos_total

def teorica_pascal(params): # Binomial Negativa
    r, p = params # r = número de éxitos, p = prob de éxito
    # scipy.stats.nbinom usa 'n' para r (número de éxitos) y 'p' para p (prob de éxito)
    # E[X] = r * (1-p) / p  (para #fracasos)
    # Var(X) = r * (1-p) / p^2 (para #fracasos)
    media = r * (1 - p) / p
    varianza = r * (1 - p) / (p**2)
    fp = lambda k_vals: stats.nbinom.pmf(k_vals, n=r, p=p)
    return media, varianza, None, fp

# --- 6. DISTRIBUCIÓN BINOMIAL (discreta) ---
def generar_binomial(n_ensayos, p_exito):
    """Genera un valor de una distribución Binomial(n, p)."""
    if not isinstance(n_ensayos, int) or n_ensayos < 0:
        raise ValueError("n_ensayos debe ser un entero no negativo.")
    if not (0 <= p_exito <= 1):
        raise ValueError("p_exito debe estar en [0, 1].")
        
    num_exitos = 0
    for _ in range(n_ensayos):
        if generar_U01() < p_exito:
            num_exitos += 1
    return num_exitos

def teorica_binomial(params):
    n, p = params
    media = n * p
    varianza = n * p * (1 - p)
    fp = lambda k_vals: stats.binom.pmf(k_vals, n=n, p=p)
    return media, varianza, None, fp

# --- 7. DISTRIBUCIÓN HIPERGEOMÉTRICA (discreta) ---
def generar_hipergeometrica(N_pop, K_exitos_pop, n_muestra):
    """
    Genera un valor de una distribución Hipergeométrica.
    N_pop: tamaño total de la población.
    K_exitos_pop: número total de ítems "éxito" en la población.
    n_muestra: tamaño de la muestra extraída sin reemplazo.
    """
    if not all(isinstance(x, int) for x in [N_pop, K_exitos_pop, n_muestra]):
        raise ValueError("Todos los parámetros deben ser enteros.")
    if not (0 <= K_exitos_pop <= N_pop and 0 <= n_muestra <= N_pop):
        raise ValueError("Parámetros inconsistentes para Hipergeométrica.")

    k_exitos_en_muestra = 0
    N_actual = N_pop
    K_actual = K_exitos_pop
    
    for _ in range(n_muestra):
        if N_actual == 0: break # No quedan elementos para sacar
        
        prob_exito_actual = K_actual / N_actual if N_actual > 0 else 0
        if generar_U01() < prob_exito_actual:
            k_exitos_en_muestra += 1
            K_actual -= 1
        N_actual -= 1
        if K_actual < 0: K_actual = 0 
            
    return k_exitos_en_muestra

def teorica_hipergeometrica(params):
    N_pop, K_exitos_pop, n_muestra = params
    # scipy.stats.hypergeom usa M (N_pop), n (K_exitos_pop), N (n_muestra)
    media = stats.hypergeom.mean(M=N_pop, n=K_exitos_pop, N=n_muestra)
    varianza = stats.hypergeom.var(M=N_pop, n=K_exitos_pop, N=n_muestra)
    fp = lambda k_vals: stats.hypergeom.pmf(k_vals, M=N_pop, n=K_exitos_pop, N=n_muestra)
    return media, varianza, None, fp

# --- 8. DISTRIBUCIÓN POISSON (discreta) ---
# Método de Knuth
def generar_poisson(lam):
    """Genera un valor de una distribución Poisson(lambda) usando el método de Knuth."""
    if lam < 0:
        raise ValueError("Lambda (lam) debe ser no negativo.")
    if lam == 0: return 0

    L = math.exp(-lam)
    k = 0
    p_acumulada = 1.0
    while True:
        p_acumulada *= generar_U01()
        if p_acumulada < L:
            break
        k += 1
    return k

def teorica_poisson(params):
    lam = params[0]
    media = lam
    varianza = lam
    fp = lambda k_vals: stats.poisson.pmf(k_vals, mu=lam)
    return media, varianza, None, fp

# --- 9. DISTRIBUCIÓN EMPÍRICA DISCRETA (discreta) ---
def generar_empirica_discreta(valores, probabilidades):
    """
    Genera un valor de una distribución empírica discreta.
    valores: lista de valores posibles.
    probabilidades: lista de probabilidades asociadas a cada valor.
    """
    if len(valores) != len(probabilidades):
        raise ValueError("Valores y probabilidades deben tener la misma longitud.")
    if not math.isclose(sum(probabilidades), 1.0, abs_tol=1e-9):
        # print(f"Advertencia: Suma de probabilidades para empírica es {sum(probabilidades)}")
        # Normalizar probabilidades si no suman exactamente 1
        s = sum(probabilidades)
        if s == 0: raise ValueError("La suma de probabilidades no puede ser cero.")
        probabilidades = [p/s for p in probabilidades]


    prob_acumulada = np.cumsum(probabilidades)
    
    u = generar_U01()
    for i, c_i in enumerate(prob_acumulada):
        if u <= c_i:
            return valores[i]
    # En caso de errores de precisión con u=1.0, devolver el último valor.
    return valores[-1] 

def teorica_empirica_discreta(params):
    valores, probabilidades = params
    media = np.sum(np.array(valores) * np.array(probabilidades))
    varianza = np.sum(((np.array(valores) - media)**2) * np.array(probabilidades))
    
    # Para la FP, necesitamos un mapeo de valor a probabilidad
    prob_map = dict(zip(valores, probabilidades))
    def fp(k_vals):
        return [prob_map.get(k, 0) for k in k_vals]
        
    return media, varianza, None, fp


# --- FUNCIONES DE TESTEO ---
def testear_distribucion(nombre_dist, generador_func, params_dist, teorica_func, n_muestras=10000, es_discreta=False):
    """
    Realiza tests estadísticos básicos y visuales para un generador de distribución.
    """
    print(f"\n--- Testeando Distribución: {nombre_dist} con parámetros {params_dist} ---")
    
    # Generar muestras
    muestras = [generador_func(*params_dist) for _ in range(n_muestras)]
    
    # 1. Test Estadístico (Comparación de Momentos)
    media_muestral = np.mean(muestras)
    varianza_muestral = np.var(muestras, ddof=1) # ddof=1 para varianza muestral insesgada
    
    media_teorica, varianza_teorica, fdp_teorica, fp_teorica = teorica_func(params_dist)
    
    print(f"  Media Muestral:   {media_muestral:.4f} (Teórica: {media_teorica:.4f})")
    print(f"  Varianza Muestral: {varianza_muestral:.4f} (Teórica: {varianza_teorica:.4f})")
    
    # 2. Test Visual (Histograma vs. FDP/FP Teórica)
    plt.figure(figsize=(10, 6))
    
    if es_discreta:
        # Para discretas, es mejor usar conteos y comparar con PMF
        valores_unicos, conteos = np.unique(muestras, return_counts=True)
        frecuencias_relativas = conteos / n_muestras
        
        plt.bar(valores_unicos, frecuencias_relativas, width=0.8 if len(valores_unicos) > 1 else 0.1, label='Frec. Relativa Muestral', alpha=0.7)
        
        # Rango para la FP teórica
        if valores_unicos.size > 0:
            k_teorico_min = int(min(valores_unicos))
            k_teorico_max = int(max(valores_unicos))
            # Asegurar que el rango no sea demasiado grande para la gráfica
            if k_teorico_max - k_teorico_min > 50 and nombre_dist != "Empírica Discreta": # Ajustar este umbral
                 k_teorico_min = int(media_teorica - 3 * math.sqrt(varianza_teorica) if varianza_teorica > 0 else media_teorica - 5)
                 k_teorico_max = int(media_teorica + 3 * math.sqrt(varianza_teorica) if varianza_teorica > 0 else media_teorica + 5)
                 k_teorico_min = max(0, k_teorico_min) # Para distribuciones no negativas


            k_vals_teoricos = np.arange(max(0,k_teorico_min), k_teorico_max + 2) # +2 para incluir el último valor posible
            if fp_teorica:
                 y_fp_teorica = fp_teorica(k_vals_teoricos)
                 plt.plot(k_vals_teoricos, y_fp_teorica, 'ro-', label='FP Teórica', markersize=5)

        plt.xticks(np.unique(np.concatenate((valores_unicos, k_vals_teoricos if 'k_vals_teoricos' in locals() and k_vals_teoricos.size > 0 else np.array([])))).astype(int))

    else: # Continuas
        count, bins, ignored = plt.hist(muestras, bins=50, density=True, alpha=0.7, label='Histograma Muestral')
        
        # Rango para la FDP teórica
        x_min, x_max = plt.xlim()
        x_vals_teoricos = np.linspace(x_min, x_max, 200)
        if fdp_teorica:
            y_fdp_teorica = fdp_teorica(x_vals_teoricos)
            plt.plot(x_vals_teoricos, y_fdp_teorica, 'r-', linewidth=2, label='FDP Teórica')

    plt.title(f"Distribución {nombre_dist} ({params_dist}) - N={n_muestras}")
    plt.xlabel("Valor")
    plt.ylabel("Densidad" if not es_discreta else "Frecuencia Relativa")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"graficas/distribución_{nombre_dist}.png")
    #plt.show()


# --- BLOQUE PRINCIPAL PARA EJECUTAR TESTS ---
if __name__ == "__main__":
    N_MUESTRAS_TEST = 10000 # Puedes ajustar esto

    # 1. Uniforme
    testear_distribucion("Uniforme", generar_uniforme, (0, 10), teorica_uniforme, N_MUESTRAS_TEST)
    
    # 2. Exponencial
    testear_distribucion("Exponencial", generar_exponencial, (0.5,), teorica_exponencial, N_MUESTRAS_TEST) # lam = 0.5

    # 3. Gamma (k entero)
    testear_distribucion("Gamma (k entero)", generar_gamma_k_entero, (3, 2.0), teorica_gamma, N_MUESTRAS_TEST) # k=3, theta=2.0

    # 4. Normal
    testear_distribucion("Normal", generar_normal_box_muller, (5, 2), teorica_normal, N_MUESTRAS_TEST) # mu=5, sigma=2

    # 5. Pascal (Binomial Negativa)
    testear_distribucion("Pascal (Bin. Negativa)", generar_pascal, (5, 0.4), teorica_pascal, N_MUESTRAS_TEST, es_discreta=True) # r=5 éxitos, p=0.4

    # 6. Binomial
    testear_distribucion("Binomial", generar_binomial, (20, 0.25), teorica_binomial, N_MUESTRAS_TEST, es_discreta=True) # n=20, p=0.25

    # 7. Hipergeométrica
    testear_distribucion("Hipergeométrica", generar_hipergeometrica, (100, 30, 10), teorica_hipergeometrica, N_MUESTRAS_TEST, es_discreta=True) # N_pop=100, K_exitos_pop=30, n_muestra=10

    # 8. Poisson
    testear_distribucion("Poisson", generar_poisson, (4.5,), teorica_poisson, N_MUESTRAS_TEST, es_discreta=True) # lam=4.5

    # 9. Empírica Discreta
    valores_empirica = [10, 20, 30, 40, 50]
    probs_empirica = [0.1, 0.2, 0.4, 0.2, 0.1]
    testear_distribucion("Empírica Discreta", generar_empirica_discreta, (valores_empirica, probs_empirica), teorica_empirica_discreta, N_MUESTRAS_TEST, es_discreta=True)

    valores_empirica_2 = [1, 2, 3]
    probs_empirica_2 = [1/3, 1/3, 1/3] # Equiprobable
    testear_distribucion("Empírica Discreta (Equiprobable)", generar_empirica_discreta, (valores_empirica_2, probs_empirica_2), teorica_empirica_discreta, N_MUESTRAS_TEST, es_discreta=True)