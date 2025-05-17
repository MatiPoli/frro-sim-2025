import random
import math
import matplotlib.pyplot as plt
import numpy as np # Para media, varianza y linspace para gráficas teóricas
from scipy.stats import uniform, expon, gamma, norm, nbinom, binom, hypergeom, poisson # Para FDP/FP teóricas en tests

# -----------------------------------------------------------------------------
# GENERADOR BASE U(0,1)
# -----------------------------------------------------------------------------
def generar_U01():
    """Genera un número pseudoaleatorio uniforme continuo entre 0 y 1."""
    return random.random()

# -----------------------------------------------------------------------------
# GENERADORES DE DISTRIBUCIONES
# -----------------------------------------------------------------------------

# 1. UNIFORME (continua)
# Método: Transformada Inversa
def generar_uniforme(a, b):
    """Genera un valor de una distribución Uniforme(a, b) usando T. Inversa."""
    if a >= b:
        raise ValueError("El parámetro 'a' debe ser menor que 'b'.")
    u = generar_U01()
    return a + (b - a) * u

# 2. EXPONENCIAL (continua)
# Método: Transformada Inversa
def generar_exponencial(lam):
    """Genera un valor de una distribución Exponencial(lambda) usando T. Inversa."""
    if lam <= 0:
        raise ValueError("El parámetro 'lam' (lambda) debe ser positivo.")
    u = generar_U01()
    # Usamos -ln(u) ya que si u ~ U(0,1), 1-u también es ~ U(0,1)
    # y evita el caso de log(0) si u llegara a ser exactamente 1 (aunque random.random() es [0,1) )
    if u == 0: # Evitar math.log(0)
        return float('inf') # O manejar de otra forma, teóricamente P(X=inf)=0
    return -math.log(u) / lam

# 3. GAMMA (continua)
# Método: Rechazo (Algoritmo de Ahrens-Dieter para k >= 1, simplificado para ilustración)
# NOTA: Implementar un método de rechazo eficiente y correcto para Gamma es complejo.
#       El siguiente es una simplificación y puede no ser eficiente o universal.
#       Para k entero, sumar k exponenciales (T. Inversa indirecta) sería más simple y válido.
#       Dado que la restricción es T.Inversa o Rechazo, y T.Inversa directa no es trivial,
#       se intenta un rechazo.
#       Si el TP lo permite, para k entero, la suma de exponenciales es preferible.
#       Aquí se opta por describir un rechazo general, aunque su implementación completa
#       y eficiente excede el "liquidarlo en 2hs" si no se tiene ya.
#       Para fines prácticos, si k es entero, SE USA SUMA DE EXPONENCIALES (que es una forma de T. Inversa compuesta).
#       Si k no es entero y se exige rechazo puro, la implementación puede ser más larga.
#       Vamos a usar la suma de exponenciales si k es entero, ya que es un método común y derivado de T. Inversa.
#       Si se pide explícitamente un método de rechazo para Gamma general, se necesitaría uno más sofisticado.

def generar_gamma(k, theta):
    """
    Genera un valor de una distribución Gamma(k, theta).
    Si k es entero, usa la suma de k Exponenciales (T. Inversa compuesta).
    Si k no es entero, este método simple no es aplicable y se requeriría
    un método de rechazo más complejo (ej. Ahrens-Dieter, Best).
    Por simplicidad y la restricción de tiempo, solo se implementa para k entero.
    """
    if k <= 0 or theta <= 0:
        raise ValueError("k y theta deben ser positivos.")

    if isinstance(k, int) and k > 0:
        # Método: Suma de k exponenciales (Transformada Inversa Compuesta)
        # Cada exponencial tiene lambda = 1/theta
        lam_exp = 1.0 / theta
        suma_exponenciales = 0
        for _ in range(k):
            suma_exponenciales += generar_exponencial(lam_exp)
        return suma_exponenciales
    else:
        # Para k no entero, se necesitaría un método de rechazo.
        # Ejemplo conceptual de rechazo (no eficiente, solo para ilustrar el concepto):
        # Esto es solo un ESQUEMA, un buen método de rechazo para Gamma es más complejo.
        # print(f"Advertencia: Generar Gamma para k no entero ({k}) requiere un método de rechazo complejo.")
        # print("         Este es un placeholder y no un generador de rechazo eficiente para Gamma.")
        # # Un ejemplo muy simplificado y probablemente ineficiente de rechazo:
        # # Necesitaríamos una función mayorante y un dominio. Esto es difícil para Gamma general.
        # # Para k < 1, se usa a veces el algoritmo de Ahrens y Dieter (GS).
        # # Para k > 1, se usa a veces el algoritmo de Cheng (GB) o Marsaglia.
        # # DADA LA RESTRICCIÓN DE TIEMPO Y MÉTODOS, ESTA PARTE SE SIMPLIFICA.
        # # Si se permite, la aproximación de Wilson-Hilferty o Teorema Central del Límite para k grande
        # # podrían ser opciones, pero no son T.Inversa ni Rechazo directo.
        # # Por ahora, lanzaremos un error si k no es entero para mantenernos en T.Inversa/Rechazo simple.
        raise NotImplementedError(f"Generador Gamma para k no entero ({k}) con Rechazo simple no implementado. "
                                  f"Solo k entero (suma de exponenciales) está soportado bajo T. Inversa compuesta.")

# 4. NORMAL (continua)
# Método: Transformada Inversa (usando Box-Muller, que es una forma de T. Inversa en 2D)
# Box-Muller genera dos N(0,1) a la vez. Para simplificar, devolvemos una.
# Se podría guardar la segunda para la siguiente llamada si se optimiza.
def generar_normal(mu, sigma):
    """Genera un valor de una distribución Normal(mu, sigma) usando Box-Muller (T. Inversa)."""
    if sigma < 0: # sigma == 0 es una constante, no una distribución típica
        raise ValueError("Sigma (desviación estándar) debe ser no negativa.")
    if sigma == 0:
        return mu

    u1 = generar_U01()
    u2 = generar_U01()

    # Asegurarse de que u1 no sea 0 para evitar log(0)
    while u1 == 0:
        u1 = generar_U01()

    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    # z1 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2) # la otra N(0,1)

    return mu + sigma * z0

# 5. PASCAL (discreta) (Binomial Negativa)
# Contando el número de FRACASOS antes del r-ésimo éxito.
# Método: Transformada Inversa (compuesta, sumando 'r' Geométricas)
# Una Geométrica(p) (contando fracasos k antes del 1er éxito) se genera por T. Inversa: k = floor(ln(U) / ln(1-p))
def generar_pascal(r_exitos, p_exito):
    """
    Genera un valor de una distribución Pascal(r, p) (Binomial Negativa)
    contando el número de fracasos antes del r-ésimo éxito.
    Usa T. Inversa compuesta (suma de r Geométricas).
    """
    if not isinstance(r_exitos, int) or r_exitos <= 0:
        raise ValueError("r_exitos debe ser un entero positivo.")
    if not (0 < p_exito <= 1):
        raise ValueError("p_exito debe estar en (0, 1].")

    if p_exito == 1.0: # Éxito seguro en cada intento, 0 fracasos siempre.
        return 0

    num_fracasos_total = 0
    for _ in range(r_exitos):
        u = generar_U01()
        # Geométrica(p) contando fracasos k: k = floor(ln(U) / ln(1-p))
        # Evitar log(0) si u es 0 (aunque U01 es (0,1))
        # Evitar log(1) en el denominador si p_exito es 0 (ya validado)
        if u == 0: # Teóricamente P(U=0)=0. Si ocurre, muchos fracasos.
             # Para efectos prácticos, si U es muy pequeño, ln(U) es muy negativo, k grande.
             # Si se quiere evitar float('inf'), se puede re-generar U o manejar el límite.
             # Por simplicidad, asumimos U > 0.
             num_fracasos_total += float('inf') # Indicativo, o re-generar U
             break # Salir si un componente es infinito
        else:
            num_fracasos_total += math.floor(math.log(u) / math.log(1.0 - p_exito))
    return num_fracasos_total


# 6. BINOMIAL (discreta)
# Método: Transformada Inversa (compuesta, sumando 'n_ensayos' Bernoullis)
# Una Bernoulli(p) es 1 si U < p, 0 sino (esto es T. Inversa para Bernoulli).
def generar_binomial(n_ensayos, p_exito):
    """
    Genera un valor de una distribución Binomial(n, p)
    sumando n variables Bernoulli (T. Inversa compuesta).
    """
    if not isinstance(n_ensayos, int) or n_ensayos < 0:
        raise ValueError("n_ensayos debe ser un entero no negativo.")
    if not (0 <= p_exito <= 1):
        raise ValueError("p_exito debe estar en [0, 1].")

    num_exitos = 0
    for _ in range(n_ensayos):
        if generar_U01() < p_exito: # Esto es generar una Bernoulli(p_exito)
            num_exitos += 1
    return num_exitos

# 7. HIPERGEOMÉTRICA (discreta)
# Método: Simulación directa del muestreo (se puede ver como una forma de T. Inversa secuencial).
# Cada extracción es una Bernoulli cuya probabilidad depende de las extracciones anteriores.
def generar_hipergeometrica(N_pop_total, K_exitos_en_pop, n_muestra_tam):
    """
    Genera un valor de una distribución Hipergeométrica(N, K, n)
    simulando el proceso de muestreo sin reemplazo (T. Inversa secuencial).
    N_pop_total: tamaño total de la población.
    K_exitos_en_pop: número total de ítems "éxito" en la población.
    n_muestra_tam: tamaño de la muestra extraída.
    """
    if not all(isinstance(x, int) for x in [N_pop_total, K_exitos_en_pop, n_muestra_tam]):
        raise ValueError("Todos los parámetros deben ser enteros.")
    if not (0 <= K_exitos_en_pop <= N_pop_total and 0 <= n_muestra_tam <= N_pop_total):
        raise ValueError("Parámetros inconsistentes para Hipergeométrica.")
    if n_muestra_tam == 0:
        return 0

    k_exitos_en_muestra = 0
    N_actual = N_pop_total
    K_actual = K_exitos_en_pop

    for _ in range(n_muestra_tam):
        if N_actual == 0: # No quedan elementos para sacar
            break
        # Probabilidad de sacar un éxito en esta extracción
        prob_exito_actual = K_actual / N_actual if N_actual > 0 else 0

        if generar_U01() < prob_exito_actual: # Bernoulli para la extracción actual
            k_exitos_en_muestra += 1
            K_actual -= 1
        N_actual -= 1

        # Condición de parada si ya no se pueden obtener más éxitos o si todos los restantes deben serlo
        if K_actual < 0: K_actual = 0 # No debería pasar si la lógica es correcta
        if K_actual == 0 and k_exitos_en_muestra < n_muestra_tam: # Ya no hay exitos que sacar
            pass # Continuar sacando fracasos si n_muestra_tam no se ha alcanzado
        if K_actual == N_actual : # Todos los restantes son éxitos
             k_exitos_en_muestra += N_actual # Se suman todos los restantes
             break


    return k_exitos_en_muestra

# 8. POISSON (discreta)
# Método: Transformada Inversa (basado en la relación con tiempos entre llegadas Exponenciales - Algoritmo de Knuth)
# Se generan exponenciales hasta que su suma excede 1 (si lambda es la tasa en un intervalo unitario).
# O, más comúnmente, el algoritmo de Knuth que multiplica U(0,1) hasta que el producto < exp(-lambda).
def generar_poisson(lam):
    """
    Genera un valor de una distribución Poisson(lambda)
    usando el algoritmo de Knuth (basado en T. Inversa para Exponencial).
    """
    if lam < 0:
        raise ValueError("lambda debe ser no negativo.")
    if lam == 0:
        return 0

    L = math.exp(-lam)
    k = 0
    p_acumulada = 1.0
    while True:
        u = generar_U01()
        p_acumulada *= u
        if p_acumulada < L:
            break
        k += 1
    return k

# 9. EMPÍRICA DISCRETA (discreta)
# Método: Transformada Inversa (el método más natural y eficiente para esta distribución).
# Se construye la FDA escalonada y se invierte.
def generar_empirica_discreta(valores, probabilidades):
    """
    Genera un valor de una distribución Empírica Discreta.
    valores: lista o tupla de valores posibles.
    probabilidades: lista o tupla de probabilidades asociadas (deben sumar 1).
    Usa el método de la Transformada Inversa.
    """
    if len(valores) != len(probabilidades):
        raise ValueError("Valores y probabilidades deben tener la misma longitud.")
    if not math.isclose(sum(probabilidades), 1.0, abs_tol=1e-9):
        print(f"Advertencia: La suma de probabilidades es {sum(probabilidades)}, no 1.")
        # Normalizar probabilidades si no suman 1 por errores de flotante, o lanzar error
        # s = sum(probabilidades)
        # probabilidades = [p/s for p in probabilidades]


    prob_acumulada = []
    acum = 0.0
    for p_i in probabilidades:
        acum += p_i
        prob_acumulada.append(acum)

    # Asegurar que el último valor de la acumulada sea 1.0 para cubrir el rango de u
    if prob_acumulada and not math.isclose(prob_acumulada[-1], 1.0, abs_tol=1e-9):
         prob_acumulada[-1] = 1.0

    u = generar_U01()
    for i, c_i in enumerate(prob_acumulada):
        if u <= c_i:
            return valores[i]
    # En caso de que u sea exactamente 1 y haya problemas de precisión con la última prob acumulada
    return valores[-1]


# -----------------------------------------------------------------------------
# FUNCIONES DE TESTEO
# -----------------------------------------------------------------------------
def testear_distribucion(nombre_dist, generador_func, params_dist, teor_media_func, teor_var_func,
                         scipy_dist_func_pdf_pmf, N_muestras=10000, es_discreta=False,
                         rango_grafica_teorica=None):
    """
    Función genérica para testear un generador de distribución.
    - nombre_dist: Nombre de la distribución para los títulos.
    - generador_func: La función que genera valores de la distribución.
    - params_dist: Tupla o lista con los parámetros para el generador_func.
    - teor_media_func: Función lambda que calcula la media teórica dados los params_dist.
    - teor_var_func: Función lambda que calcula la varianza teórica dados los params_dist.
    - scipy_dist_func_pdf_pmf: Función de SciPy (pdf o pmf) para graficar la teórica.
    - N_muestras: Número de muestras a generar.
    - es_discreta: Boolean, True si la distribución es discreta.
    - rango_grafica_teorica: Tupla (min, max) para el eje x de la gráfica teórica.
                              Si es None, se infiere de los datos.
    """
    print(f"\n--- Testeando Distribución: {nombre_dist} con parámetros {params_dist} ---")

    # Generar muestras
    muestras = [generador_func(*params_dist) for _ in range(N_muestras)]

    # 1. Test Estadístico Básico (Comparación de Momentos)
    media_muestral = np.mean(muestras)
    var_muestral = np.var(muestras, ddof=1) # ddof=1 para varianza muestral insesgada

    media_teorica = teor_media_func(*params_dist)
    var_teorica = teor_var_func(*params_dist)

    print(f"  Media Muestral: {media_muestral:.4f} | Media Teórica: {media_teorica:.4f}")
    print(f"  Varianza Muestral: {var_muestral:.4f} | Varianza Teórica: {var_teorica:.4f}")

    # 2. Test Visual (Histograma vs. FDP/FP Teórica)
    plt.figure(figsize=(10, 6))
    if es_discreta:
        # Para discretas, es mejor un histograma de frecuencias relativas con barras centradas en los valores
        val_unicos, conteos = np.unique(muestras, return_counts=True)
        frecuencias_relativas = conteos / N_muestras
        plt.bar(val_unicos, frecuencias_relativas, width=0.8 if len(val_unicos) > 1 else 0.1,
                alpha=0.7, label=f'Muestras Generadas (N={N_muestras})', color='skyblue')

        # PMF teórica
        if rango_grafica_teorica:
            x_teorico = np.arange(rango_grafica_teorica[0], rango_grafica_teorica[1] + 1)
        else:
            min_val = min(val_unicos) if len(val_unicos)>0 else 0
            max_val = max(val_unicos) if len(val_unicos)>0 else 1
            x_teorico = np.arange(min_val, max_val + 1)

        y_teorico_pmf = scipy_dist_func_pdf_pmf(x_teorico, *params_dist)
        plt.stem(x_teorico, y_teorico_pmf, linefmt='r-', markerfmt='ro', basefmt=" ",
                 label='PMF Teórica') # 'use_line_collection=True' para versiones nuevas
        plt.xticks(x_teorico) # Asegurar que los ticks estén en los valores discretos
    else: # Continua
        plt.hist(muestras, bins='auto', density=True, alpha=0.7,
                 label=f'Muestras Generadas (N={N_muestras})', color='skyblue')

        # FDP teórica
        if rango_grafica_teorica:
            x_teorico = np.linspace(rango_grafica_teorica[0], rango_grafica_teorica[1], 200)
        else:
            min_val = min(muestras) if len(muestras)>0 else 0
            max_val = max(muestras) if len(muestras)>0 else 1
            x_teorico = np.linspace(min_val, max_val, 200)

        y_teorico_fdp = scipy_dist_func_pdf_pmf(x_teorico, *params_dist)
        plt.plot(x_teorico, y_teorico_fdp, 'r-', lw=2, label='FDP Teórica')

    plt.title(f'Distribución {nombre_dist}{params_dist}')
    plt.xlabel('Valor')
    plt.ylabel('Densidad / Probabilidad')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# -----------------------------------------------------------------------------
# EJECUCIÓN DE TESTS
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    N_GLOBAL = 10000 # Número de muestras para todos los tests

    # --- Test Uniforme ---
    a_unif, b_unif = 2, 10
    testear_distribucion(
        "Uniforme", generar_uniforme, (a_unif, b_unif),
        lambda a, b: (a + b) / 2,
        lambda a, b: ((b - a)**2) / 12,
        lambda x, a, b: uniform.pdf(x, loc=a, scale=b-a), # scipy.stats.uniform usa loc y scale
        N_muestras=N_GLOBAL, es_discreta=False,
        rango_grafica_teorica=(a_unif - 1, b_unif + 1)
    )

    # --- Test Exponencial ---
    lam_exp = 0.5
    testear_distribucion(
        "Exponencial", generar_exponencial, (lam_exp,),
        lambda l: 1/l,
        lambda l: 1/(l**2),
        lambda x, l: expon.pdf(x, scale=1/l), # scipy.stats.expon usa scale = 1/lambda
        N_muestras=N_GLOBAL, es_discreta=False,
        rango_grafica_teorica=(0, expon.ppf(0.999, scale=1/lam_exp)) # Hasta el percentil 99.9
    )

    # --- Test Gamma (para k entero) ---
    k_gamma, theta_gamma = 3, 2.0 # k_gamma debe ser entero para esta implementación
    # Si k_gamma no es entero, el generador_gamma actual lanzará NotImplementedError
    try:
        testear_distribucion(
            "Gamma (k entero)", generar_gamma, (k_gamma, theta_gamma),
            lambda k, t: k * t,
            lambda k, t: k * (t**2),
            lambda x, k, t: gamma.pdf(x, a=k, scale=t), # scipy.stats.gamma usa a=k (forma) y scale=theta
            N_muestras=N_GLOBAL, es_discreta=False,
            rango_grafica_teorica=(0, gamma.ppf(0.999, a=k_gamma, scale=theta_gamma))
        )
    except NotImplementedError as e:
        print(f"Error en Gamma test: {e}")
    except ValueError as e:
        print(f"Error en Gamma test (ValueError): {e}")


    # --- Test Normal ---
    mu_norm, sigma_norm = 5, 2
    testear_distribucion(
        "Normal", generar_normal, (mu_norm, sigma_norm),
        lambda mu, sigma: mu,
        lambda mu, sigma: sigma**2,
        lambda x, mu, sigma: norm.pdf(x, loc=mu, scale=sigma),
        N_muestras=N_GLOBAL, es_discreta=False,
        rango_grafica_teorica=(norm.ppf(0.001, mu_norm, sigma_norm), norm.ppf(0.999, mu_norm, sigma_norm))
    )

    # --- Test Pascal (Binomial Negativa) ---
    r_pascal, p_pascal = 4, 0.6 # r_pascal éxitos, p_pascal probabilidad de éxito
    # Nuestra función cuenta FRACASOS. Scipy nbinom cuenta éxitos o fracasos según cómo se definan los parámetros.
    # Para scipy.stats.nbinom(k, r, p): k es el número de fracasos, r es el número de éxitos, p es la probabilidad de éxito.
    testear_distribucion(
        "Pascal (N. Fracasos)", generar_pascal, (r_pascal, p_pascal),
        lambda r, p: r * (1-p) / p,            # Media de fracasos
        lambda r, p: r * (1-p) / (p**2),       # Varianza de fracasos
        lambda k, r, p: nbinom.pmf(k, r, p),   # k: nro fracasos, r: nro exitos, p: prob exito
        N_muestras=N_GLOBAL, es_discreta=True,
        rango_grafica_teorica=(0, int(nbinom.ppf(0.999, r_pascal, p_pascal)) +1 ) # +1 para incluir el último valor
    )

    # --- Test Binomial ---
    n_binom, p_binom = 10, 0.3
    testear_distribucion(
        "Binomial", generar_binomial, (n_binom, p_binom),
        lambda n, p: n * p,
        lambda n, p: n * p * (1-p),
        lambda k, n, p: binom.pmf(k, n, p),
        N_muestras=N_GLOBAL, es_discreta=True,
        rango_grafica_teorica=(0, n_binom)
    )

    # --- Test Hipergeométrica ---
    N_pop_h, K_ex_pop_h, n_muestra_h = 50, 20, 10 # Población, Éxitos en Pob, Tamaño Muestra
    # scipy.stats.hypergeom(k, M, n, N): k es el valor, M es N_pop_total, n es K_exitos_en_pop, N es n_muestra_tam
    testear_distribucion(
        "Hipergeométrica", generar_hipergeometrica, (N_pop_h, K_ex_pop_h, n_muestra_h),
        lambda M, K, n_s: n_s * (K / M),
        lambda M, K, n_s: n_s * (K/M) * (1 - K/M) * ((M - n_s) / (M - 1)) if M > 1 else 0,
        lambda k, M, K_p, n_s: hypergeom.pmf(k, M, K_p, n_s), # k, N_total_pop, K_total_exitos_pop, n_tam_muestra
        N_muestras=N_GLOBAL, es_discreta=True,
        rango_grafica_teorica=(max(0, n_muestra_h - (N_pop_h - K_ex_pop_h)), min(n_muestra_h, K_ex_pop_h))
    )


    # --- Test Poisson ---
    lam_poisson = 3.5
    testear_distribucion(
        "Poisson", generar_poisson, (lam_poisson,),
        lambda l: l,
        lambda l: l,
        lambda k, l: poisson.pmf(k, l),
        N_muestras=N_GLOBAL, es_discreta=True,
        rango_grafica_teorica=(0, int(poisson.ppf(0.999, lam_poisson)) + 1)
    )

    # --- Test Empírica Discreta ---
    valores_emp = [10, 20, 30, 40, 50]
    probs_emp =   [0.1, 0.3, 0.35, 0.15, 0.1] # Deben sumar 1
    
    # Para la PMF teórica de la empírica, necesitamos una función personalizada
    def pmf_empirica(k_val, v_list, p_list):
        res = []
        for kv in k_val:
            try:
                idx = v_list.index(kv)
                res.append(p_list[idx])
            except ValueError:
                res.append(0) # Si el valor k no está en la lista de valores, su probabilidad es 0
        return np.array(res)

    testear_distribucion(
        "Empírica Discreta", generar_empirica_discreta, (valores_emp, probs_emp),
        lambda v, p: np.sum(np.array(v) * np.array(p)),
        lambda v, p: np.sum(((np.array(v) - np.sum(np.array(v) * np.array(p)))**2) * np.array(p)),
        lambda k, v, p: pmf_empirica(k, v, p),
        N_muestras=N_GLOBAL, es_discreta=True,
        rango_grafica_teorica=(min(valores_emp), max(valores_emp))
    )

    print("\n--- Todos los tests completados. Revisa las gráficas. ---")