# main.py
import numpy as np # Para pmf_emp y algunos rangos teóricos
from scipy.stats import uniform, expon, gamma, norm, nbinom, binom, hypergeom, poisson # Para FDP/FP teóricas en tests

from config import N_GLOBAL
from inverse_transform_generators import (
    generar_uniforme, generar_exponencial, generar_normal
)
from rejection_method_generators import (
    generar_gamma_rechazo, generar_pascal_rechazo, generar_binomial_rechazo,
    generar_hipergeometrica_rechazo, generar_poisson_rechazo,
    generar_empirica_discreta_rechazo
)
from plotter import testear_distribucion


if __name__ == "__main__":
    # --- Test Uniforme (T. Inversa) ---
    a_unif, b_unif = 2, 10
    testear_distribucion( "Uniforme", generar_uniforme, (a_unif, b_unif),
        lambda x,a,b: uniform.pdf(x, loc=a, scale=b-a),
        N_muestras=N_GLOBAL, es_discreta=False, usa_rechazo=False,
        rango_grafica_teorica=(a_unif -1, b_unif +1))

    # --- Test Exponencial (T. Inversa) ---
    lam_exp = 0.5
    testear_distribucion("Exponencial", generar_exponencial, (lam_exp,),
        lambda x,l: expon.pdf(x, scale=1/l),
        N_muestras=N_GLOBAL, es_discreta=False, usa_rechazo=False,
        rango_grafica_teorica=(0, expon.ppf(0.999, scale=1/lam_exp)))

    # --- Test Normal (T. Inversa - Box Muller) ---
    mu_norm, sigma_norm = 5, 2
    testear_distribucion("Normal", generar_normal, (mu_norm, sigma_norm),
        lambda x,mu,sig: norm.pdf(x, loc=mu, scale=sig),
        N_muestras=N_GLOBAL, es_discreta=False, usa_rechazo=False,
        rango_grafica_teorica=(norm.ppf(0.001, mu_norm, sigma_norm), norm.ppf(0.999, mu_norm, sigma_norm)))

    print("\n--- Iniciando graficación para distribuciones con MÉTODO DE RECHAZO ---")

    # --- Test Gamma (RECHAZO) ---
    k_g, th_g = 0.7, 2.5 # k < 1
    testear_distribucion("Gamma (k<1)", generar_gamma_rechazo, (k_g, th_g),
        lambda x,k,t: gamma.pdf(x, a=k, scale=t),
        N_muestras=N_GLOBAL, es_discreta=False, usa_rechazo=True,
        rango_grafica_teorica=(0, gamma.ppf(0.999, a=k_g, scale=th_g) if k_g > 0 else 10))
    
    # --- Test Pascal (RECHAZO) ---
    r_pasc, p_pasc = 5, 0.4
    testear_distribucion("Pascal", generar_pascal_rechazo, (r_pasc, p_pasc),
        lambda k,r,p: nbinom.pmf(k, r, p),
        N_muestras=N_GLOBAL, es_discreta=True, usa_rechazo=True,
        rango_grafica_teorica=(0, int(nbinom.ppf(0.999, n=r_pasc, p=p_pasc)) + 5))

    # --- Test Binomial (RECHAZO) ---
    n_bin, p_bin = 25, 0.25
    testear_distribucion("Binomial", generar_binomial_rechazo, (n_bin, p_bin),
        lambda k,n,p: binom.pmf(k, n, p),
        N_muestras=N_GLOBAL, es_discreta=True, usa_rechazo=True,
        rango_grafica_teorica=(0, n_bin))

    # --- Test Hipergeométrica (RECHAZO) ---
    N_h, K_h, n_h = 60, 15, 20
    # La función de SciPy para hipergeométrica usa M (total pop), n (num type I), N (sample size)
    # Nuestro K_ex_pop es n de scipy, N_pop es M de scipy, n_muestra es N de scipy
    testear_distribucion("Hipergeométrica", generar_hipergeometrica_rechazo, (N_h, K_h, n_h),
        lambda k, M, n, N_sample: hypergeom.pmf(k, M, n, N_sample), # M,n,N -> N_pop, K_ex_pop, n_muestra
        N_muestras=N_GLOBAL, es_discreta=True, usa_rechazo=True,
        rango_grafica_teorica=(max(0, n_h-(N_h-K_h)), min(n_h, K_h)))


    # --- Test Poisson (RECHAZO) ---
    lam_pois = 8.0
    testear_distribucion("Poisson", generar_poisson_rechazo, (lam_pois,),
        lambda k,l: poisson.pmf(k, l),
        N_muestras=N_GLOBAL, es_discreta=True, usa_rechazo=True,
        rango_grafica_teorica=(0, int(poisson.ppf(0.9999, lam_pois)) + 5))

    # --- Test Empírica Discreta (RECHAZO) ---
    val_emp, prob_emp = [1,2,3,4,5,6], [0.1,0.1,0.3,0.2,0.15,0.15]
    def pmf_emp(k_val, v_list, p_list): # Helper para la teórica de la empírica
        res = []
        # Asegurarse que k_val sea iterable, incluso si es un solo número para la PMF teórica
        k_val_iter = np.atleast_1d(k_val)
        for kv in k_val_iter:
            try: 
                idx = v_list.index(kv)
                res.append(p_list[idx])
            except ValueError: 
                res.append(0)
        return np.array(res)

    testear_distribucion("Empírica Discreta", generar_empirica_discreta_rechazo, (val_emp, prob_emp),
        lambda k,v,p: pmf_emp(k,v,p),
        N_muestras=N_GLOBAL, es_discreta=True, usa_rechazo=True,
        rango_grafica_teorica=(min(val_emp),max(val_emp)))

    print("\n--- Todas las gráficas completadas. Revisa las ventanas emergentes. ---")