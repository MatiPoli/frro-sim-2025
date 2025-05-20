# plotter.py
import matplotlib.pyplot as plt
import numpy as np
# No se necesita generar_U01 directamente aquí
# Tampoco las funciones de FDP/FP teóricas de SciPy, ya que se pasan como lambda en main.py
import os

# Crear carpeta si no existe

def testear_distribucion(nombre_dist, generador_func, params_dist,
                         scipy_dist_func_pdf_pmf, N_muestras=10000, es_discreta=False,
                         rango_grafica_teorica=None, usa_rechazo=False):
    print(f"\n--- Graficando Distribución: {nombre_dist} con parámetros {params_dist} {'(Rechazo)' if usa_rechazo else '(T.Inversa)'} ---")

    muestras = []
    fallos_generacion_muestra = 0

    for i in range(N_muestras):
        muestra_generada = None
        try:
            muestra_generada = generador_func(*params_dist)
            if muestra_generada is not None:
                muestras.append(muestra_generada)
            else:
                fallos_generacion_muestra += 1
        except NotImplementedError as nie:
            print(f"OMITIDO Gráfico {nombre_dist}: {nie}")
            return
        except Exception as e:
            fallos_generacion_muestra +=1
            if fallos_generacion_muestra > N_muestras / 10 and N_muestras > 100 :
                print(f"Demasiados fallos individuales ({fallos_generacion_muestra}) generando {nombre_dist}. Abortando graficación.")
                return
            continue

    if not muestras:
        print(f"No se generaron muestras válidas para {nombre_dist}. No se puede graficar.")
        return

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
            x_teorico = np.arange(min(min_val_obs,0), max_val_obs + 1)

        if x_teorico.size == 0:
            print(f"  ADVERTENCIA {nombre_dist}: x_teorico está vacío para la PMF teórica. Rango: {rango_grafica_teorica}")
        else:
            try:
                y_teorico_pmf = scipy_dist_func_pdf_pmf(x_teorico, *params_dist)
                plt.stem(x_teorico, y_teorico_pmf, linefmt='r-', markerfmt='ro', basefmt=" ",
                         label='PMF Teórica')

                if len(x_teorico) < 20:
                    plt.xticks(x_teorico)
                elif len(x_teorico) > 0 :
                     tick_step = max(1, len(x_teorico) // 10)
                     plt.xticks(x_teorico[::tick_step].astype(int) if np.issubdtype(x_teorico.dtype, np.integer) else x_teorico[::tick_step])

            except Exception as e_plot_teor:
                print(f"Error graficando PMF teórica para {nombre_dist} {params_dist}: {e_plot_teor}")
                import traceback
                traceback.print_exc()
    else: # Continua
        plt.hist(muestras, bins='auto', density=True, alpha=0.7,
                 label=f'Muestras (N={len(muestras)})', color='skyblue')
        if rango_grafica_teorica:
            x_teorico = np.linspace(rango_grafica_teorica[0], rango_grafica_teorica[1], 200)
        else:
            min_val_obs = min(muestras) if len(muestras)>0 else 0
            max_val_obs = max(muestras) if len(muestras)>0 else 1
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
    os.makedirs("graficos", exist_ok=True)
    nombre_archivo = f"graficos/{nombre_dist}_{'_'.join(map(str, params_dist))}_{'rechazo' if usa_rechazo else 'tinversa'}.png"

    # Reemplazar caracteres no válidos para nombres de archivo en Windows
    caracteres_invalidos = ['<', '>', ':', '"', '/', '\\', '|', '?', '*', '(', ')', ',', ' ']
    for c in caracteres_invalidos:
        nombre_archivo = nombre_archivo.replace(c, '_')
    plt.savefig(nombre_archivo)