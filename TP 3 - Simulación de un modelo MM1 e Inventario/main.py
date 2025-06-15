import simpy
import random
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------
# MODELO 1: SIMULACIÓN DE COLAS M/M/1 (SIN CAMBIOS)
#-------------------------------------------------------------------------------

def cliente(env, nombre, servidor, resultados):
    """Proceso que representa el ciclo de vida de un cliente."""
    llegada = env.now #Marca la entrada del cliente
    
    if len(servidor.queue) >= servidor.capacidad_cola:
        resultados['denegados'] += 1
        return 

    with servidor.request() as req:
        yield req
        
        inicio_servicio = env.now
        tiempo_en_cola = inicio_servicio - llegada
        resultados['tiempos_en_cola'].append(tiempo_en_cola)
        
        tasa_servicio = resultados['parametros']['tasa_servicio']
        tiempo_servicio = random.expovariate(tasa_servicio)
        yield env.timeout(tiempo_servicio)
        
        salida = env.now
        tiempo_en_sistema = salida - llegada
        resultados['tiempos_en_sistema'].append(tiempo_en_sistema)
        resultados['clientes_atendidos'] += 1

def generador_clientes(env, servidor, resultados):
    """Genera clientes a una tasa de llegada exponencial."""
    tasa_llegada = resultados['parametros']['tasa_llegada']
    i = 0
    while True:
        yield env.timeout(random.expovariate(tasa_llegada))
        i += 1
        env.process(cliente(env, f'Cliente {i}', servidor, resultados))

def simular_mm1(params):
    """Ejecuta múltiples corridas de la simulación M/M/1 y devuelve los resultados."""
    print(f"\n--- Iniciando Simulación M/M/1 ---")
    print(f"Parámetros: {params}")
    
    resultados_finales = {
        'promedio_w': [], 'promedio_wq': [], 'utilizacion': [], 'prob_denegacion': []
    }

    for i in range(params['num_corridas']):
        env = simpy.Environment()
        servidor = simpy.Resource(env, capacity=1)
        servidor.capacidad_cola = params['tamano_cola_finita']
        
        resultados_corrida = {
            'tiempos_en_sistema': [], 'tiempos_en_cola': [], 'clientes_atendidos': 0, 
            'denegados': 0, 'parametros': params
        }
        
        env.process(generador_clientes(env, servidor, resultados_corrida))
        env.run(until=params['tiempo_simulacion'])
        
        total_llegadas = resultados_corrida['clientes_atendidos'] + resultados_corrida['denegados']
        if resultados_corrida['clientes_atendidos'] > 0:
            avg_w = np.mean(resultados_corrida['tiempos_en_sistema'])
            avg_wq = np.mean(resultados_corrida['tiempos_en_cola'])
            tiempo_ocupado = sum(resultados_corrida['tiempos_en_sistema']) - sum(resultados_corrida['tiempos_en_cola'])
            utilizacion = tiempo_ocupado / params['tiempo_simulacion']
        else:
            avg_w, avg_wq, utilizacion = 0, 0, 0
        prob_denegacion = resultados_corrida['denegados'] / total_llegadas if total_llegadas > 0 else 0

        resultados_finales['promedio_w'].append(avg_w)
        resultados_finales['promedio_wq'].append(avg_wq)
        resultados_finales['utilizacion'].append(utilizacion)
        resultados_finales['prob_denegacion'].append(prob_denegacion)

    print("\nResultados Promedio (de {params['num_corridas']} corridas):")
    print(f"  - Tiempo promedio en sistema (W): {np.mean(resultados_finales['promedio_w']):.4f}")
    print(f"  - Tiempo promedio en cola (Wq):   {np.mean(resultados_finales['promedio_wq']):.4f}")
    print(f"  - Utilización del servidor (ρ):   {np.mean(resultados_finales['utilizacion']):.4f}")
    if params['tamano_cola_finita'] != float('inf'):
      print(f"  - Prob. Denegación Servicio:    {np.mean(resultados_finales['prob_denegacion']):.4f}")

    plt.figure(figsize=(10, 5))
    plt.hist(resultados_finales['promedio_w'], bins=10, edgecolor='black')
    plt.title(f'Distribución del Tiempo Promedio en Sistema (W)\nλ={params["tasa_llegada"]}, μ={params["tasa_servicio"]}, K={params["tamano_cola_finita"]}')
    plt.xlabel("Tiempo Promedio en Sistema (W) por corrida")
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.show()

#-------------------------------------------------------------------------------
# MODELO 2: SIMULACIÓN DE INVENTARIO (s, S) - CÓDIGO AJUSTADO AL LIBRO
#-------------------------------------------------------------------------------

def generar_tamano_demanda(prob_acumulada):
    """Genera el tamaño de una demanda basado en una distribución de prob. discreta."""
    u = random.random()
    if u < prob_acumulada[0]: return 1
    elif u < prob_acumulada[1]: return 2
    elif u < prob_acumulada[2]: return 3
    else: return 4

def proceso_llegada_orden(env, estado, cantidad_pedida, params):
    """Espera el tiempo de entrega aleatorio y luego añade el stock."""
    tiempo_entrega = random.uniform(params['min_tiempo_entrega_meses'], params['max_tiempo_entrega_meses'])
    yield env.timeout(tiempo_entrega)
    estado['nivel'] += cantidad_pedida

def evaluacion_periodica_inventario(env, estado, params):
    """Proceso de REVISIÓN PERIÓDICA. Se ejecuta a intervalos regulares."""
    while True:
        if estado['nivel'] < params['punto_reorden_s']:
            cantidad_a_pedir = params['nivel_maximo_S'] - estado['nivel']
            costo_orden_actual = params['costo_fijo_orden'] + (params['costo_incremental_orden'] * cantidad_a_pedir)
            estado['costos']['orden'] += costo_orden_actual
            env.process(proceso_llegada_orden(env, estado, cantidad_a_pedir, params))
        # El cambio clave: el período de revisión es ahora un parámetro
        yield env.timeout(params['periodo_revision_meses'])

def generador_demanda_con_backlog(env, estado, params):
    """Genera demandas de clientes y permite que el inventario sea negativo (backlog)."""
    while True:
        yield env.timeout(random.expovariate(params['tasa_demanda_mes']))
        cantidad_demandada = generar_tamano_demanda(params['prob_acum_demanda'])
        estado['nivel'] -= cantidad_demandada

def calculador_costos_integrados(env, estado, params):
    """Aproxima la integral de costos de mantenimiento y faltante."""
    h = params['costo_mantenimiento_mes']
    p = params['costo_faltante_mes']
    intervalo_calculo = 1.0 / 30.0  # Aproximación "diaria"

    while True:
        inventario_positivo = max(0, estado['nivel'])
        backlog = max(0, -estado['nivel'])
        # Acumular costos proporcionales al intervalo de tiempo
        estado['costos']['mantenimiento'] += inventario_positivo * h * intervalo_calculo
        estado['costos']['faltante'] += backlog * p * intervalo_calculo
        yield env.timeout(intervalo_calculo)

def simular_inventario_libro(params):
    """Ejecuta una corrida de la simulación de inventario del libro."""
    print(f"\n--- Iniciando Simulación de Inventario (Modelo del Libro) ---")
    print(f"Política (s, S): ({params['punto_reorden_s']}, {params['nivel_maximo_S']}), Período Revisión: {params['periodo_revision_meses']} meses")
    
    # Bucle para N corridas (si se desea)
    # Aquí solo se muestra el resultado de la última corrida para simplicidad
    for i in range(params['num_corridas']):
        env = simpy.Environment()
        
        estado = {
            'nivel': params['inventario_inicial'],
            'costos': {'orden': 0, 'mantenimiento': 0, 'faltante': 0}
        }
        
        env.process(evaluacion_periodica_inventario(env, estado, params))
        env.process(generador_demanda_con_backlog(env, estado, params))
        env.process(calculador_costos_integrados(env, estado, params))
        
        env.run(until=params['tiempo_simulacion_meses'])

    # Calcular costos finales y promedios por mes
    num_meses = params['tiempo_simulacion_meses']
    avg_orden = estado['costos']['orden'] / num_meses
    avg_mantenimiento = estado['costos']['mantenimiento'] / num_meses
    avg_faltante = estado['costos']['faltante'] / num_meses
    avg_total = avg_orden + avg_mantenimiento + avg_faltante
    
    print("\nResultados de Costos Promedio POR MES:")
    print(f"  - Costo de Orden:        ${avg_orden:,.2f}")
    print(f"  - Costo de Mantenimiento: ${avg_mantenimiento:,.2f}")
    print(f"  - Costo de Faltante:      ${avg_faltante:,.2f}")
    print(f"  - Costo Total:           ${avg_total:,.2f}")

    labels = [f"Política ({params['punto_reorden_s']}, {params['nivel_maximo_S']})"]
    costos_plot = {'Orden': [avg_orden], 'Mantenimiento': [avg_mantenimiento], 'Faltante': [avg_faltante]}
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bottom = np.zeros(1)
    for nombre, costo in costos_plot.items():
        p = ax.bar(labels, costo, label=nombre, bottom=bottom)
        bottom += costo
    ax.set_title(f"Composición del Costo Promedio Mensual\n(Política s={params['punto_reorden_s']}, S={params['nivel_maximo_S']})")
    ax.set_ylabel('Costo Promedio ($/mes)')
    ax.legend()
    plt.show()

#-------------------------------------------------------------------------------
# EJECUCIÓN PRINCIPAL Y PARAMETRIZACIÓN
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    # --- PARÁMETROS PARA LA SIMULACIÓN M/M/1 ---
    params_mm1 = {
        'tasa_servicio': 10.0,
        'tasa_llegada': 7.5,
        'tamano_cola_finita': float('inf'),
        'num_corridas': 30,
        'tiempo_simulacion': 1000
    }
    simular_mm1(params_mm1) # Puedes descomentar esto para ejecutarlo
    
    # --- PARÁMETROS PARA LA SIMULACIÓN DE INVENTARIO (Modelo del Libro) ---
    # Unidades de tiempo: MESES
    params_inventario_libro = {
        'punto_reorden_s': 20,
        'nivel_maximo_S': 40,
        
        # Parámetro de revisión flexible
        'periodo_revision_meses': 1.0, # <-- ¡MODIFICA ESTE VALOR PARA EXPERIMENTAR!
        
        'tasa_demanda_mes': 10.0, # Tasa = 1 / media_entre_llegadas = 1 / 0.1
        'prob_acum_demanda': [1/6, 1/6 + 1/3, 1/6 + 1/3 + 1/3, 1.0],

        'min_tiempo_entrega_meses': 0.5,
        'max_tiempo_entrega_meses': 1.0,
        
        'costo_fijo_orden': 32.0,
        'costo_incremental_orden': 3.0,
        'costo_mantenimiento_mes': 1.0,
        'costo_faltante_mes': 5.0,
        
        'inventario_inicial': 60,
        'tiempo_simulacion_meses': 120,
        'num_corridas': 10
    }
    simular_inventario_libro(params_inventario_libro)