import simpy
import random
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------
# MODELO 1: SIMULACIÓN DE COLAS M/M/1 (SIN CAMBIOS)
#-------------------------------------------------------------------------------
# ... (Tu código de M/M/1 va aquí, sin ninguna modificación) ...
def cliente(env, nombre, servidor, resultados):
    llegada = env.now
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
    tasa_llegada = resultados['parametros']['tasa_llegada']
    i = 0
    while True:
        yield env.timeout(random.expovariate(tasa_llegada))
        i += 1
        env.process(cliente(env, f'Cliente {i}', servidor, resultados))

def simular_mm1(params):
    print(f"\n--- Iniciando Simulación M/M/1 ---")
    print(f"Parámetros: {params}")
    resultados_finales = {'promedio_w': [], 'promedio_wq': [], 'utilizacion': [], 'prob_denegacion': []}
    for i in range(params['num_corridas']):
        env = simpy.Environment()
        servidor = simpy.Resource(env, capacity=1)
        servidor.capacidad_cola = params['tamano_cola_finita']
        resultados_corrida = {'tiempos_en_sistema': [], 'tiempos_en_cola': [], 'clientes_atendidos': 0, 'denegados': 0, 'parametros': params}
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
    print(f"\nResultados Promedio (de {params['num_corridas']} corridas):")
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
# MODELO 2: SIMULACIÓN DE INVENTARIO - COSTOS TIEMPO-PONDERADOS (POR ÁREA) - EN MESES
#-------------------------------------------------------------------------------

# Funciones auxiliares (Sin cambios en la lógica)
def generar_tamano_demanda(prob_acumulada):
    u = random.random()
    if u < prob_acumulada[0]: return 1
    elif u < prob_acumulada[1]: return 2
    elif u < prob_acumulada[2]: return 3
    else: return 4

def actualizar_costos_por_area(env, estado, params):
    tiempo_desde_ultimo_evento = env.now - estado['tiempo_ultimo_evento']
    if tiempo_desde_ultimo_evento > 0:
        nivel_anterior = estado['nivel_ultimo_evento']
        if nivel_anterior > 0:
            estado['area_mantenimiento'] += nivel_anterior * tiempo_desde_ultimo_evento
        elif nivel_anterior < 0:
            estado['area_faltante'] += -nivel_anterior * tiempo_desde_ultimo_evento
    estado['tiempo_ultimo_evento'] = env.now
    estado['nivel_ultimo_evento'] = estado['nivel']

# Procesos principales del sistema de inventario (solo se cambian nombres de variables)
def proceso_llegada_orden(env, estado, cantidad_pedida, params):
    # Usa los parámetros en MESES
    tiempo_entrega = random.uniform(params['min_tiempo_entrega_meses'], params['max_tiempo_entrega_meses'])
    yield env.timeout(tiempo_entrega)
    actualizar_costos_por_area(env, estado, params)
    estado['nivel'] += cantidad_pedida

def generador_demanda_con_backlog(env, estado, params):
    # Usa la tasa mensual
    tasa_demanda_mes = 1.0 / params['media_llegada_demanda_meses']
    while True:
        yield env.timeout(random.expovariate(tasa_demanda_mes))
        actualizar_costos_por_area(env, estado, params)
        cantidad_demandada = generar_tamano_demanda(params['prob_acum_demanda'])
        estado['nivel'] -= cantidad_demandada

def evaluacion_periodica_inventario(env, estado, params):
    while True:
        if estado['nivel'] < params['punto_reorden_s']:
            cantidad_a_pedir = params['nivel_maximo_S'] - estado['nivel']
            costo_orden_actual = params['costo_fijo_orden'] + (params['costo_incremental_orden'] * cantidad_a_pedir)
            estado['costo_total_ordenes'] += costo_orden_actual
            env.process(proceso_llegada_orden(env, estado, cantidad_a_pedir, params))
        # Usa el período de revisión en MESES
        yield env.timeout(params['periodo_revision_meses'])

def simular_inventario_por_area(params):
    """Ejecuta una corrida con costos tiempo-ponderados en MESES."""
    # Se cambian los prints para reflejar las unidades correctas
    print(f"\n--- Iniciando Simulación de Inventario (Costos por Área/Integral) ---")
    print(f"Política (s, S): ({params['punto_reorden_s']}, {params['nivel_maximo_S']}), Período Revisión: {params['periodo_revision_meses']} meses")
    
    env = simpy.Environment()
    
    estado = {
        'nivel': params['inventario_inicial'],
        'costo_total_ordenes': 0,
        'tiempo_ultimo_evento': 0,
        'nivel_ultimo_evento': params['inventario_inicial'],
        'area_mantenimiento': 0.0,
        'area_faltante': 0.0,
    }
    
    env.process(evaluacion_periodica_inventario(env, estado, params))
    env.process(generador_demanda_con_backlog(env, estado, params))
    
    # Se usa el tiempo de simulación en MESES
    env.run(until=params['tiempo_simulacion_meses'])

    actualizar_costos_por_area(env, estado, params)

    costo_total_ordenes = estado['costo_total_ordenes']
    
    # -------- CORRECCIÓN EN EL CÁLCULO DE COSTOS --------
    # Ahora 'area_mantenimiento' es (unidades * meses)
    # y 'costo_mantenimiento' es ($ / unidad / mes)
    costo_total_mantenimiento = estado['area_mantenimiento'] * params['costo_mantenimiento_mes']
    costo_total_faltante = estado['area_faltante'] * params['costo_faltante_mes']
    
    costo_total_final = costo_total_ordenes + costo_total_mantenimiento + costo_total_faltante
    
    # El promedio mensual es ahora simplemente el costo total dividido por el número de meses
    num_meses = params['tiempo_simulacion_meses']
    avg_orden_mes = costo_total_ordenes / num_meses
    avg_mantenimiento_mes = costo_total_mantenimiento / num_meses
    avg_faltante_mes = costo_total_faltante / num_meses
    avg_total_mes = costo_total_final / num_meses
    
    print("\nResultados de Costos (Promedio por Mes):")
    print(f"  - Costo de Orden:        ${avg_orden_mes:,.2f}")
    print(f"  - Costo de Mantenimiento: ${avg_mantenimiento_mes:,.2f}")
    print(f"  - Costo de Faltante:      ${avg_faltante_mes:,.2f}")
    print(f"  - Costo Total:           ${avg_total_mes:,.2f}")

    labels = [f"Política ({params['punto_reorden_s']}, {params['nivel_maximo_S']})"]
    costos_plot = {'Orden': [avg_orden_mes], 'Mantenimiento': [avg_mantenimiento_mes], 'Faltante': [avg_faltante_mes]}
    
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
    # --- PARÁMETROS PARA LA SIMULACIÓN DE INVENTARIO (EN MESES, COMO EL LIBRO) ---
    # Se cambian los nombres y valores de los parámetros para usar MESES como unidad de tiempo
    params_inventario = {
        'punto_reorden_s': 20,
        'nivel_maximo_S': 40,
        'periodo_revision_meses': 1.0,
        
        'media_llegada_demanda_meses': 0.1,
        'prob_acum_demanda': [1/6, 1/6 + 1/3, 1/6 + 1/3 + 1/3, 1.0],
        
        'min_tiempo_entrega_meses': 0.5,
        'max_tiempo_entrega_meses': 1.0,
        
        'costo_fijo_orden': 32.0,            # K
        'costo_incremental_orden': 3.0,      # i
        'costo_mantenimiento_mes': 1.0,      # h
        'costo_faltante_mes': 5.0,           # p (pi en el libro)
        
        'inventario_inicial': 60,
        'tiempo_simulacion_meses': 120,
    }
    simular_inventario_por_area(params_inventario)