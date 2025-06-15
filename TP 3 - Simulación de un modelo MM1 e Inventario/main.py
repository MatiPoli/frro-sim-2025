import simpy
import random
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------
# MODELO 1: SIMULACIÓN DE COLAS M/M/1
#-------------------------------------------------------------------------------

def cliente(env, nombre, servidor, resultados):
    """Proceso que representa el ciclo de vida de un cliente."""
    llegada = env.now #Marca la entrada del cliente
    
    # Comprobar si hay espacio en la cola (para el caso de cola finita)
    if len(servidor.queue) >= servidor.capacidad_cola:
        # Denegación de servicio
        resultados['denegados'] += 1
        return # El cliente se va

    # El cliente solicita el servidor
    with servidor.request() as req:
        yield req #aca se queda esperando en la "cola" hasta que el servidor le devuelva el flujo.
        
        # El cliente es atendido
        inicio_servicio = env.now
        tiempo_en_cola = inicio_servicio - llegada
        resultados['tiempos_en_cola'].append(tiempo_en_cola)
        
        # Simular tiempo de servicio (Exponencial)
        tasa_servicio = resultados['parametros']['tasa_servicio']
        tiempo_servicio = random.expovariate(tasa_servicio) #tiempo que el cliente ocupa el servidor
        yield env.timeout(tiempo_servicio)
        
        # El cliente se va
        salida = env.now
        tiempo_en_sistema = salida - llegada
        resultados['tiempos_en_sistema'].append(tiempo_en_sistema)
        resultados['clientes_atendidos'] += 1

def generador_clientes(env, servidor, resultados):
    """Genera clientes a una tasa de llegada exponencial."""
    tasa_llegada = resultados['parametros']['tasa_llegada']
    i = 0
    while True:
        # Simular tiempo entre llegadas (Exponencial)
        yield env.timeout(random.expovariate(tasa_llegada)) #Se pausa la ejeción 
        i += 1
        env.process(cliente(env, f'Cliente {i}', servidor, resultados))

def simular_mm1(params):
    """Ejecuta múltiples corridas de la simulación M/M/1 y devuelve los resultados."""
    print(f"\n--- Iniciando Simulación M/M/1 ---")
    print(f"Parámetros: {params}")
    
    # Almacenar resultados de todas las corridas
    resultados_finales = {
        'promedio_w': [],  # Tiempo promedio en sistema por corrida
        'promedio_wq': [], # Tiempo promedio en cola por corrida
        'utilizacion': [], # Utilización por corrida
        'prob_denegacion': [] # Probabilidad de denegación por corrida
    }

    for i in range(params['num_corridas']):
        env = simpy.Environment()
        servidor = simpy.Resource(env, capacity=1)
        # Añadimos un atributo para el tamaño de cola finita
        servidor.capacidad_cola = params['tamano_cola_finita']
        
        resultados_corrida = {
            'tiempos_en_sistema': [],
            'tiempos_en_cola': [],
            'clientes_atendidos': 0,
            'denegados': 0,
            'parametros': params
        }
        
        env.process(generador_clientes(env, servidor, resultados_corrida))
        env.run(until=params['tiempo_simulacion'])
        
        # Calcular métricas para esta corrida
        total_llegadas = resultados_corrida['clientes_atendidos'] + resultados_corrida['denegados']
        if resultados_corrida['clientes_atendidos'] > 0:
            avg_w = np.mean(resultados_corrida['tiempos_en_sistema'])
            avg_wq = np.mean(resultados_corrida['tiempos_en_cola'])
            # La utilización es el tiempo que el servidor estuvo ocupado / tiempo total
            tiempo_ocupado = sum(resultados_corrida['tiempos_en_sistema']) - sum(resultados_corrida['tiempos_en_cola'])
            utilizacion = tiempo_ocupado / params['tiempo_simulacion']
        else:
            avg_w, avg_wq, utilizacion = 0, 0, 0
            
        prob_denegacion = resultados_corrida['denegados'] / total_llegadas if total_llegadas > 0 else 0

        resultados_finales['promedio_w'].append(avg_w)
        resultados_finales['promedio_wq'].append(avg_wq)
        resultados_finales['utilizacion'].append(utilizacion)
        resultados_finales['prob_denegacion'].append(prob_denegacion)

    # Imprimir resultados promedio de todas las corridas
    print("\nResultados Promedio (de 30 corridas):")
    print(f"  - Tiempo promedio en sistema (W): {np.mean(resultados_finales['promedio_w']):.4f}")
    print(f"  - Tiempo promedio en cola (Wq):   {np.mean(resultados_finales['promedio_wq']):.4f}")
    print(f"  - Utilización del servidor (ρ):   {np.mean(resultados_finales['utilizacion']):.4f}")
    if params['tamano_cola_finita'] != float('inf'):
      print(f"  - Prob. Denegación Servicio:    {np.mean(resultados_finales['prob_denegacion']):.4f}")

    # Generar gráfico básico
    plt.figure(figsize=(10, 5))
    plt.hist(resultados_finales['promedio_w'], bins=10, edgecolor='black')
    plt.title(f'Distribución del Tiempo Promedio en Sistema (W)\nλ={params["tasa_llegada"]}, μ={params["tasa_servicio"]}, K={params["tamano_cola_finita"]}')
    plt.xlabel("Tiempo Promedio en Sistema (W) por corrida")
    plt.ylabel("Frecuencia (de 30 corridas)")
    plt.grid(True)
    plt.show()

#-------------------------------------------------------------------------------
# MODELO 2: SIMULACIÓN DE INVENTARIO (s, S)
#-------------------------------------------------------------------------------

def monitor_inventario(env, inventario, costos, params):
    """Revisa el nivel de inventario y hace pedidos si es necesario."""
    while True:
        # Revisión continua
        if inventario.level <= params['punto_reorden_s']:
            cantidad_a_pedir = params['nivel_maximo_S'] - inventario.level
            
            # Registrar costo de la orden
            costos['orden'] += params['costo_orden']
            
            # Esperar el tiempo de entrega
            yield env.timeout(params['tiempo_entrega'])
            
            # Recibir el pedido
            yield inventario.put(cantidad_a_pedir)
        
        yield env.timeout(1) # Revisa el inventario una vez por día (unidad de tiempo)

def generador_demanda(env, inventario, costos, params):
    """Genera demandas de clientes."""
    while True:
        # Tiempo hasta la próxima demanda (basado en una tasa diaria)
        yield env.timeout(random.expovariate(params['demanda_media_diaria']))
        
        cantidad_demandada = 1 # Suponemos que cada cliente pide 1 unidad
        
        if inventario.level >= cantidad_demandada:
            yield inventario.get(cantidad_demandada)
        else:
            # Hay un faltante
            costos['faltante'] += params['costo_faltante']

def calculador_costo_mantenimiento(env, inventario, costos, params):
    """Calcula el costo de mantenimiento diariamente."""
    while True:
        costos['mantenimiento'] += inventario.level * params['costo_mantenimiento_diario']
        yield env.timeout(1) # Calcula el costo cada día

def simular_inventario(params):
    """Ejecuta múltiples corridas de la simulación de inventario."""
    print(f"\n--- Iniciando Simulación de Inventario ---")
    print(f"Parámetros de la política (s, S): {params}")
    
    resultados_finales = {
        'costo_orden': [],
        'costo_mantenimiento': [],
        'costo_faltante': [],
        'costo_total': []
    }

    for i in range(params['num_corridas']):
        env = simpy.Environment()
        inventario = simpy.Container(env, capacity=params['nivel_maximo_S'], init=params['nivel_maximo_S'])
        
        costos_corrida = {'orden': 0, 'mantenimiento': 0, 'faltante': 0}
        
        env.process(monitor_inventario(env, inventario, costos_corrida, params))
        env.process(generador_demanda(env, inventario, costos_corrida, params))
        env.process(calculador_costo_mantenimiento(env, inventario, costos_corrida, params))
        
        env.run(until=params['tiempo_simulacion'])
        
        costo_total_corrida = costos_corrida['orden'] + costos_corrida['mantenimiento'] + costos_corrida['faltante']
        
        resultados_finales['costo_orden'].append(costos_corrida['orden'])
        resultados_finales['costo_mantenimiento'].append(costos_corrida['mantenimiento'])
        resultados_finales['costo_faltante'].append(costos_corrida['faltante'])
        resultados_finales['costo_total'].append(costo_total_corrida)

    # Imprimir resultados promedio
    avg_orden = np.mean(resultados_finales['costo_orden'])
    avg_mantenimiento = np.mean(resultados_finales['costo_mantenimiento'])
    avg_faltante = np.mean(resultados_finales['costo_faltante'])
    avg_total = np.mean(resultados_finales['costo_total'])
    
    print("\nResultados de Costos Promedio (de 30 corridas):")
    print(f"  - Costo de Orden:        ${avg_orden:,.2f}")
    print(f"  - Costo de Mantenimiento: ${avg_mantenimiento:,.2f}")
    print(f"  - Costo de Faltante:      ${avg_faltante:,.2f}")
    print(f"  - Costo Total:           ${avg_total:,.2f}")

    # Generar gráfico básico de barras apiladas
    labels = [f"Política (s={params['punto_reorden_s']}, S={params['nivel_maximo_S']})"]
    costos_plot = {
        'Orden': [avg_orden],
        'Mantenimiento': [avg_mantenimiento],
        'Faltante': [avg_faltante],
    }
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bottom = np.zeros(1)
    for nombre, costo in costos_plot.items():
        p = ax.bar(labels, costo, label=nombre, bottom=bottom)
        bottom += costo
    
    ax.set_title('Composición del Costo Total de Inventario')
    ax.set_ylabel('Costo Promedio ($)')
    ax.legend()
    plt.show()


#-------------------------------------------------------------------------------
# EJECUCIÓN PRINCIPAL Y PARAMETRIZACIÓN
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    # --- PARÁMETROS PARA LA SIMULACIÓN M/M/1 ---
    # Puedes modificar estos valores para cada experimento
    params_mm1 = {
        # Elige una tasa de servicio, ej. 10 clientes por hora
        'tasa_servicio': 15.0,
        
        # Elige una tasa de llegada (λ). Prueba variando esto:
        # 0.75 * tasa_servicio = 7.5 (utilización del 75%)
        # 1.00 * tasa_servicio = 10.0 (utilización del 100%, sistema crítico)
        'tasa_llegada': 7.5, 
        
        # Para cola finita, usa un número (0, 2, 5, 10, 50). 
        # Para cola infinita, usa float('inf')
        'tamano_cola_finita': float('inf'), 
        
        'num_corridas': 30,
        'tiempo_simulacion': 1000 # Horas de simulación
    }
    simular_mm1(params_mm1)
    
    # --- PARÁMETROS PARA LA SIMULACIÓN DE INVENTARIO ---
    # Puedes modificar estos valores para probar diferentes políticas
    params_inventario = {
        'punto_reorden_s': 20, # Cuando el inventario llega a este nivel, se ordena
        'nivel_maximo_S': 100, # Se ordena para rellenar hasta este nivel
        'demanda_media_diaria': 10, # Unidades demandadas por día en promedio
        
        'costo_orden': 50.0, # Costo fijo por cada orden realizada
        'costo_mantenimiento_diario': 0.5, # Costo de almacenar 1 unidad por 1 día
        'costo_faltante': 15.0, # Costo por cada unidad no satisfecha
        
        'tiempo_entrega': 3, # Días que tarda en llegar un pedido
        
        'num_corridas': 30,
        'tiempo_simulacion': 365 # Simular por un año
    }
    simular_inventario(params_inventario)