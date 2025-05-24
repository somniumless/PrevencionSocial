import random
import pandas as pd

def generar_edad(min_edad=18, max_edad=85):
    return random.randint(min_edad, max_edad)

def generar_genero(prob_mujer=0.46):
    return random.choices(['Mujer', 'Hombre'], weights=[prob_mujer, 1 - prob_mujer])[0]

def generar_estado_civil(distribucion={'Soltero/a': 0.40, 'Casado/a': 0.35, 'Divorciado/a': 0.15, 'Viudo/a': 0.05, 'Unión libre': 0.05}):
    estados_civiles = list(distribucion.keys())
    probabilidades = list(distribucion.values())
    return random.choices(estados_civiles, weights=probabilidades)[0]

def generar_ubicacion(prob_urbano=0.75):
    return random.choices(['Urbano', 'Rural'], weights=[prob_urbano, 1 - prob_urbano])[0]

def generar_antecedente_depresion():
    return random.choices(['Sí', 'No'], weights=[0.20, 0.80])[0]

def generar_perdida_familiar():
    return random.choices(['Sí', 'No'], weights=[0.30, 0.70])[0]

def generar_trauma():
    return random.choices(['Sí', 'No'], weights=[0.20, 0.80])[0]

def generar_alcoholismo():
    return random.choices(['No', 'Ocasional', 'Regular', 'Dependencia'],
                          weights=[0.60, 0.30, 0.07, 0.03])[0]

def generar_problemas_economicos():
    return random.choices(['No', 'Leves', 'Severos'], weights=[0.70, 0.20, 0.10])[0]

def generar_cambios_estilo_vida():
    return random.choices(['No', 'Leves', 'Significativos'], weights=[0.75, 0.20, 0.05])[0]

def generar_problemas_relaciones():
    return random.choices(['Buenas', 'Leves', 'Severas'], weights=[0.70, 0.20, 0.10])[0]

def asignar_alerta(df):
    df = df.copy() 
    
    puntajes_riesgo = {
        'Antecedente Depresión Familiar': {'Sí': 0.5},
        'Pérdida Familiar Reciente': {'Sí': 1.5},
        'Trauma': {'Sí': 1},
        'Alcoholismo': {'Ocasional': 0.5, 'Regular': 1.5, 'Dependencia': 2.5},
        'Problemas Económicos': {'Leves': 1, 'Severos': 2},
        'Cambios Estilo de Vida': {'Leves': 0.5, 'Significativos': 1},
        'Problemas Relaciones': {'Leves': 1, 'Severas': 1.5}
    }
    umbral_alerta = 5.5 

    df['Puntaje_Riesgo_Calculado'] = 0.0

    for index, fila in df.iterrows():
        puntaje_actual = 0.0
        
        if fila['Antecedente Depresión Familiar'] == 'Sí':
            puntaje_actual += puntajes_riesgo['Antecedente Depresión Familiar']['Sí']
        
        if fila['Pérdida Familiar Reciente'] == 'Sí':
            puntaje_actual += puntajes_riesgo['Pérdida Familiar Reciente']['Sí']
            
        if fila['Trauma'] == 'Sí':
            puntaje_actual += puntajes_riesgo['Trauma']['Sí']
            
        if fila['Alcoholismo'] in puntajes_riesgo['Alcoholismo']:
            puntaje_actual += puntajes_riesgo['Alcoholismo'][fila['Alcoholismo']]
            
        if fila['Problemas Económicos'] in puntajes_riesgo['Problemas Económicos']:
            puntaje_actual += puntajes_riesgo['Problemas Económicos'][fila['Problemas Económicos']]
            
        if fila['Cambios Estilo de Vida'] in puntajes_riesgo['Cambios Estilo de Vida']:
            puntaje_actual += puntajes_riesgo['Cambios Estilo de Vida'][fila['Cambios Estilo de Vida']]
            
        if fila['Problemas Relaciones'] in puntajes_riesgo['Problemas Relaciones']:
            puntaje_actual += puntajes_riesgo['Problemas Relaciones'][fila['Problemas Relaciones']]
            
        df.loc[index, 'Puntaje_Riesgo_Calculado'] = puntaje_actual
    
    df['Alerta'] = df['Puntaje_Riesgo_Calculado'].apply(lambda x: 1 if x >= umbral_alerta else 0)
    
    return df

def generar_datos_aleatorios(num_muestras=1000):
    datos_raw = []
    for _ in range(num_muestras):
        muestra = {
            'Edad': generar_edad(),
            'Género': generar_genero(),
            'Estado Civil': generar_estado_civil(),
            'Ubicación': generar_ubicacion(),
            'Antecedente Depresión Familiar': generar_antecedente_depresion(),
            'Pérdida Familiar Reciente': generar_perdida_familiar(),
            'Trauma': generar_trauma(),
            'Alcoholismo': generar_alcoholismo(),
            'Problemas Económicos': generar_problemas_economicos(),
            'Cambios Estilo de Vida': generar_cambios_estilo_vida(),
            'Problemas Relaciones': generar_problemas_relaciones(),
        }
        datos_raw.append(muestra)
    
    df_aleatorio = pd.DataFrame(datos_raw)
    
    df_aleatorio = asignar_alerta(df_aleatorio)
    
    return df_aleatorio