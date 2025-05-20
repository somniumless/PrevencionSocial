import random
import pandas as pd

def generar_edad(min_edad=18, max_edad=85):
    return random.randint(min_edad, max_edad)

def generar_genero(prob_mujer=0.46):
    return random.choices(['Mujer', 'Hombre'], weights=[prob_mujer, 1 - prob_mujer])[0]

def generar_estado_civil():
    return random.choices(['Soltero/a', 'Casado/a', 'Divorciado/a', 'Viudo/a', 'Unión libre'],
                        weights=[0.40, 0.35, 0.15, 0.05, 0.05])[0]

def generar_ubicacion():
    return random.choices(['Urbano', 'Rural'], weights=[0.75, 0.25])[0]

def generar_antecedente_depresion():
    return random.choices(['Sí', 'No'], weights=[0.27, 0.73])[0]

def generar_perdida_familiar():
    return random.choices(['Sí', 'No'], weights=[0.58, 0.42])[0]

def generar_trauma():
    return random.choices(['Sí', 'No'], weights=[0.39, 0.61])[0]

def generar_alcoholismo():
    return random.choices(['No', 'Ocasional', 'Regular', 'Dependencia'],
                        weights=[0.05, 0.30, 0.55, 0.10])[0]

def generar_problemas_economicos():
    return random.choices(['No', 'Leves', 'Severos'], weights=[0.56, 0.34, 0.10])[0]

def generar_cambios_estilo_vida():
    return random.choices(['No', 'Leves', 'Significativos'], weights=[0.68, 0.23, 0.09])[0]

def generar_problemas_relaciones():
    return random.choices(['Buenas', 'Leves', 'Severas'], weights=[0.57, 0.25, 0.18])[0]

def asignar_alerta(muestra):
    if (
        muestra['Antecedente Depresión Familiar'] == 'Sí' or
        muestra['Pérdida Familiar Reciente'] == 'Sí' or
        muestra['Trauma'] == 'Sí' or
        muestra['Alcoholismo'] in ['Regular', 'Dependencia'] or
        muestra['Problemas Económicos'] == 'Severos'
    ):
        return 1
    else:
        return 0

def generar_datos_aleatorios(num_muestras=100):
    datos_aleatorios = []
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
        muestra['Alerta'] = asignar_alerta(muestra)
        datos_aleatorios.append(muestra)
    return pd.DataFrame(datos_aleatorios)
