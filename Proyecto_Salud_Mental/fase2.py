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

def generar_antecedente_depresion(prob_si=0.27):
    return random.choices(['Sí', 'No'], weights=[prob_si, 1 - prob_si])[0]

def generar_perdida_familiar(prob_si=0.58):
    return random.choices(['Sí', 'No'], weights=[prob_si, 1 - prob_si])[0]

def generar_trauma(prob_si=0.39):
    return random.choices(['Sí', 'No'], weights=[prob_si, 1 - prob_si])[0]

def generar_alcoholismo(distribucion={'No': 0.05, 'Ocasional': 0.30, 'Regular': 0.55, 'Dependencia': 0.10}):
    niveles = list(distribucion.keys())
    probabilidades = list(distribucion.values())
    return random.choices(niveles, weights=probabilidades)[0]

def generar_problemas_economicos(distribucion={'No': 0.56, 'Leves': 0.34, 'Severos': 0.10}):
    niveles = list(distribucion.keys())
    probabilidades = list(distribucion.values())
    return random.choices(niveles, weights=probabilidades)[0]

def generar_cambios_estilo_vida(distribucion={'No': 0.68, 'Leves': 0.23, 'Significativos': 0.09}):
    niveles = list(distribucion.keys())
    probabilidades = list(distribucion.values())
    return random.choices(niveles, weights=probabilidades)[0]

def generar_problemas_relaciones(distribucion={'Buenas': 0.57, 'Leves': 0.25, 'Severas': 0.18}):
    niveles = list(distribucion.keys())
    probabilidades = list(distribucion.values())
    return random.choices(niveles, weights=probabilidades)[0]

def generar_datos_aleatorios(num_muestras=100): 
    datos_aleatorios = [] 
    for _ in range(num_muestras):
        edad = generar_edad()
        genero = generar_genero()
        estado_civil = generar_estado_civil()
        ubicacion = generar_ubicacion()
        antecedente_depresion = generar_antecedente_depresion()
        perdida_familiar = generar_perdida_familiar()
        trauma = generar_trauma()
        alcoholismo = generar_alcoholismo()
        problemas_economicos = generar_problemas_economicos()
        cambios_estilo_vida = generar_cambios_estilo_vida()
        problemas_relaciones = generar_problemas_relaciones()

        muestra = {
            'Edad': edad,
            'Género': genero,
            'Estado Civil': estado_civil,
            'Ubicación': ubicacion,
            'Antecedente Depresión Familiar': antecedente_depresion,
            'Pérdida Familiar Reciente': perdida_familiar,
            'Trauma': trauma,
            'Alcoholismo': alcoholismo,
            'Problemas Económicos': problemas_economicos,
            'Cambios Estilo de Vida': cambios_estilo_vida,
            'Problemas Relaciones': problemas_relaciones,
        }
        datos_aleatorios.append(muestra) 

    df_aleatorio = pd.DataFrame(datos_aleatorios) 
    return df_aleatorio