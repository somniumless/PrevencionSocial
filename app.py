from flask import Flask, render_template, request
from fase2 import generar_datos_aleatorios
from modelo import preprocesar_datos, evaluar_modelo_cargado, all_possible_categories
import pandas as pd
import joblib

app = Flask(__name__)

modelo_cargado = None
encoders_cargados = None
feature_columns_cargadas = None

prediction_threshold = 0.8

try:
    contenido = joblib.load('modelo_suicidio.pkl')
    modelo_cargado = contenido['modelo']
    encoders_cargados = contenido['encoders']
    feature_columns_cargadas = contenido['features']
    print("Modelo, encoders y lista de características cargados exitosamente.")
except FileNotFoundError:
    print("Error: 'modelo_suicidio.pkl' no encontrado. Asegúrate de ejecutar modelo.py primero para entrenar y guardar el modelo.")
    modelo_cargado = None
    encoders_cargados = None
    feature_columns_cargadas = None
except KeyError as e:
    print(f"Error al cargar componentes del modelo: {e}. Asegúrate de que 'modelo_suicidio.pkl' contiene 'modelo', 'encoders' y 'features'.")
    modelo_cargado = None
    encoders_cargados = None
    feature_columns_cargadas = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fase1')
def fase1():
    return render_template('fase1.html')

@app.route('/fase2')
def generar_y_mostrar_datos():
    num_muestras_a_generar = 500
    datos_aleatorios = generar_datos_aleatorios(num_muestras_a_generar)
    return render_template('fase2.html', datos=datos_aleatorios.to_dict(orient='records'))

@app.route('/fase3')
def fase3():
    return render_template('fase3.html')

@app.route('/modelo')
def modelo_html():
    return render_template('Modelo.html')

@app.route('/predecir', methods=['GET', 'POST'])
def predecir():
    if request.method == 'POST':
        datos_usuario = {
            'Edad': int(request.form['edad']),
            'Género': request.form['genero'],
            'Estado Civil': request.form['estado_civil'],
            'Ubicación': request.form['ubicacion'],
            'Antecedente Depresión Familiar': request.form['depresion'],
            'Pérdida Familiar Reciente': request.form['perdida'],
            'Trauma': request.form['trauma'],
            'Alcoholismo': request.form['alcoholismo'],
            'Problemas Económicos': request.form['economicos'],
            'Cambios Estilo de Vida': request.form['cambios'],
            'Problemas Relaciones': request.form['relaciones'],
        }

        prediccion = None

        if modelo_cargado and encoders_cargados and feature_columns_cargadas:
            df_input = pd.DataFrame([datos_usuario])
            df_aligned = df_input.reindex(columns=feature_columns_cargadas, fill_value=None)
            
            df_preprocesado = preprocesar_datos(df_aligned, encoders=encoders_cargados, is_training=False)
            
            probabilidades = modelo_cargado.predict_proba(df_preprocesado)[0]
            probabilidad_riesgo_temp = probabilidades[1]

            prediccion = 1 if probabilidad_riesgo_temp >= prediction_threshold else 0
            
        else:
            prediccion = "Error: Modelo o componentes no cargados."

        return render_template('resultado.html', alerta=prediccion)
    return render_template('Modelo.html')

@app.route('/evaluacion')
def mostrar_evaluacion_modelo():
    if modelo_cargado and encoders_cargados and feature_columns_cargadas:
        metrics = evaluar_modelo_cargado(modelo_cargado, encoders_cargados, feature_columns_cargadas)
        return render_template('evaluacion.html', metrics=metrics)
    else:
        return "<h1>Error: Modelo o componentes no disponibles para evaluación.</h1><p>Asegúrate de que 'modelo_suicidio.pkl' exista y se cargue correctamente.</p>"

if __name__ == '__main__':
    app.run(debug=True)