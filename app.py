from flask import Flask, render_template, request
from fase2 import generar_datos_aleatorios
from modelo import preprocesar_datos
import pandas as pd
import joblib

app = Flask(__name__)

contenido = joblib.load('modelo_suicidio.pkl')
modelo_cargado = contenido['modelo']  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fase1')
def fase1():
    return render_template('fase1.html')

@app.route('/fase2')
def generar_y_mostrar_datos():
    datos_aleatorios = generar_datos_aleatorios() 
    return render_template('fase2.html', datos=datos_aleatorios.to_dict(orient='records'))

@app.route('/fase3')
def fase3():
    return render_template('fase3.html')

@app.route('/modelo')
def modelo_html():
    return render_template('Modelo.html')

@app.route('/predecir', methods=['GET', 'POST'])
def predecir():
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

    df_usuario = pd.DataFrame([datos_usuario])
    df_preprocesado = preprocesar_datos(df_usuario)
    prediccion = modelo_cargado.predict(df_preprocesado)[0]

    return render_template('resultado.html', alerta=prediccion)

if __name__ == '__main__':
    app.run(debug=True)
