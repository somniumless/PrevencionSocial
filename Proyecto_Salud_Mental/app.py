from flask import Flask, render_template
from fase2 import generar_datos_aleatorios

app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=True)