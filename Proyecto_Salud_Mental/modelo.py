import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from fase2 import generar_datos_aleatorios

def asignar_alerta(df):
    df = df.copy()
    df['Alerta'] = df.apply(
        lambda fila: 1 if (
            fila['Antecedente Depresión Familiar'] == 'Sí' or
            fila['Pérdida Familiar Reciente'] == 'Sí' or
            fila['Trauma'] == 'Sí' or
            fila['Alcoholismo'] in ['Regular', 'Dependencia'] or
            fila['Problemas Económicos'] == 'Severos'
        ) else 0, axis=1
    )
    return df

def preprocesar_datos(df):
    df_codificado = df.copy()
    for columna in df_codificado.columns:
        if df_codificado[columna].dtype == 'object':
            le = LabelEncoder()
            df_codificado[columna] = le.fit_transform(df_codificado[columna])
    return df_codificado

def entrenar_y_guardar_modelo():

    df = generar_datos_aleatorios(500)
    df = asignar_alerta(df)
    df = preprocesar_datos(df)
    X = df.drop(columns=['Alerta'])
    y = df['Alerta']
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train, y_train)
    
    joblib.dump(modelo, 'modelo_suicidio.pkl')
    print("Modelo guardado como modelo_suicidio.pkl")

if __name__ == "__main__":
    entrenar_y_guardar_modelo()
