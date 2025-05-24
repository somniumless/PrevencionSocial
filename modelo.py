import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from fase2 import generar_datos_aleatorios

all_possible_categories = {
    'Género': ['Mujer', 'Hombre'],
    'Estado Civil': ['Soltero/a', 'Casado/a', 'Divorciado/a', 'Viudo/a', 'Unión libre'],
    'Ubicación': ['Urbano', 'Rural'],
    'Antecedente Depresión Familiar': ['Sí', 'No'],
    'Pérdida Familiar Reciente': ['Sí', 'No'],
    'Trauma': ['Sí', 'No'],
    'Alcoholismo': ['No', 'Ocasional', 'Regular', 'Dependencia'],
    'Problemas Económicos': ['No', 'Leves', 'Severos'],
    'Cambios Estilo de Vida': ['No', 'Leves', 'Significativos'],
    'Problemas Relaciones': ['Buenas', 'Leves', 'Severas']
}

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

def preprocesar_datos(df, encoders=None, is_training=False):
    df_codificado = df.copy()
    fitted_encoders = {} if encoders is None else encoders
    
    for columna in df_codificado.columns:
        if columna in all_possible_categories:
            if is_training:
                le = LabelEncoder()
                le.fit(all_possible_categories[columna])
                df_codificado[columna] = le.transform(df_codificado[columna])
                fitted_encoders[columna] = le
            else:
                if columna in fitted_encoders:
                    df_codificado[columna] = fitted_encoders[columna].transform(df_codificado[columna])
                else:
                    raise ValueError(f"Encoder para la columna '{columna}' no encontrado. Asegúrate de cargar el modelo correctamente.")
        elif df_codificado[columna].dtype == 'object':
            pass

    if is_training:
        return df_codificado, fitted_encoders
    else:
        return df_codificado

def entrenar_y_guardar_modelo():
    print("Generando datos para entrenamiento...")
    df = generar_datos_aleatorios(1000)
    df = asignar_alerta(df) 

    X_raw = df.drop(columns=['Alerta', 'Puntaje_Riesgo_Calculado']) 
    y = df['Alerta']
    
    print(f"\n--- Distribución de la clase 'Alerta' en los datos de entrenamiento (y): ---")
    print(y.value_counts(normalize=True))
    print("---------------------------------------------------------------------\n")

    X, fitted_encoders = preprocesar_datos(X_raw, is_training=True)

    if y.nunique() > 1 and y.value_counts().min() < 2:
        print("Advertencia: La clase minoritaria tiene menos de 2 muestras. 'stratify' podría fallar.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Entrenando el modelo de Regresión Logística con GridSearchCV...")
    

    logistic_model = LogisticRegression(max_iter=50000, class_weight='balanced', random_state=42)

    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'] 
    }
    
    grid_search = GridSearchCV(estimator=logistic_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    print(f"Mejores parámetros encontrados: {grid_search.best_params_}")
    print(f"Mejor puntuación Accuracy en validación cruzada: {grid_search.best_score_:.4f}")

    joblib.dump({'modelo': best_model, 'encoders': fitted_encoders, 'features': X.columns.tolist()}, 'modelo_suicidio.pkl')
    print("Mejor modelo (con hiperparámetros optimizados) y encoders guardados como modelo_suicidio.pkl")

def evaluar_modelo_cargado(modelo_cargado, encoders, feature_columns):
    print("Generando datos para evaluación...")
    df_eval = generar_datos_aleatorios(200)
    df_eval = asignar_alerta(df_eval) 
    
    X_eval_raw = df_eval.drop(columns=['Alerta', 'Puntaje_Riesgo_Calculado'])
    y_true = df_eval['Alerta']

    for col in feature_columns:
        if col not in X_eval_raw.columns:
            X_eval_raw[col] = None
    X_eval_raw = X_eval_raw[feature_columns]

    X_eval_preprocesado = preprocesar_datos(X_eval_raw, encoders=encoders, is_training=False)
    
    y_pred = modelo_cargado.predict(X_eval_preprocesado)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0) 

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'tp': int(tp),
        'fn': int(fn),
        'fp': int(fp),
        'tn': int(tn)
    }
    print("Métricas de evaluación calculadas.")
    return metrics

if __name__ == "__main__":
    entrenar_y_guardar_modelo()