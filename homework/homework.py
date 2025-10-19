# flake8: noqa: E501# flake8: noqa: E501

##

# En este dataset se desea pronosticar el default (pago) del cliente el próximo# En este dataset se desea pronosticar el default (pago) del cliente el próximo

# mes a partir de 23 variables explicativas.# mes a partir de 23 variables explicativas.

##

#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el

#              credito familiar (suplementario).#              credito familiar (suplementario).

#         SEX: Genero (1=male; 2=female).#         SEX: Genero (1=male; 2=female).

#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).

#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).

#         AGE: Edad (years).#         AGE: Edad (years).

#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.

#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.

#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.

#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.

#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.

#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.

#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.

#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.

#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.

#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.

#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.

#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.

#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.

#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.

#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.

#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.

#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.

#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.

##

# La variable "default payment next month" corresponde a la variable objetivo.# La variable "default payment next month" corresponde a la variable objetivo.

##

# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba

# en la carpeta "files/input/".# en la carpeta "files/input/".

##

# Los pasos que debe seguir para la construcción de un modelo de# Los pasos que debe seguir para la construcción de un modelo de

# clasificación están descritos a continuación.# clasificación están descritos a continuación.

##

##

# Paso 1.# Paso 1.

# Realice la limpieza de los datasets:# Realice la limpieza de los datasets:

# - Renombre la columna "default payment next month" a "default".# - Renombre la columna "default payment next month" a "default".

# - Remueva la columna "ID".# - Remueva la columna "ID".

# - Elimine los registros con informacion no disponible.# - Elimine los registros con informacion no disponible.

# - Para la columna EDUCATION, valores > 4 indican niveles superiores# - Para la columna EDUCATION, valores > 4 indican niveles superiores

#   de educación, agrupe estos valores en la categoría "others".#   de educación, agrupe estos valores en la categoría "others".

# - Renombre la columna "default payment next month" a "default"# - Renombre la columna "default payment next month" a "default"

# - Remueva la columna "ID".# - Remueva la columna "ID".

##

##

# Paso 2.# Paso 2.

# Divida los datasets en x_train, y_train, x_test, y_test.# Divida los datasets en x_train, y_train, x_test, y_test.

##

##

# Paso 3.# Paso 3.

# Cree un pipeline para el modelo de clasificación. Este pipeline debe# Cree un pipeline para el modelo de clasificación. Este pipeline debe

# contener las siguientes capas:# contener las siguientes capas:

# - Transforma las variables categoricas usando el método# - Transforma las variables categoricas usando el método

#   one-hot-encoding.#   one-hot-encoding.

# - Ajusta un modelo de bosques aleatorios (rando forest).# - Ajusta un modelo de bosques aleatorios (rando forest).

##

##

# Paso 4.# Paso 4.

# Optimice los hiperparametros del pipeline usando validación cruzada.# Optimice los hiperparametros del pipeline usando validación cruzada.

# Use 10 splits para la validación cruzada. Use la función de precision# Use 10 splits para la validación cruzada. Use la función de precision

# balanceada para medir la precisión del modelo.# balanceada para medir la precisión del modelo.

##

##

# Paso 5.# Paso 5.

# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".

# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.

##

##

# Paso 6.# Paso 6.

# Calcule las metricas de precision, precision balanceada, recall,# Calcule las metricas de precision, precision balanceada, recall,

# y f1-score para los conjuntos de entrenamiento y prueba.# y f1-score para los conjuntos de entrenamiento y prueba.

# Guardelas en el archivo files/output/metrics.json. Cada fila# Guardelas en el archivo files/output/metrics.json. Cada fila

# del archivo es un diccionario con las metricas de un modelo.# del archivo es un diccionario con las metricas de un modelo.

# Este diccionario tiene un campo para indicar si es el conjunto# Este diccionario tiene un campo para indicar si es el conjunto

# de entrenamiento o prueba. Por ejemplo:# de entrenamiento o prueba. Por ejemplo:

##

# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}

# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}

##

##

# Paso 7.# Paso 7.

# Calcule las matrices de confusion para los conjuntos de entrenamiento y# Calcule las matrices de confusion para los conjuntos de entrenamiento y

# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila

# del archivo es un diccionario con las metricas de un modelo.# del archivo es un diccionario con las metricas de un modelo.

# de entrenamiento o prueba. Por ejemplo:# de entrenamiento o prueba. Por ejemplo:

##

# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}

# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}

##

import os
import json
import gzip
import pickle
import zipfile
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.compose import ColumnTransformer


def load_data(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(zf.namelist()[0]) as f:
            df = pd.read_csv(f)
    
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    df.drop(columns=["ID"], inplace=True)
    df.dropna(inplace=True)
    df['EDUCATION'] = df['EDUCATION'].clip(upper=4)
    
    return df


def create_pipeline():
    numeric_features = [
        "LIMIT_BAL", "AGE", 
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1))
    ])


def train_model(X_train, y_train):
    model = create_pipeline()
    
    param_grid = {
        "classifier__n_estimators": [300, 500],
        "classifier__max_depth": [None, 25],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2],
        "classifier__max_features": ["sqrt", None]
    }
    
    total_combos = len(param_grid['classifier__n_estimators']) * len(param_grid['classifier__max_depth']) * len(param_grid['classifier__min_samples_split']) * len(param_grid['classifier__min_samples_leaf']) * len(param_grid['classifier__max_features'])
    total_fits = total_combos * 10
    print(f"Iniciando GridSearchCV con {total_combos} combinaciones × 10 folds = {total_fits} entrenamientos...")
    print(f"Tiempo estimado: 15-20 minutos\n")
    
    grid_search = GridSearchCV(
        model, 
        param_grid, 
        cv=10, 
        scoring="balanced_accuracy", 
        verbose=2,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nMejores parámetros: {grid_search.best_params_}")
    print(f"Mejor score: {grid_search.best_score_:.4f}")
    
    return grid_search


def evaluate_and_save(model, X_train, y_train, X_test, y_test, output_file):
    results = []
    
    for dataset, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        y_pred = model.predict(X)
        
        results.append({
            "type": "metrics",
            "dataset": dataset,
            "precision": float(precision_score(y, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
            "recall": float(recall_score(y, y_pred)),
            "f1_score": float(f1_score(y, y_pred))
        })
    
    for dataset, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        
        results.append({
            "type": "cm_matrix",
            "dataset": dataset,
            "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
            "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])}
        })
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    print("\nMétricas guardadas:")
    for r in results[:2]:
        print(f"  {r['dataset']}: acc={r['balanced_accuracy']:.3f}, f1={r['f1_score']:.3f}")


if __name__ == "__main__":
    MODEL_FILE = "files/models/model.pkl.gz"
    METRICS_FILE = "files/output/metrics.json"
    
    print("=" * 60)
    print("CARGANDO DATOS")
    print("=" * 60)
    train_df = load_data("files/input/train_data.csv.zip")
    test_df = load_data("files/input/test_data.csv.zip")
    print(f"✓ Train: {train_df.shape} | Test: {test_df.shape}\n")
    
    X_train = train_df.drop(columns=["default"])
    y_train = train_df["default"]
    X_test = test_df.drop(columns=["default"])
    y_test = test_df["default"]
    
    if os.path.exists(MODEL_FILE):
        print("=" * 60)
        print("CARGANDO MODELO EXISTENTE")
        print("=" * 60)
        with gzip.open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        print(f"✓ Modelo cargado desde {MODEL_FILE}\n")
    else:
        print("=" * 60)
        print("ENTRENANDO MODELO")
        print("=" * 60)
        model = train_model(X_train, y_train)
        
        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
        with gzip.open(MODEL_FILE, "wb") as f:
            pickle.dump(model, f)
        print(f"\n✓ Modelo guardado en {MODEL_FILE}\n")
    
    print("=" * 60)
    print("EVALUANDO MODELO")
    print("=" * 60)
    evaluate_and_save(model, X_train, y_train, X_test, y_test, METRICS_FILE)
    print(f"\n✓ Completado. Métricas en {METRICS_FILE}")
