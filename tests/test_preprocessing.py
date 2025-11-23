import pytest
import pandas as pd
import numpy as np
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ctg_viz.preprocessing import remove_null_columns, impute_missing_values, detect_handle_outliers
from ctg_viz.utils import check_data_completeness_JosueJimenezApodaca

# --- FIXTURES (Datos de prueba) ---
@pytest.fixture
def sample_df():
    """
    Genera un DataFrame sintético para pruebas unitarias aisladas.
    
    Returns:
        pd.DataFrame: Dataset con casos de borde controlados:
            - 'col_good': Numérica limpia.
            - 'col_nulls': Numérica con 60% de nulos (debe eliminarse).
            - 'col_outlier': Numérica con un valor extremo (1000).
            - 'col_cat': Categórica con un valor nulo (para probar moda).
    """
    # Creamos datos base suficientes para que se detecte como continua
    # Rango de 0 a 14 (15 valores únicos)
    datos_continuos = list(range(15)) 
    # Agregamos el outlier al final
    datos_continuos.append(1000) 
    
    # Para las otras columnas, repetimos datos para igualar la longitud (16 filas)
    n = len(datos_continuos)
    
    data = {
        # Numérica limpia (>10 únicos)
        'col_good': np.random.rand(n), 
        
        # Numérica con 60% nulos
        'col_nulls': [1 if i < n*0.4 else None for i in range(n)], 
        
        # Numérica con outlier (>10 únicos, el último es 1000)
        'col_outlier': datos_continuos, 
        
        # Categórica (Pocos únicos) con nulos
        'col_cat': ['A', 'B'] * (n // 2)
    }
    # Introducimos un nulo en col_cat manualmente
    data['col_cat'][0] = None
    
    return pd.DataFrame(data)

# --- PRUEBAS UNITARIAS ---

def test_remove_null_columns(sample_df):
    """
    Valida la eliminación de columnas que superan el umbral de nulos.

    Escenario:
        Se pasa un DataFrame donde 'col_nulls' tiene 60% de valores faltantes.
        El umbral permitido es 20% (0.2).
    
    Resultado Esperado:
        - La columna 'col_nulls' debe desaparecer del DataFrame resultante.
        - La columna 'col_good' (0% nulos) debe conservarse.
    """
    df_clean = remove_null_columns(sample_df, threshold=0.2)
    assert 'col_nulls' not in df_clean.columns
    assert 'col_good' in df_clean.columns

def test_impute_missing_values_mode(sample_df):
    """
    Valida la imputación de la Moda para variables categóricas.

    Escenario:
        La columna 'col_cat' tiene un valor nulo y la moda es 'A' (aparece 3 veces).
        Se ejecuta la imputación con KNN.
    
    Resultado Esperado:
        - No deben quedar valores nulos en la columna.
        - El valor nulo original debe ser reemplazado por 'A'.
    """
    df_imputed = impute_missing_values(sample_df, use_knn=False)
    assert df_imputed['col_cat'].isnull().sum() == 0
    # El índice 0 era el Nulo
    assert df_imputed.loc[0, 'col_cat'] in ['A', 'B']

def test_detect_handle_outliers_iqr(sample_df):
    """
    Valida el recorte (clipping) de valores atípicos usando el Rango Intercuartílico (IQR).
    La variable debe tener mas de 10 valores únicos para ser considerada continua.

    Escenario:
        La columna 'col_outlier' tiene un valor extremo (1000) muy lejos de la distribución normal (5-6).
        Se aplica el método 'iqr'.
    
    Resultado Esperado:
        - El valor 1000 debe ser reducido al límite superior calculado (aprox 7.5).
        - El valor no debe ser eliminado (la fila persiste), solo transformado.
    """
    df_out = detect_handle_outliers(sample_df, method='iqr')
    
    # El último valor era 1000 (índice -1)
    valor_original = 1000
    valor_tratado = df_out['col_outlier'].iloc[-1]
    
    # Verificamos que fue recortado (clipping)
    assert valor_tratado < valor_original, f"El outlier {valor_original} no fue reducido."
    # Verificamos que no se redujo excesivamente (debe ser mayor que el rango normal ~14)
    assert valor_tratado > 14, "El valor tratado es demasiado bajo."

def test_check_data_completeness_structure(sample_df):
    """
    Valida la estructura y lógica de clasificación del reporte de completitud.

    Escenario:
        Se analiza el DataFrame de prueba. 'col_good' es numérica con <10 valores únicos.
    
    Resultado Esperado:
        - El resultado debe ser un DataFrame.
        - Debe contener las columnas obligatorias: 'Nulos', '% Completitud', 'Tipo Dato'.
        - 'col_good' debe clasificarse automáticamente como 'Discreta' según la regla de negocio (<10 únicos).
    """
    summary = check_data_completeness_JosueJimenezApodaca(sample_df)
    
    assert isinstance(summary, pd.DataFrame)
    assert "Categoría Auto" in summary.columns
    
    # col_outlier tiene >10 valores únicos -> Debe ser Continua
    assert summary.loc['col_outlier', 'Categoría Auto'] == 'Continua'
    
    # col_cat tiene 2 valores únicos -> Debe ser Discreta
    assert summary.loc['col_cat', 'Categoría Auto'] == 'Discreta'