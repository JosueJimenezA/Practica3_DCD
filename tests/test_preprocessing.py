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
    data = {
        'col_good': [10, 12, 11, 10, 12],
        'col_nulls': [1, None, None, None, 5], 
        'col_outlier': [5, 5, 6, 5, 1000], 
        'col_cat': ['A', 'B', 'A', None, 'A']
    }
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
    # El umbral es 0.2 (20%). col_nulls tiene 60% de nulos.
    df_clean = remove_null_columns(sample_df, threshold=0.2)
    
    # Aserciones
    assert 'col_nulls' not in df_clean.columns, "La columna con exceso de nulos no fue eliminada."
    assert 'col_good' in df_clean.columns, "Una columna válida fue eliminada incorrectamente."

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
    df_imputed = impute_missing_values(sample_df, use_knn=True)
    
    # Verificar que ya no hay nulos en la columna categórica
    assert df_imputed['col_cat'].isnull().sum() == 0, "Quedaron valores nulos en columna categórica."
    
    # Verificar que el valor imputado es la moda ('A')
    # El índice 3 era el Nulo original
    assert df_imputed.loc[3, 'col_cat'] == 'A', "La imputación no usó la moda correctamente."

def test_detect_handle_outliers_iqr(sample_df):
    """
    Valida el recorte (clipping) de valores atípicos usando el Rango Intercuartílico (IQR).

    Escenario:
        La columna 'col_outlier' tiene un valor extremo (1000) muy lejos de la distribución normal (5-6).
        Se aplica el método 'iqr'.
    
    Resultado Esperado:
        - El valor 1000 debe ser reducido al límite superior calculado (aprox 7.5).
        - El valor no debe ser eliminado (la fila persiste), solo transformado.
    """
    df_out = detect_handle_outliers(sample_df, method='iqr')
    
    # El valor original era 1000. Después del clipping por IQR, debe ser mucho menor.
    # (El rango normal es aprox 5-6, el límite superior será cerca de 7.5)
    assert df_out.loc[4, 'col_outlier'] < 1000, "El outlier no fue reducido/tratado."
    assert df_out.loc[4, 'col_outlier'] > 5, "El valor tratado es demasiado bajo."

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
    
    # Verificar que retorna un DataFrame
    assert isinstance(summary, pd.DataFrame)
    
    # Verificar columnas esperadas en el reporte
    expected_cols = ["Nulos", "% Completitud", "Tipo Dato", "Categoría Auto"]
    for col in expected_cols:
        assert col in summary.columns, f"Falta la columna {col} en el reporte."
        
    # Verificar clasificación (col_good tiene <10 valores únicos, debe ser Discreta)
    # Nota: Ajusta esto según tu lógica exacta de 'continuo' vs 'discreto'
    assert summary.loc['col_good', 'Categoría Auto'] == 'Discreta'