import pandas as pd
import numpy as np

def check_data_completeness_JosueJimenezApodaca(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analiza el dataset y retorna un resumen de completitud, tipos y estadísticas.
    Cumple con los requisitos de conteo de nulos, porcentajes, tipos y 
    clasificación automática de variables continuas/discretas.
    
    Args:
        df (pd.DataFrame): El dataset a analizar.

    Returns:
        pd.DataFrame: Resumen con columnas: 
                      [Nulos, % Completitud, Tipo Dato, Estadísticas, Categoría Auto]
    """
    summary_data = []
    
    for col in df.columns:
        # 1. Conteo de nulos 
        null_count = df[col].isnull().sum()
        
        # 2. Porcentaje de completitud 
        # Completitud = 100% - %Nulos
        completeness = 100 * (1 - (null_count / len(df)))
        
        # 3. Tipo de dato 
        dtype = df[col].dtype
        
        # 4. Estadísticos de dispersión 
        # Solo aplican si es numérico. Si es texto, ponemos "N/A"
        stats = "N/A"
        if pd.api.types.is_numeric_dtype(df[col]):
            # Formato compacto: Min, Max, Desviación Estándar
            stats = f"Min:{df[col].min():.2f}, Max:{df[col].max():.2f}, Std:{df[col].std():.2f}"
            
        # 5. Clasificar automáticamente columnas 
        # - Continuas: Más de 10 valores únicos y tipo numérico 
        # - Discretas: Menos de 10 valores únicos 
        unique_vals = df[col].nunique()
        
        if pd.api.types.is_numeric_dtype(df[col]) and unique_vals > 10:
            category = "Continua"
        else:
            category = "Discreta"
            
        summary_data.append({
            "Columna": col,
            "Nulos": null_count,
            "% Completitud": round(completeness, 2),
            "Tipo Dato": str(dtype),
            "Estadísticas": stats,
            "Categoría Auto": category
        })
    
    # Convertimos la lista de diccionarios en un DataFrame y ponemos el nombre de columna como índice
    return pd.DataFrame(summary_data).set_index("Columna")