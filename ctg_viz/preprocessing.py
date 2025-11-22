import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Union, Optional

def remove_null_columns(df: pd.DataFrame, threshold: float = 0.2) -> pd.DataFrame:
    null_percentages = df.isnull().mean()
    cols_to_keep = null_percentages[null_percentages <= threshold].index
    return df[cols_to_keep].copy()

def impute_missing_values(df: pd.DataFrame, use_knn: bool = False) -> pd.DataFrame:
    df_clean = df.copy()
    
    # Identificamos columnas numéricas y categóricas explícitamente
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns

    if use_knn:
        from sklearn.impute import KNNImputer
        
        # 1. Tratamiento Numérico con KNN
        # Solo entramos si hay columnas numéricas para evitar errores
        if len(numeric_cols) > 0:
            imputer = KNNImputer(n_neighbors=5)
            df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
        
        # 2. Tratamiento Categórico
        # KNN no imputa categóricos, así que debemos usar la Moda para estas columnas
        for col in categorical_cols:
             if not df_clean[col].mode().empty:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                
    else:
        # Numéricos -> Mediana
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
        # Categóricos -> Moda
        for col in categorical_cols:
            if not df_clean[col].mode().empty:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                    
    return df_clean

def detect_handle_outliers(
    df: pd.DataFrame, 
    method: str = 'iqr', 
    return_plots: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, plt.Figure]]]:
    """
        df (pd.DataFrame): DataFrame numérico.
        method (str): 'iqr' o 'z-score'.
        return_plots (bool): Si True, devuelve una tupla (DataFrame, Diccionario de Figuras).
                             Si False, devuelve solo el DataFrame.

    """
    df_out = df.copy()
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns
    generated_plots = {}
    
    sns.set_theme(style="whitegrid")

    for col in numeric_cols:
        original_data = df_out[col].copy()
        
        lower_bound, upper_bound = 0, 0
        if method == 'iqr':
            Q1 = df_out[col].quantile(0.25)
            Q3 = df_out[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        elif method == 'z-score':
            mean = df_out[col].mean()
            std = df_out[col].std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            
        df_out[col] = np.clip(df_out[col], lower_bound, upper_bound)
        
        # Graficos
        if return_plots:
            n_outliers = sum((original_data < lower_bound) | (original_data > upper_bound))
            
            if n_outliers > 0:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Gráfico Antes
                sns.boxplot(y=original_data, ax=axes[0], color="salmon")
                axes[0].set_title(f"Original: {col}\n({n_outliers} outliers detectados)")
                
                # Gráfico Después
                sns.boxplot(y=df_out[col], ax=axes[1], color="skyblue")
                axes[1].set_title(f"Procesado: {col}\n(Clipping aplicado)")
                
                # Título general y ajuste
                plt.suptitle(f"Tratamiento de Outliers: {col} ({method.upper()})")
                plt.tight_layout()
                
                # la imagen solo se guarda en el objeto, no en el notebook
                plt.close(fig) 
                
                generated_plots[col] = fig

    if return_plots:
        return df_out, generated_plots
    
    return df_out