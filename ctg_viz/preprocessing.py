import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Union, Optional
sns.set_theme(style="whitegrid")

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Union, List

def _is_continuous(series: pd.Series, threshold: int = 10) -> bool:
    """
    Helper interno para determinar si una columna es 'Continua' 
    según la regla del proyecto: Numérica y > 10 valores únicos.
    """
    return pd.api.types.is_numeric_dtype(series) and series.nunique() > threshold

def remove_null_columns(df: pd.DataFrame, threshold: float = 0.2) -> pd.DataFrame:
    """
        Función para eliminar columnas con un porcentaje de valores nulos definido por el umbral.
        En este caso, se elimina si más del 20% de los datos son nulos.

        Args:
        df (pd.DataFrame): DataFrame numérico.
        method (str): 'iqr' o 'z-score'.
        
        returns:
        return_plots (bool): Si True, devuelve una tupla (DataFrame, Diccionario de Figuras).
                             Si False, devuelve solo el DataFrame.

    """
    null_percentages = df.isnull().mean()
    cols_to_keep = null_percentages[null_percentages <= threshold].index
    return df[cols_to_keep].copy()

def impute_missing_values(df: pd.DataFrame, use_knn: bool = False) -> pd.DataFrame:
    """

        Función para imputar valores faltantes en el DataFrame.
        Se puede usar KNN para numéricos, si no se aplica KNN aplica entonces Mediana por defecto.
        Para discretas siempre es la moda.
        Se incluye tratamiento separado para variables categóricas y numéricas.

        Args:
        df (pd.DataFrame): DataFrame numérico.
        method (str): 'iqr' o 'z-score'.
        
        Returns:
        return_plots (bool): Si True, devuelve una tupla (DataFrame, Diccionario de Figuras).
                             Si False, devuelve solo el DataFrame.
    
    Reglas:
    - Continuas (Numéricas > 10 valores): Usan Mediana o KNN.
    - Discretas (Numéricas <= 10 valores) y Categóricas: Usan MODA.
    """
    df_clean = df.copy()
    
    # 1. Identificar columnas basadas en la lógica de negocio
    cols_continuous = [c for c in df_clean.columns if _is_continuous(df_clean[c])]
    # El resto son discretas (incluye numéricas pequeñas y strings)
    cols_discrete = [c for c in df_clean.columns if c not in cols_continuous]

    # 2. Imputación de Continuas
    if use_knn and cols_continuous:
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=5)
        # KNN devuelve array, asignamos con cuidado para no perder índice
        df_clean[cols_continuous] = imputer.fit_transform(df_clean[cols_continuous])
    else:
        # Fallback a Mediana para continuas
        for col in cols_continuous:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # 3. Imputación de Discretas/Categóricas (SIEMPRE MODA)
    # Esto protege variables como NSP (1,2,3) de recibir decimales
    for col in cols_discrete:
        if not df_clean[col].mode().empty:
            moda = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(moda)
                    
    return df_clean


def detect_handle_outliers(
    df: pd.DataFrame, 
    method: str = 'iqr', 
    return_plots: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
        Función para detectar y manejar outliers en columnas numéricas.
        Se trunca los valores fuera de los límites definidos por el método seleccionado 
        Valores atípicos serán llevados al límite más cercano.

        Args:
        df (pd.DataFrame): DataFrame numérico.
        method (str): 'iqr' o 'z-score'.

        Returns:
        return_plots (bool): Si True, devuelve una tupla (DataFrame, Diccionario de Figuras).
                             Si False, devuelve solo el DataFrame.
    """
    df_out = df.copy()
    generated_plots = {}

    # 1. Seleccionar SOLO las variables continuas reales
    # Esto evita recortar variables categóricas codificadas como números
    cols_to_process = [c for c in df_out.columns if _is_continuous(df_out[c])]
    
    for col in cols_to_process:
        original_data = df_out[col].copy()
        
        # 2. Cálculo de límites
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
            
        # 3. Tratamiento (Clipping)
        df_out[col] = np.clip(df_out[col], lower_bound, upper_bound)
        
        # 4. Generación de Gráficos (Opcional)
        if return_plots:
            n_outliers = sum((original_data < lower_bound) | (original_data > upper_bound))
            
            if n_outliers > 0:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                sns.boxplot(y=original_data, ax=axes[0], color="salmon")
                axes[0].set_title(f"Original: {col}\n({n_outliers} outliers)")
                sns.boxplot(y=df_out[col], ax=axes[1], color="skyblue")
                axes[1].set_title(f"Procesado: {col}\n(Límites: {lower_bound:.2f} - {upper_bound:.2f})")
                plt.suptitle(f"Tratamiento Outliers: {col}")
                plt.tight_layout()
                plt.close(fig)
                generated_plots[col] = fig

    if return_plots:
        return df_out, generated_plots
    
    return df_out