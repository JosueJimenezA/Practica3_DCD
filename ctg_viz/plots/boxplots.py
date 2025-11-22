import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Optional

def plot_boxplot(df: pd.DataFrame, x: str, y: str, facet_col: Optional[str] = None) -> go.Figure:
    """
    Boxplot interactivo con opción de faceting (subgráficos).
    
    Args:
        df (pd.DataFrame): Datos.
        x (str): Variable categórica (Eje X).
        y (str): Variable numérica (Eje Y).
        facet_col (str, optional): Variable para dividir en columnas.
    """
    fig = px.box(
        df, 
        x=x, 
        y=y, 
        color=x,
        facet_col=facet_col,
        points="outliers", # Mostrar solo outliers como puntos
        title=f"Distribución de {y} por {x}" + (f" (divido por {facet_col})" if facet_col else ""),
        template="plotly_white"
    )
    
    if facet_col:
        fig.update_xaxes(matches=None) 
        
    return fig