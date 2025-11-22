import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Optional

def plot_histogram_interactivo(df: pd.DataFrame, col: str, group_by: Optional[str] = None) -> go.Figure:
    """
    Histograma interactivo con gráfico marginal de caja (Boxplot superior).
    
    Args:
        df (pd.DataFrame): Datos.
        col (str): Variable numérica.
        group_by (str, optional): Variable categórica para agrupar colores.
    """
    fig = px.histogram(
        df, 
        x=col, 
        color=group_by,
        marginal="box",
        opacity=0.7,
        nbins=50,
        barmode="overlay",
        title=f"Distribución de {col}",
        template="plotly_white" 
    )
    
    fig.update_layout(bargap=0.1)
    return fig