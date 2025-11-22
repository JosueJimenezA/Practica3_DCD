import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Optional

def plot_correlation_heatmap(df: pd.DataFrame, method: str = 'pearson') -> go.Figure:
    """
    Genera un Heatmap interactivo de correlación optimizado para muchas variables.
    
    Args:
        df (pd.DataFrame): Datos.
        method (str): Método de correlación ('pearson', 'spearman').
        
    Returns:
        go.Figure: Objeto figura de Plotly listo para Streamlit.
    """
    # Filtrar solo numéricos
    corr_matrix = df.select_dtypes(include=['number']).corr(method=method).round(2)
    
    dynamic_height = max(600, len(corr_matrix) * 25)
    
    fig = px.imshow(
        corr_matrix,
        text_auto=False,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title=f"Matriz de Correlación Interactiva ({method.capitalize()})"
    )
    
    fig.update_layout(
        height=dynamic_height,
        width=dynamic_height + 100, 
        autosize=False
    )
    
    fig.update_traces(hovertemplate='Variable X: %{x}<br>Variable Y: %{y}<br>Correlación: %{z}')
    
    return fig