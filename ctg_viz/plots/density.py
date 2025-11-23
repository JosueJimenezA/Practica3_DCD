import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_violin(df: pd.DataFrame, x: str, y: str) -> go.Figure:
    """
    Permite crear un violin plot interactivo que muestra la densidad y los puntos subyacentes.
    
    Args:
        df (pd.DataFrame): Dataset con los datos.
        x (str): Nombre de la columna categórica para el eje X.
        y (str): Nombre de la columna numérica para el eje Y.
    Returns:
        go.Figure: Objeto de figura de Plotly con el violin plot.
    """
    fig = px.violin(
        df, 
        y=y, 
        x=x, 
        color=x, 
        box=True, 
        points="all",
        hover_data=df.columns,
        title=f"Densidad y Dispersión de {y} por {x}",
        template="plotly_white"
    )
    return fig