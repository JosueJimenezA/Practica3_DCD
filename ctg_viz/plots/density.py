import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_violin(df: pd.DataFrame, x: str, y: str) -> go.Figure:
    """
    Violin plot interactivo que muestra la densidad y los puntos subyacentes.
    Reemplaza al Swarmplot estático para grandes volúmenes de datos.
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