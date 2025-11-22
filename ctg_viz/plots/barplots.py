import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_bar(df: pd.DataFrame, col: str, horizontal: bool = False) -> go.Figure:
    """
    Gráfico de barras interactivo ordenado por frecuencia.
    """

    counts = df[col].value_counts().reset_index()
    counts.columns = ['Categoría', 'Frecuencia']
    counts = counts.sort_values(by='Frecuencia', ascending=True if horizontal else False)
    
    if horizontal:
        fig = px.bar(
            counts, x='Frecuencia', y='Categoría',
            color='Frecuencia', 
            orientation='h',
            title=f"Frecuencia de {col} (Horizontal)",
            text_auto='.2s', 
            template="plotly_white"
        )
    else:
        fig = px.bar(
            counts, x='Categoría', y='Frecuencia',
            color='Frecuencia',
            title=f"Frecuencia de {col} (Vertical)",
            text_auto='.2s',
            template="plotly_white"
        )
        
    fig.update_traces(textposition='outside')
    return fig