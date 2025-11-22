import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

# Importamos nuestra librería personalizada
from ctg_viz.preprocessing import remove_null_columns, impute_missing_values, detect_handle_outliers
from ctg_viz.utils import check_data_completeness_JosueJimenezApodaca
from ctg_viz.plots.histograms import plot_histogram_interactivo
from ctg_viz.plots.boxplots import plot_boxplot
from ctg_viz.plots.barplots import plot_bar
from ctg_viz.plots.heatmap import plot_correlation_heatmap
from ctg_viz.plots.density import plot_violin

# Configuración de la página (Título y Layout ancho)
st.set_page_config(page_title="Dashboard CTG", layout="wide")

st.title("🏥 Dashboard de Análisis Cardiotocográfico (CTG)")
st.markdown("""
Esta aplicación permite analizar datos clínicos, limpiar valores atípicos 
y visualizar patrones utilizando la librería **`ctg_viz`**.
""")

# --- BARRA LATERAL (CONTROLES) ---
st.sidebar.header("1. Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type="csv")

# Cargar datos (o usar el default si no se sube nada)
@st.cache_data
def load_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Archivo cargado exitosamente.")
else:
    # Intenta cargar desde la carpeta data/ por defecto
    default_path = "data/CTG.csv" 
    df = load_data(default_path)
    if df is not None:
        st.sidebar.info(f"Usando dataset por defecto: {default_path}")
    else:
        st.error("Por favor sube un archivo CSV para comenzar.")
        st.stop()

# --- LÓGICA DE PROCESAMIENTO ---
st.sidebar.header("2. Preprocesamiento")
apply_clean = st.sidebar.checkbox("Aplicar Limpieza Automática", value=True)
knn_impute = st.sidebar.checkbox("Usar Imputación KNN", value=True)

if apply_clean:
    with st.spinner('Limpiando datos...'):
        # 1. Pipeline de limpieza usando la libreria personalizada
        df_clean = remove_null_columns(df, threshold=0.2)
        df_clean = impute_missing_values(df_clean, use_knn=knn_impute)
        # 2. Outliers (Pedimos los plots también)
        df_final, outlier_figs = detect_handle_outliers(df_clean, method='iqr', return_plots=True)
else:
    df_final = df.copy()
    outlier_figs = {}

# --- LÓGICA DE CLASIFICACIÓN (Global para toda la App) ---
reporte = check_data_completeness_JosueJimenezApodaca(df_final)

# Continuas (más de 10 valores únicos y tipo numérico)​
# Discretas (menos de 10 valores únicos)​
vars_continuas = reporte[reporte['Categoría Auto'] == 'Continua'].index.tolist()
vars_discretas = reporte[reporte['Categoría Auto'] == 'Discreta'].index.tolist()

# --- PESTAÑAS PRINCIPALES ---
tab1, tab2, tab3 = st.tabs(["📊 Resumen de Datos", "🧹 Calidad & Outliers", "📈 Visualización Interactiva"])

with tab1:
    st.header("Vista General del Dataset")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Filas", df_final.shape[0])
    with col2:
        st.metric("Columnas", df_final.shape[1], delta=df_final.shape[1] - df.shape[1])
    
    st.subheader("Primeras filas (Procesadas)")
    st.dataframe(df_final.head())
    
    st.subheader("Estadísticos Descriptivos")
    st.dataframe(df_final.describe())


with tab2:
    st.header("Reporte de Calidad y Limpieza")
    
    st.subheader("1. Clasificación y Completitud")
    st.write("Este reporte se generó automáticamente usando la función `check_data_completeness`.")
    
    st.dataframe(
        reporte.style.format({
            "% Completitud": "{:.2f}%",
            "Nulos": "{:.0f}"
        }).background_gradient(subset=["% Completitud"], cmap="RdYlGn", vmin=80, vmax=100)
    )
    
    st.divider()
    
    # Identificamos qué variables continuas tuvieron outliers y cuáles no
    outliers_relevantes = {k: v for k, v in outlier_figs.items() if k in vars_continuas}
    
    # Calculamos las que NO están en el diccionario de figuras (Diferencia de conjuntos)
    vars_sin_outliers = sorted([var for var in vars_continuas if var not in outliers_relevantes])
    
    st.subheader("2. Tratamiento de Outliers (Evidencia Visual)")
    
    if outliers_relevantes:
        st.warning(f"⚠️ Se detectaron y trataron valores atípicos en **{len(outliers_relevantes)}** variables continuas.")
        
        col_izq, col_der = st.columns([1, 3])
        
        with col_izq:
            col_sel = st.radio("Selecciona variable:", list(outliers_relevantes.keys()))
            
        with col_der:
            if col_sel:
                st.markdown(f"**Comparativa Antes vs. Después para `{col_sel}`**")
                st.pyplot(outliers_relevantes[col_sel])
    else:
        st.success("✅ No se detectaron outliers en ninguna variable continua (o la limpieza está desactivada).")

    st.divider()

    st.subheader("3. Variables Estables (Sin Outliers)")
    
    if vars_sin_outliers:
        st.success(f"✨ Las siguientes **{len(vars_sin_outliers)}** variables continuas tienen una distribución estable y no requirieron recorte por IQR:")
        
        st.pills("Variables Limpias", vars_sin_outliers, selection_mode="single")

    else:
        st.info("Todas las variables continuas presentaron al menos un valor atípico.")


with tab3:
    st.header("Explorador Gráfico Interactivo")
    
    plot_type = st.selectbox("Tipo de Gráfico", 
                             ["Histograma", "Boxplot", "Violin Plot", "Barras", "Heatmap Correlación"])
    
    # Lógica dinámica basada en tus listas oficiales
    if plot_type == "Histograma":
        # Solo permitimos variables CONTINUAS en el eje X
        col_dist = st.selectbox("Variable Numérica (Continua)", vars_continuas)
        
        # Solo permitimos variables DISCRETAS para agrupar (colores)
        # Agregamos [None] por si el usuario no quiere agrupar
        group = st.selectbox("Agrupar por (Discreta/Categórica)", [None] + vars_discretas)
        
        fig = plot_histogram_interactivo(df_final, col=col_dist, group_by=group)
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Boxplot":
        # Eje Y: Continuas (Métricas)
        col_y = st.selectbox("Variable Numérica (Y)", vars_continuas)
        # Eje X: Discretas (Grupos)
        col_x = st.selectbox("Categoría (X)", vars_discretas)
        # Facet: Discretas
        facet = st.selectbox("Separar por (Facet - Opcional)", [None] + vars_discretas)
        
        fig = plot_boxplot(df_final, x=col_x, y=col_y, facet_col=facet)
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Violin Plot":
        col_v_y = st.selectbox("Variable Y (Continua)", vars_continuas)
        col_v_x = st.selectbox("Variable X (Discreta)", vars_discretas)
        fig = plot_violin(df_final, x=col_v_x, y=col_v_y)
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Barras":
        # Las barras son inherentemente para contar variables DISCRETAS
        col_bar = st.selectbox("Variable Categórica", vars_discretas)
        horiz = st.checkbox("Horizontal", value=True)
        fig = plot_bar(df_final, col=col_bar, horizontal=horiz)
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Heatmap Correlación":
        method = st.radio("Método de Correlación", ["pearson", "spearman"])
        # El heatmap usa todo el dataframe numérico, no requiere inputs
        fig = plot_correlation_heatmap(df_final, method=method)
        st.plotly_chart(fig, use_container_width=True)

# Pie de página
st.sidebar.markdown("---")
st.sidebar.write("Desarrollado para Práctica 3 DCD - Josue Jimenez Apodaca")