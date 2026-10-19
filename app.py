import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Calculadora de Infiltración - Kostiakov", page_icon="💧", layout="wide")

st.title("💧 Aplicación Web: Evaluación de Infiltración de Agua en el Suelo")
st.markdown("### Modelo de Kostiakov ($L = A \\cdot T^n$) por Mínimos Cuadrados")

st.sidebar.header("📝 Ingreso de Datos")
st.sidebar.markdown("Modifica los valores en la tabla o sube tu archivo Excel/CSV.")

# Datos por defecto basados en el TP
default_data = pd.DataFrame({
    "T (min)": [1, 3, 5, 10, 15, 20, 30, 40, 50, 60],
    "L (cm)": [0.3, 0.4, 0.7, 0.95, 1.0, 1.2, 1.3, 1.5, 1.6, 1.7]
})

uploaded_file = st.sidebar.file_uploader("Sube tu archivo Excel o CSV", type=["xlsx", "csv"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df_input = pd.read_csv(uploaded_file)
    else:
        df_input = pd.read_excel(uploaded_file)
    st.sidebar.success("¡Archivo cargado con éxito!")
else:
    df_input = default_data

st.subheader("📊 Tabla de Datos de Ensayo")
edited_df = st.data_editor(df_input, num_rows="dynamic", use_container_width=True)

if len(edited_df) >= 3:
    T = edited_df.iloc[:, 0].values.astype(float)
    L = edited_df.iloc[:, 1].values.astype(float)
    
    mask = (T > 0) & (L > 0)
    T_clean = T[mask]
    L_clean = L[mask]
    
    if len(T_clean) >= 3:
        # Cálculo por Mínimos Cuadrados (Regresión Log-Log)
        X = np.log10(T_clean)
        Y = np.log10(L_clean)
        
        N = len(T_clean)
        sum_X = np.sum(X)
        sum_Y = np.sum(Y)
        sum_XY = np.sum(X * Y)
        sum_X2 = np.sum(X**2)
        
        n = (N * sum_XY - sum_X * sum_Y) / (N * sum_X2 - sum_X**2)
        log_A = (sum_Y - n * sum_X) / N
        A = 10**log_A
        
        # Coeficiente de determinación R²
        Y_pred_log = log_A + n * X
        ss_res = np.sum((Y - Y_pred_log)**2)
        ss_tot = np.sum((Y - np.mean(Y))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Curvas suavizadas
        T_smooth = np.linspace(T_clean.min(), T_clean.max(), 100)
        L_smooth = A * (T_smooth**n)
        Iv_smooth = 60 * n * A * (T_smooth**(n - 1))
        
        st.markdown("---")
        st.subheader("🎯 Resultados del Modelo de Kostiakov")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Coeficiente A", f"{A:.4f}")
        col2.metric("Exponente n", f"{n:.4f}")
        col3.metric("Coeficiente R²", f"{r2:.4f}")
        col4.metric("Infiltración Básica Ib", f"{Iv_smooth[-1]:.2f} cm/h")
        
        st.latex(f"L = {A:.4f} \\cdot T^{{{n:.4f}}}")
        
        st.markdown("---")
        st.subheader("📈 Gráficos de Comportamiento")
        
        tab1, tab2, tab3 = st.tabs(["Lámina Acumulada ($L$ vs $T$)", "Velocidad ($I_v$ vs $T$)", "Ajuste Logarítmico"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(T_clean, L_clean, color='blue', label='Datos de Campo', s=50)
            ax.plot(T_smooth, L_smooth, color='red', label=f'Kostiakov: L = {A:.3f} T^{{{n:.3f}}}', linewidth=2)
            ax.set_xlabel('Tiempo T (min)')
            ax.set_ylabel('Lámina Acumulada L (cm)')
            ax.set_title('Curva de Infiltración Acumulada')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)
            
        with tab2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(T_smooth, Iv_smooth, color='green', label='Velocidad Instantánea Iv', linewidth=2)
            ax.set_xlabel('Tiempo T (min)')
            ax.set_ylabel('Velocidad de Infiltración Iv (cm/h)')
            ax.set_title('Velocidad de Infiltración vs Tiempo')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)
            
        with tab3:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(X, Y, color='purple', label='Log(T) vs Log(L)', s=50)
            ax.plot(X, Y_pred_log, color='orange', label=f'Regresión: Y = {n:.3f}X + {log_A:.3f}', linewidth=2)
            ax.set_xlabel('Log10(T)')
            ax.set_ylabel('Log10(L)')
            ax.set_title('Ajuste por Mínimos Cuadrados')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)
            
    else:
        st.warning("Por favor ingrese al menos 3 registros válidos mayores a 0.")
else:
    st.warning("Ingrese al menos 3 filas de datos.")
