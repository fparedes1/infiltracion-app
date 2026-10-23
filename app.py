import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Infiltración Multi-Ensayo - Kostiakov", page_icon="💧", layout="wide")

st.title("💧 Evaluación de Infiltración del Agua en el Suelo (Multi-Medición)")
st.markdown("### Modelo de Kostiakov ($L = A \\cdot T^n$)")

st.sidebar.header("📂 Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo Excel (.xlsx) o CSV", type=["xlsx", "csv"])

# Datos de ejemplo por defecto (2 ensayos)
default_ensayo_1 = pd.DataFrame({
    "T (min)": [1, 3, 5, 10, 15, 20, 30, 40, 50, 60],
    "L (cm)": [0.3, 0.4, 0.7, 0.95, 1.0, 1.2, 1.3, 1.5, 1.6, 1.7]
})

default_ensayo_2 = pd.DataFrame({
    "T (min)": [1, 2, 3, 4, 5, 10, 15, 20, 25, 30],
    "L (cm)": [0.4, 0.5, 0.5, 0.6, 0.7, 0.9, 1.2, 1.4, 1.7, 1.7]
})

def calcular_kostiakov(df_data):
    """Calcula coeficientes A, n, R2 e Ib de Kostiakov a partir de un DataFrame."""
    T = df_data.iloc[:, 0].values.astype(float)
    L = df_data.iloc[:, 1].values.astype(float)
    
    mask = (T > 0) & (L > 0)
    T_clean = T[mask]
    L_clean = L[mask]
    
    if len(T_clean) < 3:
        return None
        
    X = np.log10(T_clean)
    Y = np.log10(L_clean)
    
    N = len(T_clean)
    sum_X, sum_Y = np.sum(X), np.sum(Y)
    sum_XY, sum_X2 = np.sum(X * Y), np.sum(X**2)
    
    n = (N * sum_XY - sum_X * sum_Y) / (N * sum_X2 - sum_X**2)
    log_A = (sum_Y - n * sum_X) / N
    A = 10**log_A
    
    # R2
    Y_pred_log = log_A + n * X
    ss_res = np.sum((Y - Y_pred_log)**2)
    ss_tot = np.sum((Y - np.mean(Y))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Curvas
    T_smooth = np.linspace(T_clean.min(), T_clean.max(), 100)
    L_smooth = A * (T_smooth**n)
    Iv_smooth = 60 * n * A * (T_smooth**(n - 1))
    
    return {
        "A": A, "n": n, "R2": r2, "Ib": Iv_smooth[-1],
        "T_clean": T_clean, "L_clean": L_clean,
        "T_smooth": T_smooth, "L_smooth": L_smooth, "Iv_smooth": Iv_smooth,
        "X": X, "Y": Y, "Y_pred_log": Y_pred_log, "log_A": log_A
    }

# Lectura de datos según archivo o defecto
ensayos_dict = {}

if uploaded_file is not None:
    if uploaded_file.name.endswith('.xlsx'):
        excel_file = pd.ExcelFile(uploaded_file)
        # Si tiene la columna "Ensayo" en la primera hoja
        df_first = pd.read_excel(excel_file, sheet_name=excel_file.sheet_names[0])
        if "Ensayo" in df_first.columns:
            for nombre_ensayo, grupo in df_first.groupby("Ensayo"):
                ensayos_dict[str(nombre_ensayo)] = grupo.drop(columns=["Ensayo"])
        else:
            # Caso por hojas/solapas
            for sheet in excel_file.sheet_names:
                df_sheet = pd.read_excel(excel_file, sheet_name=sheet)
                if not df_sheet.empty and df_sheet.shape[1] >= 2:
                    ensayos_dict[sheet] = df_sheet
    else:
        df_csv = pd.read_csv(uploaded_file)
        if "Ensayo" in df_csv.columns:
            for nombre_ensayo, grupo in df_csv.groupby("Ensayo"):
                ensayos_dict[str(nombre_ensayo)] = grupo.drop(columns=["Ensayo"])
        else:
            ensayos_dict["Ensayo Unico"] = df_csv
else:
    ensayos_dict["Ensayo_1 (Lote A)"] = default_ensayo_1
    ensayos_dict["Ensayo_2 (Lote B)"] = default_ensayo_2

st.sidebar.markdown("---")
selected_ensayo = st.sidebar.selectbox("🎯 Seleccionar Ensayo a Analizar", list(ensayos_dict.keys()))

# Procesar todos los ensayos para resumen comparativo
resumen_list = []
calculos_dict = {}

for nombre, df_ens in ensayos_dict.items():
    res = calcular_kostiakov(df_ens)
    if res is not None:
        calculos_dict[nombre] = res
        resumen_list.append({
            "Ensayo / Pestaña": nombre,
            "Coeficiente A": round(res["A"], 4),
            "Exponente n": round(res["n"], 4),
            "Ecuación Kostiakov": f"L = {res['A']:.4f} * T^{res['n']:.4f}",
            "R²": round(res["R2"], 4),
            "Infiltración Básica Ib (cm/h)": round(res["Ib"], 2)
        })

# Vista Resumen Comparativo
st.subheader("📋 Tabla Comparativa de Resultados")
if resumen_list:
    df_resumen = pd.DataFrame(resumen_list)
    st.dataframe(df_resumen, use_container_width=True)
else:
    st.error("No se pudieron procesar las mediciones. Revisa la estructura del archivo.")

st.markdown("---")
st.subheader(f"🔍 Análisis Detallado: {selected_ensayo}")

df_actual = ensayos_dict[selected_ensayo]
col_tabla, col_grafico = st.columns([1, 2])

with col_tabla:
    st.markdown("**Tabla de Datos Cargados**")
    edited_df = st.data_editor(df_actual, num_rows="dynamic", use_container_width=True, key=selected_ensayo)

res_actual = calcular_kostiakov(edited_df)

if res_actual is not None:
    with col_grafico:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Coeficiente A", f"{res_actual['A']:.4f}")
        c2.metric("Exponente n", f"{res_actual['n']:.4f}")
        c3.metric("Coeficiente R²", f"{res_actual['R2']:.4f}")
        c4.metric("Inf. Básica Ib", f"{res_actual['Ib']:.2f} cm/h")

        tab1, tab2, tab3 = st.tabs(["Lámina ($L$ vs $T$)", "Velocidad ($I_v$ vs $T$)", "Ajuste Logarítmico"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(res_actual["T_clean"], res_actual["L_clean"], color='blue', label='Datos Campo', s=50)
            ax.plot(res_actual["T_smooth"], res_actual["L_smooth"], color='red', label=f'L = {res_actual["A"]:.3f} T^{res_actual["n"]:.3f}')
            ax.set_xlabel('Tiempo T (min)')
            ax.set_ylabel('Lámina L (cm)')
            ax.set_title('Infiltración Acumulada')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)
            
        with tab2:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(res_actual["T_smooth"], res_actual["Iv_smooth"], color='green', label='Iv (cm/h)')
            ax.set_xlabel('Tiempo T (min)')
            ax.set_ylabel('Velocidad Iv (cm/h)')
            ax.set_title('Velocidad de Infiltración')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)

        with tab3:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(res_actual["X"], res_actual["Y"], color='purple', label='Log(T) vs Log(L)')
            ax.plot(res_actual["X"], res_actual["Y_pred_log"], color='orange', label=f'Y = {res_actual["n"]:.3f}X + {res_actual["log_A"]:.3f}')
            ax.set_xlabel('Log10(T)')
            ax.set_ylabel('Log10(L)')
            ax.set_title('Ajuste Logarítmico')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)
