"""
app.py
Aplicación Streamlit — Cross Docking MIP Solver
Caso LogiFast CR — Yu & Egbelu (2008)
"""

import io
import time
import pandas as pd
import streamlit as st
from solver import (
    leer_ts5_desde_texto,
    validar_balance,
    fusionar_datos,
    resolver_mip,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LogiFast CR — Cross Docking MIP",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# ESTILOS CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Encabezado principal */
    .main-header {
        background: linear-gradient(135deg, #1F3864 0%, #2E75B6 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p  { margin: 0.3rem 0 0 0; opacity: 0.85; font-size: 0.95rem; }

    /* Tarjetas de métricas */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #2E75B6;
        margin-bottom: 1rem;
    }
    .metric-card.green  { border-left-color: #1E8449; }
    .metric-card.orange { border-left-color: #C0521A; }
    .metric-card.purple { border-left-color: #6B2FA0; }
    .metric-value { font-size: 2rem; font-weight: 700; color: #1F3864; }
    .metric-label { font-size: 0.85rem; color: #666; margin-top: 0.2rem; }

    /* Encabezados de sección */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1F3864;
        border-bottom: 2px solid #2E75B6;
        padding-bottom: 0.4rem;
        margin: 1.2rem 0 0.8rem 0;
    }

    /* Badge de estado */
    .badge-optimal  { background:#D5F5E3; color:#1E8449;
                      padding:3px 10px; border-radius:20px; font-weight:600; }
    .badge-infeasible { background:#FADBD8; color:#922B21;
                        padding:3px 10px; border-radius:20px; font-weight:600; }

    /* Secuencia de camiones */
    .truck-seq {
        display: flex; flex-wrap: wrap; gap: 8px;
        align-items: center; margin: 0.5rem 0;
    }
    .truck-chip-in {
        background: #D6E4F7; color: #1F3864;
        border: 2px solid #2E75B6;
        border-radius: 8px; padding: 6px 14px;
        font-weight: 700; font-size: 0.95rem;
    }
    .truck-chip-out {
        background: #D5F5E3; color: #1E5631;
        border: 2px solid #1E8449;
        border-radius: 8px; padding: 6px 14px;
        font-weight: 700; font-size: 0.95rem;
    }
    .arrow { font-size: 1.3rem; color: #999; }

    /* Nota al pie */
    .footnote { font-size: 0.78rem; color: #888; font-style: italic;
                margin-top: 0.5rem; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #F0F4FA; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def construir_df_matriz(datos: dict, tipo: str) -> pd.DataFrame:
    """Convierte r(i,n) o s(o,n) en DataFrame para mostrar."""
    I, O, N = datos['I'], datos['O'], datos['N']
    r, s    = datos['r'], datos['s']
    prods   = list(range(1, N + 1))

    if tipo == 'r':
        filas = list(range(1, I + 1))
        label_fila = [f"Camión i={i}" for i in filas]
        vals  = [[r.get((i, n), 0) for n in prods] for i in filas]
    else:
        filas = list(range(1, O + 1))
        label_fila = [f"Camión o={o}" for o in filas]
        vals  = [[s.get((o, n), 0) for n in prods] for o in filas]

    cols = [f"n={n}" for n in prods]
    df   = pd.DataFrame(vals, index=label_fila, columns=cols)
    df['TOTAL'] = df.sum(axis=1)
    total_row   = df.sum(axis=0)
    total_row.name = "TOTAL"
    df = pd.concat([df, total_row.to_frame().T])
    return df


def df_tiempos_entrada(resultado: dict) -> pd.DataFrame:
    seq = resultado['secuencia_entrada']
    te  = resultado['tiempos_entrada']
    rows = []
    for pos, i in enumerate(seq, 1):
        ci = te[i]['c']
        fi = te[i]['F']
        rows.append({
            'Posición en secuencia': pos,
            'Camión de entrada': f'i = {i}',
            'Entrada al muelle  c(i)  (min)': ci,
            'Salida del muelle  F(i)  (min)': fi,
            'Tiempo en muelle  (min)': round(fi - ci, 2),
        })
    return pd.DataFrame(rows)


def df_tiempos_salida(resultado: dict) -> pd.DataFrame:
    seq = resultado['secuencia_salida']
    ts  = resultado['tiempos_salida']
    rows = []
    for pos, o in enumerate(seq, 1):
        do_ = ts[o]['d']
        lo  = ts[o]['L']
        rows.append({
            'Posición en secuencia': pos,
            'Camión de salida': f'o = {o}',
            'Entrada al muelle  d(o)  (min)': do_,
            'Salida del muelle  L(o)  (min)': lo,
            'Tiempo en muelle  (min)': round(lo - do_, 2),
        })
    return pd.DataFrame(rows)


def df_asignaciones(resultado: dict) -> pd.DataFrame:
    rows = []
    for (i, o, n), u in sorted(resultado['asignacion'].items()):
        rows.append({
            'Camión entrada': f'i = {i}',
            'Camión salida': f'o = {o}',
            'Producto': f'n = {n}',
            'Unidades  x(i,o,n)': u,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def secuencia_html(seq: list, chip_class: str, prefijo: str) -> str:
    chips = []
    for k, idx in enumerate(seq):
        chips.append(f'<span class="{chip_class}">{prefijo}={idx}</span>')
        if k < len(seq) - 1:
            chips.append('<span class="arrow">→</span>')
    return (
        '<div class="truck-seq">'
        + ''.join(chips)
        + '</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuración del modelo")

    st.markdown("### 📂 Datos base (TS5)")
    archivo_base = st.file_uploader(
        "Cargar archivo TS5 base",
        type=["txt"],
        key="base",
        help="Archivo con formato: i N  o N  n N  r i n qty  s o n qty"
    )

    st.markdown("---")
    st.markdown("### ➕ Datos adicionales (opcional)")
    st.markdown(
        "<small>Agrega camiones y productos extra. "
        "Se fusionan automáticamente con el TS5 base.</small>",
        unsafe_allow_html=True
    )
    archivo_extra = st.file_uploader(
        "Cargar archivo adicional (mismo formato TS5)",
        type=["txt"],
        key="extra",
        help="Los camiones se renumeran automáticamente para evitar conflictos."
    )

    st.markdown("---")
    st.markdown("### 🔧 Parámetros operativos")
    D   = st.number_input("D — Cambio entre camiones (min)", value=10.0,
                           min_value=0.0, step=1.0)
    V   = st.number_input("V — Traslado recepción→despacho (min)", value=5.0,
                           min_value=0.0, step=1.0)
    tau = st.number_input("τ — Carga/descarga por unidad (min)", value=1.0,
                           min_value=0.1, step=0.1)
    tlim = st.slider("Tiempo límite del solver (seg)", 30, 600, 300, 30)

    st.markdown("---")
    resolver_btn = st.button("🚀  Resolver modelo MIP", use_container_width=True,
                             type="primary")


# ─────────────────────────────────────────────────────────────────────────────
# ENCABEZADO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
  <h1>🚛 Cross Docking MIP Solver — LogiFast CR</h1>
  <p>Modelo de Programación Lineal Mixta · Yu & Egbelu (2008) ·
     Minimización del makespan de operación</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PESTAÑAS
# ─────────────────────────────────────────────────────────────────────────────

tab_resumen, tab_matrices, tab_asignacion, tab_ayuda = st.tabs([
    "📊 Resumen de resultados",
    "📋 Matrices de datos",
    "🔗 Asignación x(i,o,n)",
    "❓ Ayuda y modelo",
])


# ─────────────────────────────────────────────────────────────────────────────
# ESTADO DE SESIÓN
# ─────────────────────────────────────────────────────────────────────────────

if 'resultado' not in st.session_state:
    st.session_state['resultado'] = None
if 'datos'     not in st.session_state:
    st.session_state['datos']     = None


# ─────────────────────────────────────────────────────────────────────────────
# LÓGICA DE RESOLUCIÓN
# ─────────────────────────────────────────────────────────────────────────────

if resolver_btn:
    # Verificar que haya un archivo base
    if archivo_base is None:
        st.sidebar.error("⚠️ Debes cargar al menos el archivo TS5 base.")
        st.stop()

    # Leer datos base
    contenido_base = archivo_base.read().decode('utf-8')
    datos = leer_ts5_desde_texto(contenido_base)

    # Fusionar con datos adicionales si existen
    if archivo_extra is not None:
        contenido_extra = archivo_extra.read().decode('utf-8')
        datos_extra = leer_ts5_desde_texto(contenido_extra)
        datos = fusionar_datos(datos, datos_extra)
        st.sidebar.success(
            f"✓ Datos fusionados: {datos['I']} camiones entrada, "
            f"{datos['O']} salida, {datos['N']} productos"
        )

    # Validar balance
    ok, errores = validar_balance(datos)
    if not ok:
        st.sidebar.error("❌ Desbalance de inventario detectado:")
        for e in errores:
            st.sidebar.warning(e)
        st.stop()

    # Resolver
    with st.spinner("⏳ Resolviendo modelo MIP (CBC solver)..."):
        t0 = time.time()
        resultado = resolver_mip(datos, D=D, V=V, tau=tau,
                                 tiempo_limite=tlim, gap=0.01)
        t_solver = round(time.time() - t0, 2)
        resultado['t_solver'] = t_solver

    st.session_state['resultado'] = resultado
    st.session_state['datos']     = datos

    if resultado['makespan'] is None:
        st.sidebar.error(f"❌ Sin solución: {resultado['estado']}")
    else:
        st.sidebar.success(
            f"✅ Optimal — Makespan = {resultado['makespan']} min "
            f"({t_solver}s)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PESTAÑA 1: RESUMEN
# ─────────────────────────────────────────────────────────────────────────────

with tab_resumen:
    resultado = st.session_state.get('resultado')

    if resultado is None:
        st.info(
            "👈 Carga el archivo TS5 en el panel izquierdo y presiona "
            "**Resolver modelo MIP** para ver los resultados aquí."
        )
        st.stop()

    if resultado['makespan'] is None:
        st.error(f"El solver no encontró solución factible. Estado: {resultado['estado']}")
        st.stop()

    # ── Métricas principales ──────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{resultado['makespan']:.1f} min</div>
          <div class="metric-label">⏱ Makespan óptimo (tiempo mínimo de operación)</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card green">
          <div class="metric-value">{resultado['I']}</div>
          <div class="metric-label">🚛 Camiones de entrada (i)</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card orange">
          <div class="metric-value">{resultado['O']}</div>
          <div class="metric-label">🚚 Camiones de salida (o)</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card purple">
          <div class="metric-value">{resultado['N']}</div>
          <div class="metric-label">📦 Tipos de producto (n)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Secuencias óptimas ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔢 Secuencias óptimas de atención</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Muelle de recepción — camiones de entrada (i)**")
        st.markdown(
            secuencia_html(resultado['secuencia_entrada'], "truck-chip-in", "i"),
            unsafe_allow_html=True
        )
        st.markdown(
            '<p class="footnote">El camión a la izquierda descarga primero.</p>',
            unsafe_allow_html=True
        )

    with col_b:
        st.markdown("**Muelle de despacho — camiones de salida (o)**")
        st.markdown(
            secuencia_html(resultado['secuencia_salida'], "truck-chip-out", "o"),
            unsafe_allow_html=True
        )
        st.markdown(
            '<p class="footnote">El camión a la izquierda carga primero.</p>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── Tabla camiones de entrada ─────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">📥 Tiempos — Muelle de recepción (camiones de entrada i)</div>',
        unsafe_allow_html=True
    )
    df_in = df_tiempos_entrada(resultado)
    st.dataframe(
        df_in.style
             .format({'Entrada al muelle  c(i)  (min)': '{:.2f}',
                      'Salida del muelle  F(i)  (min)': '{:.2f}',
                      'Tiempo en muelle  (min)': '{:.2f}'})
             .set_properties(**{'text-align': 'center'})
             .highlight_max(subset=['Tiempo en muelle  (min)'],
                            color='#FDEBD0')
             .highlight_min(subset=['Tiempo en muelle  (min)'],
                            color='#D5F5E3'),
        use_container_width=True,
        hide_index=True
    )

    # ── Tabla camiones de salida ──────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">📤 Tiempos — Muelle de despacho (camiones de salida o)</div>',
        unsafe_allow_html=True
    )
    df_out = df_tiempos_salida(resultado)
    st.dataframe(
        df_out.style
              .format({'Entrada al muelle  d(o)  (min)': '{:.2f}',
                       'Salida del muelle  L(o)  (min)': '{:.2f}',
                       'Tiempo en muelle  (min)': '{:.2f}'})
              .set_properties(**{'text-align': 'center'})
              .highlight_max(subset=['Tiempo en muelle  (min)'],
                             color='#FDEBD0')
              .highlight_min(subset=['Tiempo en muelle  (min)'],
                             color='#D5F5E3'),
        use_container_width=True,
        hide_index=True
    )

    # ── Parámetros usados ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">⚙️ Parámetros utilizados</div>',
                unsafe_allow_html=True)
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    col_p1.metric("D — Cambio de camión", f"{resultado['D']} min")
    col_p2.metric("V — Traslado interno", f"{resultado['V']} min")
    col_p3.metric("τ — Por unidad", f"{resultado['tau']} min")
    col_p4.metric("Tiempo solver", f"{resultado.get('t_solver', '—')} seg")

    # ── Descarga de resultados ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">⬇️ Exportar resultados</div>',
                unsafe_allow_html=True)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        csv_in = df_in.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar tiempos entrada (.csv)",
                           csv_in, "tiempos_entrada.csv", "text/csv")
    with col_d2:
        csv_out = df_out.to_csv(index=False).encode('utf-8')
        st.download_button("📤 Descargar tiempos salida (.csv)",
                           csv_out, "tiempos_salida.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# PESTAÑA 2: MATRICES
# ─────────────────────────────────────────────────────────────────────────────

with tab_matrices:
    datos = st.session_state.get('datos')
    if datos is None:
        st.info("Resuelve el modelo primero para ver las matrices aquí.")
        st.stop()

    st.markdown('<div class="section-header">Matriz r(i,n) — Unidades por camión de entrada</div>',
                unsafe_allow_html=True)
    df_r = construir_df_matriz(datos, 'r')
    st.dataframe(
        df_r.style
            .highlight_max(axis=None, color='#D6E4F7')
            .format(lambda x: '—' if x == 0 else int(x)),
        use_container_width=True
    )

    st.markdown('<div class="section-header">Matriz s(o,n) — Unidades requeridas por camión de salida</div>',
                unsafe_allow_html=True)
    df_s = construir_df_matriz(datos, 's')
    st.dataframe(
        df_s.style
            .highlight_max(axis=None, color='#D5F5E3')
            .format(lambda x: '—' if x == 0 else int(x)),
        use_container_width=True
    )

    # Balance
    ok, errores = validar_balance(datos)
    if ok:
        total = sum(datos['r'].values())
        st.success(f"✅ Balance de inventario verificado — {total} unidades totales en el sistema.")
    else:
        for e in errores:
            st.error(e)


# ─────────────────────────────────────────────────────────────────────────────
# PESTAÑA 3: ASIGNACIÓN
# ─────────────────────────────────────────────────────────────────────────────

with tab_asignacion:
    resultado = st.session_state.get('resultado')
    if resultado is None or resultado['makespan'] is None:
        st.info("Resuelve el modelo primero.")
        st.stop()

    st.markdown('<div class="section-header">Asignación óptima x(i,o,n) — Unidades transferidas</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Cada fila indica cuántas unidades del producto **n** deben transferirse "
        "del camión de entrada **i** al camión de salida **o**."
    )

    df_asig = df_asignaciones(resultado)
    if df_asig.empty:
        st.warning("No se encontraron asignaciones.")
    else:
        # Filtros interactivos
        col_f1, col_f2, col_f3 = st.columns(3)
        entradas_disp = sorted(df_asig['Camión entrada'].unique())
        salidas_disp  = sorted(df_asig['Camión salida'].unique())
        prods_disp    = sorted(df_asig['Producto'].unique())

        with col_f1:
            sel_i = st.multiselect("Filtrar camión entrada",
                                   entradas_disp, default=entradas_disp)
        with col_f2:
            sel_o = st.multiselect("Filtrar camión salida",
                                   salidas_disp, default=salidas_disp)
        with col_f3:
            sel_n = st.multiselect("Filtrar producto",
                                   prods_disp, default=prods_disp)

        df_filt = df_asig[
            df_asig['Camión entrada'].isin(sel_i) &
            df_asig['Camión salida'].isin(sel_o) &
            df_asig['Producto'].isin(sel_n)
        ]
        st.dataframe(df_filt, use_container_width=True, hide_index=True)
        st.markdown(
            f"**Total unidades asignadas:** "
            f"{df_filt['Unidades  x(i,o,n)'].sum():,}"
        )

        csv_asig = df_filt.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Descargar asignaciones (.csv)",
                           csv_asig, "asignaciones.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# PESTAÑA 4: AYUDA
# ─────────────────────────────────────────────────────────────────────────────

with tab_ayuda:
    st.markdown("## ❓ Guía de uso y modelo matemático")

    with st.expander("📁 Formato del archivo TS5", expanded=True):
        st.markdown("""
El archivo debe ser un `.txt` con columnas separadas por tabulación o espacio:

```
i   5               ← número de camiones de entrada
o   3               ← número de camiones de salida
n   8               ← número de tipos de producto
r   1   1   170     ← camión entrada i=1, producto n=1, cantidad=170
r   2   1   6       ← camión entrada i=2, producto n=1, cantidad=6
s   1   1   75      ← camión salida  o=1, producto n=1, cantidad=75
s   2   1   150     ← camión salida  o=2, producto n=1, cantidad=150
```

**Regla de balance:** para cada producto n, la suma de todas las filas `r` debe
igualar la suma de todas las filas `s`. De lo contrario el modelo es infactible.
        """)

    with st.expander("➕ Cómo agregar datos extra"):
        st.markdown("""
Carga un segundo archivo con el **mismo formato TS5** en el campo
**Datos adicionales** del panel izquierdo.

La aplicación renumera automáticamente los camiones del archivo extra
para evitar conflictos. Por ejemplo, si el TS5 base tiene i=1..5,
los camiones del archivo extra pasan a ser i=6, 7, ...

**Recuerda:** los datos extra también deben estar balanceados
(incluyendo lo que ya existe en el TS5 base).
        """)

    with st.expander("🔢 Variables de decisión del modelo"):
        st.markdown("""
| Variable | Tipo | Descripción |
|---|---|---|
| T | Continua | Makespan — tiempo total de operación (objetivo a minimizar) |
| c(i) | Continua | Tiempo de entrada del camión i al muelle de recepción |
| F(i) | Continua | Tiempo de salida del camión i del muelle de recepción |
| d(o) | Continua | Tiempo de entrada del camión o al muelle de despacho |
| L(o) | Continua | Tiempo de salida del camión o del muelle de despacho |
| x(i,o,n) | **Entera** | Unidades del producto n de camión i a camión o |
| v(i,o) | **Binaria** | 1 si hay transferencia entre camión i y camión o |
| p(i,i') | **Binaria** | 1 si camión i precede a camión i' en recepción |
| q(o,o') | **Binaria** | 1 si camión o precede a camión o' en despacho |
        """)

    with st.expander("📐 Parámetros operativos"):
        st.markdown("""
| Símbolo | Descripción | Valor LogiFast CR |
|---|---|---|
| D | Tiempo de cambio entre camiones en el mismo muelle | 10 min |
| V | Tiempo de traslado de recepción a despacho | 5 min |
| τ | Tiempo de carga/descarga por unidad de producto | 1 min/unidad |
        """)

    with st.expander("📚 Referencia"):
        st.markdown("""
Yu, W., & Egbelu, P. J. (2008). Scheduling of inbound and outbound trucks
in cross docking systems with temporary storage.
*European Journal of Operational Research, 184*(1), 377–396.
https://doi.org/10.1016/j.ejor.2006.10.047
        """)
