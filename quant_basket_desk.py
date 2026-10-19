import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Quant Basket Trading Desk v3.0", layout="wide", page_icon="🏀"
)


# ==========================================
# 1. CAPA DE BASE DE DATOS (PERSISTENCIA V3)
# ==========================================
def init_db():
    conn = sqlite3.connect("wallstreet_basket.db", check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT,
            partido TEXT,
            cuarto TEXT,
            min_restantes REAL,
            pts_actuales INTEGER,
            pts_q_anterior INTEGER,
            faltas_totales INTEGER,
            proyeccion_ia REAL,
            linea_mercado REAL,
            señal TEXT,
            edge REAL,
            cuota REAL,
            stake_pct REAL,
            pts_finales_reales INTEGER,
            resultado TEXT,
            pnl_unidades REAL
        )
    """)
    conn.commit()
    return conn


conn = init_db()


# ==========================================
# 2. MOTOR BAYESIANO V3 (CON INERCIA DE MITAD)
# ==========================================
def calcular_modelo_en_vivo_v3(
    t_total,
    t_rest,
    pts_act,
    r_pre,
    pts_q_anterior,
    faltas_eq1,
    faltas_eq2,
    racha_tiro,
    linea_mkt,
    cuota,
    modo_auditoria,
):
    t_jugado = max(0.5, t_total - t_rest)

    # 1. Factor de Inercia / Momentum de la Mitad Anterior
    # Si la media normal de un cuarto FIBA es 40 pts y el cuarto anterior tuvo 55 pts,
    # el factor de inercia empuja la expectativa base un porcentaje hacia arriba.
    media_teorica_cuarto = 40.0 if t_total == 10.0 else 56.0
    factor_inercia = pts_q_anterior / media_teorica_cuarto
    r_pre_ajustado = r_pre * (
        0.7 + (0.3 * factor_inercia)
    )  # Ponderación suave del ritmo previo

    # 2. Peso Bayesiano Temporal
    omega = (t_rest / t_total) ** 1.3

    # 3. Ritmo vivo proyectado y ajuste por regresión a la media de tiro
    ritmo_bruto_proyectado = (pts_act / t_jugado) * t_total
    factor_regresion = 1.0 - (0.45 * (racha_tiro / 100.0))
    r_live_adj = ritmo_bruto_proyectado * factor_regresion

    # 4. Penalización por Bonus de Faltas
    delta_bonus = 0.0
    for f in [faltas_eq1, faltas_eq2]:
        if f >= 4:
            min_en_bonus = max(0.0, t_rest - (4 - min(4, f)) * 1.8)
            delta_bonus += min_en_bonus * 0.48
        elif f == 3 and t_rest > 5.0:
            delta_bonus += (t_rest - 3.5) * 0.22

    # 5. Proyección Final integrando la Inercia del Partido
    ritmo_combinado = (omega * r_pre_ajustado) + ((1 - omega) * r_live_adj)
    proyeccion_final = (
        pts_act + ((t_rest / t_total) * ritmo_combinado) + delta_bonus
    )

    sigma_base = 6.2 if t_total == 12 else 5.2
    sigma_restante = sigma_base * np.sqrt(t_rest / t_total)

    # 6. Cálculo de Probabilidad y Edge
    diferencial = proyeccion_final - linea_mkt
    z_score = diferencial / max(0.5, sigma_restante)
    prob_over = 1 / (1 + np.exp(-1.702 * z_score))
    prob_under = 1.0 - prob_over

    umbral_edge = 0.5 if modo_auditoria else 1.5
    señal = "NO OPERAR (Sin Ventaja)"
    prob_exito = 0.5

    if abs(diferencial) >= umbral_edge:
        if diferencial > 0:
            señal = (
                "COMPRA FUERTE (OVER)"
                if diferencial >= 2.5
                else "COMPRA (OVER)"
            )
            prob_exito = prob_over
        else:
            señal = (
                "VENTA FUERTE (UNDER)"
                if diferencial <= -2.5
                else "VENTA (UNDER)"
            )
            prob_exito = prob_under

    # 7. Criterio de Kelly Fraccionado (0.25x)
    b = cuota - 1.0
    q = 1.0 - prob_exito
    kelly_puro = ((b * prob_exito) - q) / b if b > 0 else 0
    stake_recomendado = max(0.0, kelly_puro * 0.25 * 100)
    if modo_auditoria and stake_recomendado == 0.0 and abs(diferencial) >= 0.5:
        stake_recomendado = 1.0

    return (
        round(proyeccion_final, 2),
        round(sigma_restante, 2),
        round(diferencial, 2),
        señal,
        round(prob_exito * 100, 1),
        round(stake_recomendado, 2),
        round(r_pre_ajustado, 1),
    )


# ==========================================
# 3. INTERFAZ DEL TRADING DESK V3.0
# ==========================================
st.title(
    "🏛️ Wall Street Quant Desk v3.0: Live Basketball Quarters (Inercia de"
    " Mitad)"
)
tab1, tab2 = st.tabs([
    "⚡ Terminal de Ejecución en Vivo",
    "📊 Auditoría IA & Gestión",
])

with tab1:
    col_input, col_chart = st.columns([1, 2])

    with col_input:
        st.subheader("1. Parámetros del Partido & Inercia")
        liga_min = st.radio(
            "Duración del Cuarto",
            [12.0, 10.0],
            captions=["NBA (12 min)", "FIBA / Euroliga (10 min)"],
            horizontal=True,
        )

        default_pre = 56.5 if liga_min == 12.0 else 42.0

        partido = st.text_input(
            "Ticker / Partido", "CS Rapid Bucuresti vs BC CSU Sibiu"
        )
        cuarto = st.selectbox("Cuarto Actual", ["Q3", "Q4", "Q1", "Q2"])

        # Nuevo campo clave: ¿Cuántos puntos hubo en el cuarto inmediatamente anterior?
        pts_q_anterior = st.number_input(
            "Puntos Totales Cuarto Anterior (Ej. Q2 hizo 55 pts)", 20, 90, 55
        )

        c1, c2 = st.columns(2)
        t_rest = c1.slider(
            "Minutos Restantes",
            0.5,
            float(liga_min),
            float(liga_min - 0.5),
            0.25,
        )
        pts_act = c2.number_input("Puntos Actuales del Cuarto (Suma)", 0, 100, 2)

        c3, c4 = st.columns(2)
        r_pre = c3.number_input(
            "Expectativa Base Pre-Cuarto", 25.0, 75.0, default_pre, 0.5
        )
        linea_mkt = c4.number_input(
            "Línea Casa de Apuestas LIVE", 25.0, 90.0, 45.5, 0.5
        )

        c5, c6 = st.columns(2)
        f_eq1 = c5.number_input("Faltas Equipo Local", 0, 10, 1)
        f_eq2 = c6.number_input("Faltas Equipo Visita", 0, 10, 1)

        racha_tiro = st.slider("Desviación de Acierto en Vivo", -30, 30, 0)
        cuota = st.number_input(
            "Cuota del Mercado (Decimal)", 1.50, 2.50, 1.75, 0.01
        )
        modo_auditoria = st.checkbox("Modo Paper Trading / Registro", value=True)

        (
            proy,
            sigma,
            edge,
            señal,
            prob,
            stake,
            r_pre_adj,
        ) = calcular_modelo_en_vivo_v3(
            liga_min,
            t_rest,
            pts_act,
            r_pre,
            pts_q_anterior,
            f_eq1,
            f_eq2,
            racha_tiro,
            linea_mkt,
            cuota,
            modo_auditoria,
        )

    with col_chart:
        st.info(
            f"🔥 **Factor de Inercia Aplicado:** Como el cuarto anterior tuvo"
            f" **{pts_q_anterior} puntos**, la IA ajustó tu expectativa base de"
            f" `{r_pre}` a `🌿 {r_pre_adj} pts` para este cuarto."
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Proyección IA Final", f"{proy} Pts", f"{edge:+} vs Mercado")
        m2.metric("Línea Mercado Live", f"{linea_mkt} Pts")
        m3.metric("Probabilidad Real IA", f"{prob}%", señal)
        m4.metric("Tamaño Posición (Kelly)", f"{stake}% Banca")

        # GRÁFICO: CONO DE PROBABILIDAD
        t_jugado = liga_min - t_rest
        tiempos = np.linspace(t_jugado, liga_min, 20)
        fraccion_avance = (tiempos - t_jugado) / max(0.1, t_rest)

        trayectoria_central = pts_act + fraccion_avance * (proy - pts_act)
        banda_sup = trayectoria_central + (
            1.28 * sigma * np.sqrt(fraccion_avance)
        )
        banda_inf = trayectoria_central - (
            1.28 * sigma * np.sqrt(fraccion_avance)
        )

        fig_cone = go.Figure()
        fig_cone.add_trace(
            go.Scatter(
                x=np.concatenate([tiempos, tiempos[::-1]]),
                y=np.concatenate([banda_sup, banda_inf[::-1]]),
                fill="toself",
                fillcolor="rgba(0, 200, 255, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Cono IA",
            )
        )
        fig_cone.add_trace(
            go.Scatter(
                x=tiempos,
                y=trayectoria_central,
                mode="lines",
                line=dict(color="#00C8FF", width=3),
                name="Ruta IA",
            )
        )
        fig_cone.add_hline(
            y=linea_mkt,
            line_dash="dash",
            line_color="#FF4B4B",
            annotation_text=f"Línea Casa ({linea_mkt})",
            annotation_position="top left",
        )
        fig_cone.add_trace(
            go.Scatter(
                x=[t_jugado],
                y=[pts_act],
                mode="markers",
                marker=dict(size=12, color="#00FF7F"),
                name="Actual",
            )
        )

        fig_cone.update_layout(
            title="Cono de Probabilidad con Inercia de Mitad",
            xaxis_title="Minuto Transcurrido del Cuarto",
            yaxis_title="Puntos Acumulados",
            template="plotly_dark",
            height=340,
        )
        st.plotly_chart(fig_cone, use_container_width=True)

        # 2 PASOS DE EJECUCIÓN
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("### ⚡ PASO 1: Abrir Orden")
            if st.button(
                "🚀 REGISTRAR ENTRADA Q3",
                type="primary",
                use_container_width=True,
            ):
                c = conn.cursor()
                c.execute(
                    """INSERT INTO trades (fecha, partido, cuarto, min_restantes, pts_actuales, pts_q_anterior, faltas_totales, 
                                     proyeccion_ia, linea_mercado, señal, edge, cuota, stake_pct, pts_finales_reales, resultado, pnl_unidades)
                                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 'PENDIENTE', 0.0)""",
                    (
                        datetime.now().strftime("%Y-%m-%d %H:%M"),
                        partido,
                        cuarto,
                        t_rest,
                        pts_act,
                        pts_q_anterior,
                        f_eq1 + f_eq2,
                        proy,
                        linea_mkt,
                        señal,
                        edge,
                        cuota,
                        stake,
                    ),
                )
                conn.commit()
                st.toast("✅ Apuesta registrada con inercia de Q2 guardada.")
                st.rerun()

        with col_p2:
            st.markdown("### 🏁 PASO 2: Liquidar Cuarto")
            df_p = pd.read_sql_query(
                "SELECT * FROM trades WHERE resultado = 'PENDIENTE' ORDER BY id"
                " DESC",
                conn,
            )
            if len(df_p) == 0:
                st.info("Sin órdenes pendientes.")
            else:
                opciones = {
                    (
                        f"ID #{r['id']} | {r['partido']} ({r['cuarto']}) - Línea"
                        f" {r['linea_mercado']}"
                    ): r["id"]
                    for _, r in df_p.iterrows()
                }
                sel = st.selectbox("Selecciona orden a cerrar:", list(opciones.keys()))
                id_c = opciones[sel]
                pts_fin = st.number_input("Puntos Finales Q3:", 0, 120, 48)

                if st.button("💾 CERRAR Y CALCULAR P&L", use_container_width=True):
                    row = df_p[df_p["id"] == id_c].iloc[0]
                    es_ov = "OVER" in row["señal"] or (
                        "NO OPERAR" in row["señal"] and row["edge"] >= 0
                    )
                    if pts_fin == row["linea_mercado"]:
                        res, pnl = "PUSH", 0.0
                    elif (es_ov and pts_fin > row["linea_mercado"]) or (
                        not es_ov and pts_fin < row["linea_mercado"]
                    ):
                        res, pnl = "WIN", row["stake_pct"] * (row["cuota"] - 1.0)
                    else:
                        res, pnl = "LOSS", -row["stake_pct"]

                    c = conn.cursor()
                    c.execute(
                        "UPDATE trades SET pts_finales_reales = ?, resultado ="
                        " ?, pnl_unidades = ? WHERE id = ?",
                        (pts_fin, res, round(pnl, 2), id_c),
                    )
                    conn.commit()
                    st.success(f"Cerrado: {res} ({pnl:+.2f}%)")
                    st.rerun()

with tab2:
    st.subheader("Historial de Operaciones con Inercia")
    df_t = pd.read_sql_query("SELECT * FROM trades", conn)
    st.dataframe(df_t, use_container_width=True)