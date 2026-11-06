import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

import motor

st.set_page_config(page_title="Siniestros viales · Corrientes", page_icon="🗺️", layout="wide")
st.title("🗺️ Puntos calientes de siniestros y recomendación de infraestructura")
st.caption("Corrientes Capital · Haz clic en un punto caliente para ver qué tipo de siniestro predomina "
           "y qué infraestructura podría ayudar a prevenirlo.")

COLOR_GRAV = {"FATAL": "#c0392b", "CON HERIDOS": "#f39c12", "SOLO DAÑOS": "#2e86c1"}
COLOR_PRIO = {"ALTA": "🔴", "MEDIA": "🟠", "BAJA": "🟢"}

# ------------------------------------------------------------------ datos
st.sidebar.header("Datos")
archivo = st.sidebar.file_uploader("df_final.csv (del notebook)", type=["csv"])


@st.cache_data
def leer(origen):
    return pd.read_csv(origen)


if archivo is not None:
    crudo = leer(archivo)
elif os.path.exists("df_final.csv"):
    crudo = leer("df_final.csv")
else:
    st.info("Sube `df_final.csv` en la barra lateral (el notebook lo genera en la última celda de código) "
            "o ponlo junto a `app.py` en el repositorio.")
    st.stop()

df_todo = motor.preparar(crudo)
df_geo = motor.con_coordenadas(df_todo)

st.sidebar.header("Detección de zonas")
radio = st.sidebar.slider("Radio de agrupación (m)", 50, 500, 200, 25)
min_acc = st.sidebar.slider("Mínimo de siniestros por zona", 2, 10, 3)
ver_puntos = st.sidebar.checkbox("Mostrar siniestros individuales", value=True)

df = motor.detectar_zonas(df_geo, radio, min_acc)
zonas = motor.resumen_zonas(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Siniestros en el archivo", len(df_todo))
c2.metric("Ubicados en el mapa", len(df_geo))
c3.metric("Puntos calientes", len(zonas))
c4.metric("Siniestros en puntos calientes", int((df["zona"] >= 0).sum()))

if zonas.empty:
    st.warning("No se detectaron puntos calientes con esos parámetros. Sube el radio o baja el mínimo.")
    st.stop()

# ------------------------------------------------------------------ mapa
col_mapa, col_info = st.columns([3, 2])

with col_mapa:
    m = folium.Map(location=[df_geo["latitud"].mean(), df_geo["longitud"].mean()],
                   zoom_start=13, tiles="cartodbpositron")
    if ver_puntos:
        for _, r in df.iterrows():
            folium.CircleMarker(
                [r.latitud, r.longitud], radius=3, weight=0, fill=True, fill_opacity=0.7,
                fill_color=COLOR_GRAV.get(r.clasificacion_gravedad, "#7f8c8d"),
                tooltip=f"Siniestro {r.get('id_feu', '')} · {r.tipo_colision} · {r.clasificacion_gravedad}",
            ).add_to(m)
    mx = zonas["siniestros"].max()
    for _, z in zonas.iterrows():
        folium.CircleMarker(
            [z.lat, z.lon], radius=12 + 18 * z.siniestros / mx,
            color="#922b21", weight=2, fill=True, fill_color="#e74c3c", fill_opacity=0.35,
            tooltip=f"Zona {int(z.zona)} · {z.calle}",
        ).add_to(m)
    mapa = st_folium(m, height=620, use_container_width=True,
                     returned_objects=["last_object_clicked", "last_object_clicked_tooltip"])
    st.caption("🔴 círculo grande = punto caliente · puntos: 🔴 fatal · 🟠 con heridos · 🔵 solo daños")

# ------------------------------------------------------------------ clic -> zona
zona_sel = None
tt = (mapa or {}).get("last_object_clicked_tooltip") or ""
mt = re.match(r"Zona (\d+)", tt)
if mt:
    zona_sel = int(mt.group(1))
elif not tt and (mapa or {}).get("last_object_clicked"):
    p = mapa["last_object_clicked"]
    d = np.hypot(zonas["lat"] - p["lat"], zonas["lon"] - p["lng"])
    if d.min() < 0.003:  # ~300 m
        zona_sel = int(zonas.loc[d.idxmin(), "zona"])

# ------------------------------------------------------------------ panel
with col_info:
    if zona_sel is None or zona_sel not in set(zonas["zona"]):
        st.info("👆 Haz clic en un círculo rojo grande del mapa.")
        st.markdown("**Ranking de puntos calientes**")
        tabla = zonas[["zona", "calle", "siniestros", "tipo", "fatales", "con_heridos"]].rename(
            columns={"zona": "Zona", "calle": "Calle principal", "siniestros": "Siniestros",
                     "tipo": "Tipo predominante", "fatales": "Fatales", "con_heridos": "Con heridos"})
        st.dataframe(tabla, hide_index=True, use_container_width=True)
    else:
        sub = df[df["zona"] == zona_sel]
        info = zonas[zonas["zona"] == zona_sel].iloc[0]
        a = motor.analizar(sub)

        st.subheader(f"📍 Zona {zona_sel} · {info.calle}")
        x, y = st.columns(2)
        x.metric("Siniestros registrados", a["n"])
        y.metric("Tipo predominante", a["tipo"], f"{a['pct_tipo']:.0%}" if a["pct_tipo"] else None, delta_color="off")
        st.markdown(f"**Prioridad sugerida:** {COLOR_PRIO[a['prioridad']]} {a['prioridad']}")

        st.markdown("**Tipos de colisión en la zona**")
        st.bar_chart(a["conteo"])

        with st.expander("Gravedad y características", expanded=True):
            for h in a["hallazgos"]:
                st.write(f"- {h}")

        st.success("🛠️ Infraestructura recomendada")
        for medida, motivo in a["medidas"]:
            st.markdown(f"✅ **{medida}**  \n<small>{motivo}</small>", unsafe_allow_html=True)

        for aviso in a["avisos"]:
            st.caption(f"⚠️ {aviso}")

        with st.expander("Ver los siniestros de esta zona"):
            cols = [c for c in ["id_feu", "momento_exacto", "direccion_completa", "tipo_colision",
                                "clasificacion_gravedad", "configuracion_de_la_via", "semaforo",
                                "fuente_coordenada"] if c in sub.columns]
            st.dataframe(sub[cols], hide_index=True, use_container_width=True)
