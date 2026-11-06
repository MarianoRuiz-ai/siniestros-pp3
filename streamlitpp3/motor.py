"""Lógica de la app: carga de datos, detección de puntos calientes y motor de recomendaciones.
Pensado para el df_final.csv que genera el notebook Trabajo_pp2_.ipynb."""
import re
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

SIN_DATO = "Sin dato"


# ---------------------------------------------------------------- datos
def preparar(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    # En el dataset la columna 'zona' es el nombre de la calle; 'zona' se reserva para el punto caliente
    if "zona" in df:
        df = df.rename(columns={"zona": "zona_calle"})
    for c in ["latitud", "longitud"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    if "momento_exacto" in df:
        df["momento_exacto"] = pd.to_datetime(df["momento_exacto"], errors="coerce")
        df["hora"] = df["momento_exacto"].dt.hour
    elif "hora" not in df:
        df["hora"] = np.nan
    df["tipo_colision"] = (df["tipo_colision"].fillna(SIN_DATO).astype(str).str.strip()
                           .replace({"": SIN_DATO, "S/D": SIN_DATO, "nan": SIN_DATO}))
    if "clasificacion_gravedad" not in df:
        h = pd.to_numeric(df.get("heridos", 0), errors="coerce").fillna(0) if "heridos" in df else 0
        f = pd.to_numeric(df.get("fallecidos", 0), errors="coerce").fillna(0) if "fallecidos" in df else 0
        df["clasificacion_gravedad"] = np.where(f > 0, "FATAL", np.where(h > 0, "CON HERIDOS", "SOLO DAÑOS"))
    df["clasificacion_gravedad"] = df["clasificacion_gravedad"].fillna("").astype(str).str.upper().str.strip()
    # heridos / fallecidos: se usan si existen; si no, se derivan de la clasificación de gravedad
    for c, clase in [("heridos", "CON HERIDOS"), ("fallecidos", "FATAL")]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        else:
            df[c] = (df["clasificacion_gravedad"] == clase).astype(int)
    for c in ["configuracion_de_la_via", "semaforo", "via_publica", "observaciones_generales",
              "calle", "fuente_coordenada", "nombre_via", "zona_calle"]:
        if c not in df:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str).str.strip()
    # nombre de calle para titular las zonas
    sin_calle = df["calle"] == ""
    df.loc[sin_calle, "calle"] = df.loc[sin_calle, "zona_calle"]
    sin_calle = df["calle"] == ""
    df.loc[sin_calle, "calle"] = df.loc[sin_calle, "nombre_via"]
    return df


def con_coordenadas(df):
    m = df["latitud"].notna() & df["longitud"].notna()
    m &= df["latitud"].between(-28, -27) & df["longitud"].between(-59, -58)
    return df[m].reset_index(drop=True)


# ---------------------------------------------------------------- zonas
def detectar_zonas(df, radio_m=200, min_acc=3):
    """Agrupa siniestros cercanos (DBSCAN, distancia haversine). Devuelve el df con columna 'zona' (-1 = ruido)."""
    df = df.copy()
    if len(df) < min_acc:
        df["zona"] = -1
        return df
    coords = np.radians(df[["latitud", "longitud"]].values)
    df["zona"] = DBSCAN(eps=radio_m / 6371000.0, min_samples=min_acc,
                        metric="haversine", algorithm="ball_tree").fit_predict(coords)
    return df


def resumen_zonas(df):
    z = df[df["zona"] >= 0]
    if z.empty:
        return pd.DataFrame()
    def top(s):
        s = s[s != SIN_DATO]
        return s.value_counts().index[0] if len(s) else SIN_DATO
    def nombre(s):
        s = s[s != ""]
        return s.value_counts().index[0].title() if len(s) else "Zona sin nombre"
    r = (z.groupby("zona")
          .agg(lat=("latitud", "mean"), lon=("longitud", "mean"), siniestros=("zona", "size"),
               tipo=("tipo_colision", top), calle=("calle", nombre),
               fatales=("fallecidos", lambda s: int((s > 0).sum())),
               con_heridos=("heridos", lambda s: int((s > 0).sum())))
          .reset_index())
    return r.sort_values("siniestros", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- recomendaciones
def _uniq(items):
    vistos, out = set(), []
    for med, motivo in items:
        if med not in vistos:
            vistos.add(med)
            out.append((med, motivo))
    return out


def analizar(sub):
    """Diagnóstico de una zona: tipo predominante, gravedad, perfil de la vía y recomendaciones."""
    n = len(sub)
    conteo = sub["tipo_colision"].value_counts()
    conocidos = conteo.drop(SIN_DATO, errors="ignore")
    tipo = conocidos.index[0] if len(conocidos) else SIN_DATO
    pct_tipo = (conocidos.iloc[0] / conocidos.sum()) if len(conocidos) else 0.0
    pct_sin_dato = conteo.get(SIN_DATO, 0) / n

    fatal = int((sub["fallecidos"] > 0).sum())
    heridos = int(((sub["heridos"] > 0) & (sub["fallecidos"] == 0)).sum())
    pct_victimas = (fatal + heridos) / n

    config_ok = sub["configuracion_de_la_via"].str.strip().ne("").any()
    cruce = (sub["configuracion_de_la_via"].str.contains("Intersecci", case=False)).mean() if config_ok else np.nan
    sem_conocido = sub["semaforo"][sub["semaforo"].str.strip().ne("")]
    sin_sem = sem_conocido.str.contains("Sin", case=False).mean() if len(sem_conocido) else np.nan
    via = sub["via_publica"][sub["via_publica"] != ""]
    via_pred = via.value_counts().index[0] if len(via) else ""
    es_ruta = "ruta" in via_pred.lower()
    es_avenida = "avenida" in via_pred.lower()

    horas = sub["hora"].dropna()
    noche = ((horas >= 19) | (horas < 6)).mean() if len(horas) else np.nan
    obs = sub["observaciones_generales"]
    exceso_vel = int(obs.str.contains("velocidad", case=False).sum())
    no_prio = int(obs.str.contains("prioridad", case=False).sum())
    aprox = sub["fuente_coordenada"].str.startswith("geocodificada").mean()

    hallazgos = [f"{fatal} fatal(es) y {heridos} con heridos de {n} siniestros ({pct_victimas:.0%} con víctimas)"]
    if len(conocidos):
        hallazgos.append(f"Tipo predominante: {tipo} ({pct_tipo:.0%} de los casos con tipo cargado)")
    if not np.isnan(cruce):
        hallazgos.append(f"{cruce:.0%} ocurren en intersecciones")
    if not np.isnan(sin_sem):
        hallazgos.append(f"{sin_sem:.0%} sin semáforo (sobre los que tienen el dato)")
    if not np.isnan(noche):
        hallazgos.append(f"{noche:.0%} de noche (19–6 h)")
    if via_pred:
        hallazgos.append(f"Tipo de vía predominante: {via_pred}")
    if exceso_vel:
        hallazgos.append(f"{exceso_vel} registro(s) mencionan exceso de velocidad")
    if no_prio:
        hallazgos.append(f"{no_prio} registro(s) mencionan no respetar la prioridad")

    es_cruce = (not np.isnan(cruce)) and cruce >= 0.5
    sem_no = (not np.isnan(sin_sem)) and sin_sem >= 0.5
    med = []

    if tipo in ("Lateral", "Angulo"):
        if np.isnan(cruce) and np.isnan(sin_sem):
            # El archivo no informa si es cruce ni si hay semáforo: recomendación general para choques laterales
            med += [("Reductores de velocidad (lomadas) en las aproximaciones al cruce, a estudiar",
                     "Los choques laterales suelen ocurrir en cruces; bajar la velocidad de llegada reduce la gravedad"),
                    ("Señalización de prioridad reforzada (PARE / CEDA EL PASO) y línea de detención",
                     "Ordena quién tiene paso en el cruce"),
                    ("Mejorar visibilidad en esquinas (restringir estacionamiento y despejar ochavas)",
                     "Los choques laterales suelen asociarse a poca visibilidad en la esquina"),
                    ("Control de velocidad (radares o fotomultas)", "Reduce la gravedad del impacto")]
            if es_avenida:
                med.append(("Evaluar semaforización o ajuste de fases si el cruce ya tiene semáforo",
                            "En avenidas, los cruces con mucho flujo suelen requerir regulación semafórica"))
        elif es_cruce and sem_no:
            med += [("Reductores de velocidad (lomadas) en las aproximaciones al cruce",
                     "Choques laterales en cruces sin semáforo: bajar la velocidad de llegada reduce la gravedad"),
                    ("Señalización de prioridad reforzada (PARE / CEDA EL PASO) y línea de detención",
                     "El cruce no tiene semáforo que ordene la prioridad"),
                    ("Mejorar visibilidad en esquinas (restringir estacionamiento y despejar ochavas)",
                     "Los choques laterales suelen asociarse a poca visibilidad en la esquina")]
            if n >= 4 and pct_victimas >= 0.5:
                med.append(("Evaluar mini-rotonda o semaforización del cruce",
                            "Varios siniestros con víctimas en el mismo cruce sin semáforo"))
        elif es_cruce:
            med += [("Revisar fases y ciclos del semáforo; fase protegida para giros",
                     "Los choques laterales en cruces con semáforo suelen vincularse a conflictos de fase o giro"),
                    ("Control automático de luz roja", "Reduce cruces con semáforo en rojo")]
        else:
            med += [("Señalización y demarcación de las bocacalles laterales",
                     "Choque lateral fuera de intersección formal: conflicto con vías de acceso"),
                    ("Control de velocidad (radares o fotomultas)", "Reduce la gravedad del impacto")]
        if es_avenida and not np.isnan(sin_sem) and not sem_no:
            med.append(("Control de velocidad y ajuste de la señalización semafórica",
                        "En avenidas con semáforo suelen funcionar mejor que los reductores"))
    elif tipo == "Frontal":
        if es_ruta:
            med += [("Banquinas pavimentadas y defensas laterales", "Choque frontal en ruta: reducir invasión de carril y salidas de calzada"),
                    ("Doble línea central continua con tachas reflectivas", "Impide el sobrepaso en zonas de riesgo")]
        else:
            med += [("Separador central físico", "Evita la invasión del carril contrario"),
                    ("Doble línea continua con tachas reflectivas", "Refuerza la separación de sentidos")]
    elif tipo == "Alcance":
        med += [("Sincronización semafórica (onda verde) y ajuste de ciclos", "Los alcances se asocian a frenadas bruscas y congestión en semáforos"),
                ("Señalización anticipada de semáforo o cruce", "Da más tiempo de reacción a quien llega"),
                ("Control de velocidad (radares o fotomultas)", "Reduce la distancia de frenado necesaria")]
    elif tipo == "Rozamiento":
        med += [("Demarcación de carriles y ancho de carril adecuado", "El roce lateral suele deberse a carriles angostos o mal marcados"),
                ("Ordenar el estacionamiento", "Libera ancho útil de calzada")]
    elif tipo == "Multiple":
        med += [("Auditoría integral de seguridad vial en el lugar", "Siniestros con varios vehículos: causas combinadas"),
                ("Señalización anticipada y control de velocidad", "Reduce la probabilidad de choques en cadena")]
    else:
        med += [("Auditoría de seguridad vial en el lugar", "No hay tipo de colisión predominante cargado"),
                ("Completar la carga del tipo de colisión en el FEU", "Permite recomendar con más precisión")]

    if fatal > 0:
        med.append(("Auditoría de seguridad vial en el lugar", "Hay al menos una víctima fatal en la zona"))
    if pct_victimas >= 0.5 and n >= 4:
        med.append(("Zona de velocidad reducida (30 km/h) con control", "Más de la mitad de los siniestros dejaron víctimas"))
    if exceso_vel:
        med.append(("Control de velocidad (radares o fotomultas)", "Las observaciones mencionan exceso de velocidad"))
    if not np.isnan(noche) and noche >= 0.4:
        med.append(("Mejorar iluminación y demarcación retrorreflectiva", "Gran parte de los siniestros ocurre de noche"))

    if fatal > 0 or (pct_victimas >= 0.6 and n >= 4):
        prioridad = "ALTA"
    elif pct_victimas >= 0.3:
        prioridad = "MEDIA"
    else:
        prioridad = "BAJA"

    avisos = []
    if n < 5:
        avisos.append(f"Pocos casos ({n}): la tendencia es débil.")
    if pct_sin_dato >= 0.3:
        avisos.append(f"{pct_sin_dato:.0%} de los siniestros no tiene tipo de colisión cargado.")
    if aprox >= 0.5:
        avisos.append(f"{aprox:.0%} de los puntos tiene ubicación aproximada (geocodificada por dirección).")
    if tipo in ("Lateral", "Angulo") and np.isnan(cruce) and np.isnan(sin_sem):
        avisos.append("El archivo no informa configuración de la vía ni semáforo: las medidas para choques laterales son generales.")
    avisos.append("Recomendación orientativa: un técnico municipal debe evaluarla en el lugar.")

    return dict(n=n, tipo=tipo, pct_tipo=pct_tipo, conteo=conteo, prioridad=prioridad,
                hallazgos=hallazgos, medidas=_uniq(med), avisos=avisos)
