from datetime import date, datetime, time, timedelta
import csv
import io
import os
import tempfile

import folium
from folium.plugins import MousePosition
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import rasterio
from rasterio.warp import transform
import requests
import streamlit as st
from streamlit_folium import st_folium


# =========================================================
# 0. CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="LUNASCOUT | Lunar Site Comparison",
    layout="wide"
)
st.title("🌕 LUNASCOUT | Lunar Landing Site Intelligence")
st.caption(
    "Comparación de sitios y análisis temporal de la posición aparente "
    "del Sol y la Tierra mediante NASA/JPL Horizons."
)


# =========================================================
# 1. FUENTES NASA MOON TREK: WAYPOINTS APOLLO Y CAPAS
# =========================================================
URLS_APOLLO = {
    "Apollo 11": "https://trek.nasa.gov/moon/trekarcgis/rest/services/Apollo11_waypoints/MapServer/0/query?f=json&where=1%3D1&returnGeometry=true&outFields=*&outSR=104903",
    "Apollo 12": "https://trek.nasa.gov/moon/trekarcgis/rest/services/Apollo12_waypoints/MapServer/0/query?f=json&where=1%3D1&returnGeometry=true&outFields=*&outSR=104903",
    "Apollo 14": "https://trek.nasa.gov/moon/trekarcgis/rest/services/Apollo14_waypoints/MapServer/0/query?f=json&where=1%3D1&returnGeometry=true&outFields=*&outSR=104903",
    "Apollo 15": "https://trek.nasa.gov/moon/trekarcgis/rest/services/Apollo15_waypoints/MapServer/0/query?f=json&where=1%3D1&returnGeometry=true&outFields=*&outSR=104903",
    "Apollo 16": "https://trek.nasa.gov/moon/trekarcgis/rest/services/Apollo16_waypoints/MapServer/0/query?f=json&where=1%3D1&returnGeometry=true&outFields=*&outSR=104903",
    "Apollo 17": "https://trek.nasa.gov/moon/trekarcgis2/rest/services/Apollo17_waypoints/MapServer/0/query?f=json&where=1%3D1&returnGeometry=true&outFields=*&outSR=104903",
}

CAPAS_ALTA_RES = {
    "Apollo 11": {
        "nombre": "Apollo 11 Ultra Res (26 cm)",
        "url": "https://trek.nasa.gov/tiles/Moon/EQ/apollo11_26cm_mosaic_byte_geo_1_2_highContrast/1.0.0/default/default028mm/{z}/{y}/{x}.png",
        "max_native_zoom": 16,
    },
    "Apollo 12": {
        "nombre": "Apollo 12 Alta Res (60x60 km)",
        "url": "https://trek.nasa.gov/tiles/Moon/EQ/A12_60x60km.0_3.eq/1.0.0/default/default028mm/{z}/{y}/{x}.png",
        "max_native_zoom": 14,
    },
    "Apollo 14": {
        "nombre": "Apollo 14 Alta Res (60x60 km)",
        "url": "https://trek.nasa.gov/tiles/Moon/EQ/A14_60x60km.0_3.eq/1.0.0/default/default028mm/{z}/{y}/{x}.png",
        "max_native_zoom": 14,
    },
    "Apollo 15": {
        "nombre": "Apollo 15 Alta Res (60x60 km)",
        "url": "https://trek.nasa.gov/tiles/Moon/EQ/A15_60x60km.0_3.eq/1.0.0/default/default028mm/{z}/{y}/{x}.png",
        "max_native_zoom": 14,
    },
    "Apollo 16": {
        "nombre": "Apollo 16 Alta Res (60x60 km)",
        "url": "https://trek.nasa.gov/tiles/Moon/EQ/A16_60x60km.0_3.eq/1.0.0/default/default028mm/{z}/{y}/{x}.png",
        "max_native_zoom": 14,
    },
    "Apollo 17": {
        "nombre": "Apollo 17 Alta Res (60x60 km)",
        "url": "https://trek.nasa.gov/tiles/Moon/EQ/A17_60x60km.0_3.eq/1.0.0/default/default028mm/{z}/{y}/{x}.png",
        "max_native_zoom": 14,
    },
}


@st.cache_data(ttl=86400, show_spinner=False)
def cargar_banco_misiones():
    banco = {}
    for mision, url in URLS_APOLLO.items():
        try:
            respuesta = requests.get(url, timeout=20)
            respuesta.raise_for_status()
            datos = respuesta.json()
            filas = [
                {**f.get("attributes", {}), **f.get("geometry", {})}
                for f in datos.get("features", [])
            ]
            banco[mision] = pd.DataFrame(filas)
        except Exception as exc:
            banco[mision] = pd.DataFrame()
            st.warning(f"No se pudieron cargar los waypoints de {mision}: {exc}")
    return banco


banco_misiones = cargar_banco_misiones()


# =========================================================
# 2. CONFIGURACIÓN DE LA COMPARACIÓN
# =========================================================
st.sidebar.header("01 · Configuración de misión")

misiones_seleccionadas = st.sidebar.multiselect(
    "Sitios Apollo para comparar:",
    options=list(URLS_APOLLO.keys()),
    default=["Apollo 11"],
    help=(
        "La coordenada de referencia de cada misión Apollo se calcula como "
        "el promedio de sus waypoints disponibles. No debe interpretarse "
        "como la coordenada oficial del punto de touchdown."
    ),
)

incluir_custom = st.sidebar.checkbox("Agregar coordenada personalizada")
if incluir_custom:
    nombre_custom = st.sidebar.text_input(
        "Nombre del sitio personalizado", value="Custom Site"
    )
    lat_custom = st.sidebar.number_input(
        "Latitud personalizada (°)",
        min_value=-90.0,
        max_value=90.0,
        value=-89.0,
        step=0.01,
        format="%.5f",
    )
    lon_custom = st.sidebar.number_input(
        "Longitud personalizada (°)",
        min_value=-180.0,
        max_value=180.0,
        value=0.0,
        step=0.01,
        format="%.5f",
    )

st.sidebar.divider()
st.sidebar.header("02 · Ventana temporal (UT)")
fecha_hoy = date.today()
f_ini = st.sidebar.date_input("Fecha de inicio:", value=fecha_hoy)
hora_ini = st.sidebar.time_input("Hora de inicio UT:", value=time(0, 0))
f_fin = st.sidebar.date_input(
    "Fecha de fin:", value=fecha_hoy + timedelta(days=1)
)
hora_fin = st.sidebar.time_input("Hora de fin UT:", value=time(0, 0))
inicio_ut = datetime.combine(f_ini, hora_ini)
fin_ut = datetime.combine(f_fin, hora_fin)

intervalos_muestreo = {
    "5 minutos": ("5 min", 5),
    "15 minutos": ("15 min", 15),
    "30 minutos": ("30 min", 30),
    "1 hora": ("60 min", 60),
    "2 horas": ("120 min", 120),
    "6 horas": ("360 min", 360),
}
seleccion_intervalo = st.sidebar.selectbox(
    "Intervalo de muestreo:",
    options=list(intervalos_muestreo.keys()),
    index=3,
    help="Un intervalo menor produce más muestras y requiere más consultas/procesamiento.",
)
step_size, intervalo_minutos = intervalos_muestreo[seleccion_intervalo]

st.sidebar.divider()
st.sidebar.header("03 · Mapa")
zoom_mapa = st.sidebar.slider("Zoom del mapa:", 2, 18, 7)

st.sidebar.divider()
st.sidebar.header("04 · Topografía LOLA DEM (opcional)")
archivo_dem_subido = st.sidebar.file_uploader(
    "Subir archivo DEM (.tif)", type=["tif", "tiff"]
)
ruta_tif_local = st.sidebar.text_input(
    "O ruta del DEM en el repositorio:",
    value="LRO_LOLA_DEM_Global_128ppd_v04.tif",
)


# Convertir las misiones seleccionadas a coordenadas de referencia.
sitios_a_evaluar = []
for mision in misiones_seleccionadas:
    df_mision = banco_misiones.get(mision, pd.DataFrame())
    if df_mision.empty or not {"x", "y"}.issubset(df_mision.columns):
        st.sidebar.warning(f"{mision}: no hay coordenadas x/y disponibles.")
        continue

    coordenadas = df_mision[["x", "y"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    if coordenadas.empty:
        continue

    sitios_a_evaluar.append({
        "nombre": f"{mision} · promedio de waypoints",
        "mision": mision,
        "lat": float(coordenadas["y"].mean()),
        "lon": float(coordenadas["x"].mean()),
    })

if incluir_custom:
    sitios_a_evaluar.append({
        "nombre": nombre_custom.strip() or "Custom Site",
        "mision": None,
        "lat": float(lat_custom),
        "lon": float(lon_custom),
    })

# Asegurar que los nombres usados como claves de resultados sean únicos.
nombres_usados = set()
for sitio in sitios_a_evaluar:
    nombre_base = sitio["nombre"]
    nombre_final = nombre_base
    sufijo = 2
    while nombre_final in nombres_usados:
        nombre_final = f"{nombre_base} ({sufijo})"
        sufijo += 1
    sitio["nombre"] = nombre_final
    nombres_usados.add(nombre_final)

if len(sitios_a_evaluar) > 5:
    st.error("Selecciona como máximo cinco sitios por análisis.")
    st.stop()
if not sitios_a_evaluar:
    st.info("Selecciona al menos una misión Apollo o agrega coordenadas personalizadas.")
    st.stop()
if fin_ut <= inicio_ut:
    st.error("La fecha y hora final deben ser posteriores a la fecha y hora inicial.")
    st.stop()

horas_ventana = (fin_ut - inicio_ut).total_seconds() / 3600.0
muestras_estimadas = int(horas_ventana * 60 / intervalo_minutos) + 1
if muestras_estimadas > 1500:
    st.warning(
        f"La configuración generará aproximadamente {muestras_estimadas:,} muestras "
        "por objeto y sitio. Considera usar un intervalo mayor para reducir el "
        "tiempo de consulta."
    )


# =========================================================
# 3. NASA/JPL HORIZONS: PARSEO CSV Y EFEMÉRIDES
# =========================================================
def parsear_tabla_horizons_csv(texto_resultado):
    """Extrae fecha/hora UT, azimut y elevación de una tabla Horizons CSV."""
    lineas = texto_resultado.splitlines()

    try:
        idx_soe = next(i for i, linea in enumerate(lineas) if "$$SOE" in linea)
        idx_eoe = next(i for i, linea in enumerate(lineas) if "$$EOE" in linea)
    except StopIteration as exc:
        raise ValueError(
            "La respuesta de Horizons no contiene los marcadores $$SOE/$$EOE. "
            "Comprueba los parámetros de consulta."
        ) from exc

    # Buscar el encabezado CSV que contiene la fecha y las columnas AZ/EL.
    encabezado = None
    for linea in reversed(lineas[:idx_soe]):
        texto = linea.strip()
        texto_lower = texto.lower()
        if "date" in texto_lower and ("azi" in texto_lower or "azimuth" in texto_lower):
            encabezado = texto
            break

    if encabezado is None:
        raise ValueError(
            "No se localizó el encabezado CSV de fecha, azimut y elevación "
            "en la respuesta de Horizons."
        )

    nombres = next(csv.reader([encabezado], skipinitialspace=True))
    nombres = [n.strip().strip('"') for n in nombres]

    filas = []
    bloque_datos = lineas[idx_soe + 1:idx_eoe]
    for linea in bloque_datos:
        if not linea.strip():
            continue
        fila = next(csv.reader([linea], skipinitialspace=True))
        # Igualar la longitud de cada fila con la del encabezado.
        if len(fila) < len(nombres):
            fila += [""] * (len(nombres) - len(fila))
        elif len(fila) > len(nombres):
            fila = fila[:len(nombres)]
        filas.append(fila)

    if not filas:
        raise ValueError("Horizons no devolvió muestras dentro del intervalo solicitado.")

    def buscar_columna(palabras):
        for i, nombre in enumerate(nombres):
            nombre_lower = nombre.lower()
            if any(p in nombre_lower for p in palabras):
                return i
        raise ValueError(f"No se encontró una columna con alguno de estos nombres: {palabras}")

    idx_fecha = buscar_columna(["date"])
    idx_azimut = buscar_columna(["azimuth", "azi"])
    idx_elevacion = buscar_columna(["elevation", "elev"])

    df = pd.DataFrame(filas, columns=nombres)
    fecha_raw = (
        df.iloc[:, idx_fecha]
        .astype(str)
        .str.replace(r"^A\.D\.\s*", "", regex=True)
        .str.strip()
    )

    resultado = pd.DataFrame({
        "datetime": pd.to_datetime(fecha_raw, errors="coerce"),
        "azimuth_deg": pd.to_numeric(df.iloc[:, idx_azimut], errors="coerce"),
        "elevation_deg": pd.to_numeric(df.iloc[:, idx_elevacion], errors="coerce"),
    })
    resultado = resultado.dropna(subset=["datetime"]).sort_values("datetime")
    resultado = resultado.drop_duplicates(subset=["datetime"]).reset_index(drop=True)

    if resultado.empty:
        raise ValueError("No fue posible convertir las fechas de la tabla de Horizons.")

    return resultado


@st.cache_data(ttl=3600, show_spinner=False)
def consultar_objeto_horizons(comando, latitud, longitud, fecha_inicio, fecha_fin, paso):
    """Consulta un objeto (Sol o Tierra) desde una coordenada sobre la Luna."""
    base_url = "https://ssd.jpl.nasa.gov/api/horizons.api"

    parametros = {
        "format": "json",
        "COMMAND": str(comando),
        "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "OBSERVER",
        "CENTER": "coord@301",
        "COORD_TYPE": "GEODETIC",
        # Horizons espera longitud, latitud y altura, en ese orden.
        "SITE_COORD": f"{longitud},{latitud},0",
        "START_TIME": str(fecha_inicio),
        "STOP_TIME": str(fecha_fin),
        "STEP_SIZE": str(paso),
        "TIME_TYPE": "UT",
        "TIME_DIGITS": "SECONDS",
        "CAL_FORMAT": "CAL",
        "QUANTITIES": "4",
        "ANG_FORMAT": "DEG",
        "CSV_FORMAT": "YES",
        "EXTRA_PREC": "YES",
    }

    respuesta = requests.get(base_url, params=parametros, timeout=40)
    respuesta.raise_for_status()
    payload = respuesta.json()

    if payload.get("error"):
        raise RuntimeError(f"JPL Horizons: {payload['error']}")
    if not payload.get("result"):
        raise RuntimeError("JPL Horizons devolvió una respuesta sin tabla de resultados.")

    return parsear_tabla_horizons_csv(payload["result"])


@st.cache_data(ttl=3600, show_spinner=False)
def evaluar_sitio_lunar(nombre, latitud, longitud, fecha_inicio, fecha_fin, paso):
    """Consulta Sol y Tierra y devuelve una serie temporal unificada por sitio."""
    df_sol = consultar_objeto_horizons(
        10, latitud, longitud, fecha_inicio, fecha_fin, paso
    ).rename(columns={
        "azimuth_deg": "sun_azimuth_deg",
        "elevation_deg": "sun_elevation_deg",
    })

    df_tierra = consultar_objeto_horizons(
        399, latitud, longitud, fecha_inicio, fecha_fin, paso
    ).rename(columns={
        "azimuth_deg": "earth_azimuth_deg",
        "elevation_deg": "earth_elevation_deg",
    })

    df = pd.merge(df_sol, df_tierra, on="datetime", how="outer")
    df = df.sort_values("datetime").reset_index(drop=True)
    df["sun_visible_geometric"] = df["sun_elevation_deg"] > 0
    df["earth_visible_geometric"] = df["earth_elevation_deg"] > 0
    df["earth_above_10deg"] = df["earth_elevation_deg"] >= 10

    return {
        "sitio": nombre,
        "lat": float(latitud),
        "lon": float(longitud),
        "datos": df,
    }


# =========================================================
# 4. FUNCIONES DE MÉTRICAS Y DEM
# =========================================================
def agregar_duracion_muestras(df, fecha_fin):
    """Asigna a cada muestra el intervalo hasta la siguiente; la última llega al fin."""
    salida = df.copy().sort_values("datetime").reset_index(drop=True)
    limite = pd.Timestamp(str(fecha_fin))
    siguiente = salida["datetime"].shift(-1)
    siguiente.iloc[-1] = limite
    salida["duration_h"] = (
        (siguiente - salida["datetime"]).dt.total_seconds() / 3600.0
    ).clip(lower=0)
    return salida


def calcular_metricas_sitio(resultado, fecha_fin, horas_totales):
    df = agregar_duracion_muestras(resultado["datos"], fecha_fin)

    def duracion(mask):
        return float(df.loc[mask.fillna(False), "duration_h"].sum())

    h_sol = duracion(df["sun_elevation_deg"] > 0)
    h_tierra = duracion(df["earth_elevation_deg"] > 0)
    h_tierra_10 = duracion(df["earth_elevation_deg"] >= 10)
    h_sol_5_20 = duracion(df["sun_elevation_deg"].between(5, 20, inclusive="both"))
    h_sol_mayor_70 = duracion(df["sun_elevation_deg"] > 70)

    def porcentaje(h):
        return 100.0 * h / horas_totales if horas_totales > 0 else np.nan

    resumen = {
        "Sitio": resultado["sitio"],
        "Latitud (°)": resultado["lat"],
        "Longitud (°)": resultado["lon"],
        "Sol sobre horizonte (h)": h_sol,
        "Sol sobre horizonte (%)": porcentaje(h_sol),
        "Tierra geométricamente visible (h)": h_tierra,
        "Tierra ≥ 10° (h)": h_tierra_10,
        "Sol entre 5° y 20° (h)": h_sol_5_20,
        "Sol > 70° (h)": h_sol_mayor_70,
    }
    return resumen, df


def analizar_dem_local(ruta_tif, lat_val, lon_val):
    """Función DEM original; analiza el archivo indicado para un punto."""
    if not ruta_tif or not os.path.exists(ruta_tif):
        return None
    try:
        with rasterio.open(ruta_tif) as dataset:
            matriz = dataset.read(1).astype(float)
            if dataset.nodata is not None:
                matriz[matriz == dataset.nodata] = np.nan

            res_x, res_y = dataset.res
            # Se conserva la lógica original del notebook. Ver nota técnica
            # al final: las unidades/proyección del DEM deben validarse.
            factor_m = 30323.0 if res_x < 0.1 else 1.0
            dy, dx = np.gradient(
                matriz,
                abs(res_y) * factor_m,
                abs(res_x) * factor_m,
            )
            pendiente_mapa = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

            alt_punto, pend_punto = np.nan, np.nan
            try:
                if dataset.crs is not None and not dataset.crs.is_geographic:
                    crs_lunar_geo = {"proj": "longlat", "a": 1737400, "b": 1737400}
                    xs, ys = transform(
                        crs_lunar_geo, dataset.crs, [lon_val], [lat_val]
                    )
                    fila, col = dataset.index(xs[0], ys[0])
                else:
                    fila, col = dataset.index(lon_val, lat_val)

                if 0 <= fila < matriz.shape[0] and 0 <= col < matriz.shape[1]:
                    alt_punto = matriz[fila, col]
                    pend_punto = pendiente_mapa[fila, col]
            except Exception:
                pass

            return {
                "matriz_alt": matriz,
                "matriz_pend": pendiente_mapa,
                "alt_punto": alt_punto,
                "pend_punto": pend_punto,
                "alt_media": float(np.nanmean(matriz)),
                "pend_media": float(np.nanmean(pendiente_mapa)),
                "porcentaje_seguro": float(np.nanmean(pendiente_mapa < 5.0) * 100),
            }
    except Exception as exc:
        st.warning(f"No se pudo analizar el DEM: {exc}")
        return None


# =========================================================
# 5. EJECUCIÓN COMPARATIVA: TODOS LOS SITIOS CON LA MISMA CONFIGURACIÓN
# =========================================================
resultados = []
errores = []

with st.spinner(
    f"Consultando JPL Horizons para {len(sitios_a_evaluar)} sitio(s), "
    f"con paso de {step_size}..."
):
    for sitio in sitios_a_evaluar:
        try:
            resultado = evaluar_sitio_lunar(
                sitio["nombre"],
                sitio["lat"],
                sitio["lon"],
                inicio_ut.strftime("%Y-%m-%d %H:%M"),
                fin_ut.strftime("%Y-%m-%d %H:%M"),
                step_size,
            )
            resultado["mision"] = sitio["mision"]
            resultados.append(resultado)
        except Exception as exc:
            errores.append((sitio["nombre"], str(exc)))

for nombre, error in errores:
    st.error(f"{nombre}: {error}")

if not resultados:
    st.error("No se pudo obtener información de ningún sitio. Revisa las fechas y la respuesta de Horizons.")
    st.stop()


# =========================================================
# 6. COMPARACIÓN SIMULTÁNEA DE SITIOS
# =========================================================
st.subheader("📊 Comparación simultánea de sitios")
st.caption(
    f"Ventana: {inicio_ut:%Y-%m-%d %H:%M} a {fin_ut:%Y-%m-%d %H:%M} UT · "
    f"Muestreo: {step_size} · Las duraciones se calculan entre timestamps consecutivos."
)

resumenes = []
datos_por_sitio = {}
for resultado in resultados:
    resumen, df_con_duracion = calcular_metricas_sitio(
        resultado, fin_ut, horas_ventana
    )
    resumenes.append(resumen)
    datos_por_sitio[resultado["sitio"]] = df_con_duracion

df_comparacion = pd.DataFrame(resumenes)
st.dataframe(
    df_comparacion,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Latitud (°)": st.column_config.NumberColumn(format="%.5f"),
        "Longitud (°)": st.column_config.NumberColumn(format="%.5f"),
        "Sol sobre horizonte (h)": st.column_config.NumberColumn(format="%.2f h"),
        "Sol sobre horizonte (%)": st.column_config.NumberColumn(format="%.1f%%"),
        "Tierra geométricamente visible (h)": st.column_config.NumberColumn(format="%.2f h"),
        "Tierra ≥ 10° (h)": st.column_config.NumberColumn(format="%.2f h"),
        "Sol entre 5° y 20° (h)": st.column_config.NumberColumn(format="%.2f h"),
        "Sol > 70° (h)": st.column_config.NumberColumn(format="%.2f h"),
    },
)

st.caption(
    "Interpretación: ‘Tierra geométricamente visible’ significa elevación > 0°. "
    "‘Tierra ≥ 10°’ es un margen angular configurable de referencia, no una "
    "garantía de enlace RF. ‘Sol > 70°’ se muestra como exposición solar elevada, "
    "no como predicción térmica."
)


# =========================================================
# 7. MAPA LUNAR CON TODOS LOS SITIOS SELECCIONADOS
# =========================================================
st.subheader("🌍 Mapa lunar comparativo · NASA Moon Trek")

centro_lat = float(np.mean([r["lat"] for r in resultados]))
longitudes_rad = np.radians([r["lon"] for r in resultados])
centro_lon = float(np.degrees(np.arctan2(
    np.sin(longitudes_rad).mean(), np.cos(longitudes_rad).mean()
)))
mapa = folium.Map(
    location=[centro_lat, centro_lon],
    zoom_start=zoom_mapa,
    crs="EPSG4326",
    tiles=None,
    control_scale=True,
)

folium.TileLayer(
    tiles="https://trek.nasa.gov/tiles/Moon/EQ/LRO_WAC_Mosaic_Global_303ppd_v02/1.0.0/default/default028mm/{z}/{y}/{x}.jpg",
    attr="NASA / LRO WAC Global",
    name="Superficie Lunar Global",
    max_zoom=20,
    max_native_zoom=7,
    overlay=False,
).add_to(mapa)

# Si se comparan varios sitios, ajustar el encuadre para mostrar todos.
if len(resultados) > 1:
    mapa.fit_bounds([
        [min(r["lat"] for r in resultados), min(r["lon"] for r in resultados)],
        [max(r["lat"] for r in resultados), max(r["lon"] for r in resultados)],
    ])

colores = ["blue", "red", "green", "purple", "orange"]
for i, resultado in enumerate(resultados):
    nombre = resultado["sitio"]
    color = colores[i % len(colores)]
    df_site = datos_por_sitio[nombre]

    horas_sol = float(
        df_site.loc[df_site["sun_elevation_deg"] > 0, "duration_h"].sum()
    )
    horas_tierra = float(
        df_site.loc[df_site["earth_elevation_deg"] > 0, "duration_h"].sum()
    )

    popup = (
        f"<b>{nombre}</b><hr>"
        f"Latitud: {resultado['lat']:.5f}°<br>"
        f"Longitud: {resultado['lon']:.5f}°<br>"
        f"Sol sobre horizonte: {horas_sol:.2f} h<br>"
        f"Tierra geométricamente visible: {horas_tierra:.2f} h"
    )

    folium.Marker(
        location=[resultado["lat"], resultado["lon"]],
        popup=folium.Popup(popup, max_width=300),
        tooltip=nombre,
        icon=folium.Icon(color=color, icon="rocket", prefix="fa"),
    ).add_to(mapa)

    # Capas de alta resolución y waypoints de las misiones seleccionadas.
    mision = resultado.get("mision")
    if mision in CAPAS_ALTA_RES:
        capa = CAPAS_ALTA_RES[mision]
        folium.TileLayer(
            tiles=capa["url"],
            attr=f"NASA / LROC / {mision}",
            name=capa["nombre"],
            max_zoom=20,
            max_native_zoom=capa["max_native_zoom"],
            overlay=True,
        ).add_to(mapa)

    if mision and mision in banco_misiones:
        df_wp = banco_misiones[mision]
        if {"x", "y"}.issubset(df_wp.columns):
            for _, punto in df_wp.iterrows():
                try:
                    wp_lat = float(punto["y"])
                    wp_lon = float(punto["x"])
                except (TypeError, ValueError):
                    continue
                nombre_wp = punto.get("name", "Waypoint")
                folium.CircleMarker(
                    location=[wp_lat, wp_lon],
                    radius=3,
                    color=color,
                    fill=True,
                    fill_opacity=0.75,
                    tooltip=f"{mision}: {nombre_wp}",
                ).add_to(mapa)

folium.LayerControl(collapsed=False).add_to(mapa)
MousePosition(
    position="bottomleft",
    separator=" | Lon: ",
    prefix="Lat: ",
).add_to(mapa)

st.caption("Los marcadores grandes representan las coordenadas de referencia analizadas. Los pequeños representan waypoints Apollo.")
mapa_data = st_folium(mapa, width=None, height=560)
if mapa_data and mapa_data.get("last_clicked"):
    c_lat = mapa_data["last_clicked"]["lat"]
    c_lon = mapa_data["last_clicked"]["lng"]
    st.info(
        f"Coordenadas seleccionadas en el mapa: latitud {c_lat:.5f}°, "
        f"longitud {c_lon:.5f}°. Puedes copiarlas al campo de sitio personalizado."
    )


# =========================================================
# 8. TIMELINE INTERACTIVO DEL SOL Y LA TIERRA
# =========================================================
st.subheader("🕒 Timeline dinámico · Sol y Tierra")
st.caption(
    "Cada curva muestra la elevación aparente respecto al horizonte geométrico "
    "del punto lunar. Usa el zoom, el hover y el selector temporal para explorar la ventana."
)

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.10,
    subplot_titles=("Elevación aparente del Sol", "Elevación aparente de la Tierra"),
)

for resultado in resultados:
    nombre = resultado["sitio"]
    df = datos_por_sitio[nombre]

    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["sun_elevation_deg"],
            mode="lines",
            name=f"{nombre} · Sol",
            connectgaps=False,
            hovertemplate="%{x|%Y-%m-%d %H:%M} UT<br>Elevación: %{y:.2f}°<extra>%{fullData.name}</extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["earth_elevation_deg"],
            mode="lines",
            name=f"{nombre} · Tierra",
            connectgaps=False,
            hovertemplate="%{x|%Y-%m-%d %H:%M} UT<br>Elevación: %{y:.2f}°<extra>%{fullData.name}</extra>",
        ),
        row=2,
        col=1,
    )

fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
fig.add_hrect(
    y0=5,
    y1=20,
    fillcolor="green",
    opacity=0.10,
    line_width=0,
    row=1,
    col=1,
    annotation_text="Banda de referencia 5°–20°",
    annotation_position="top left",
)
fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
fig.add_hline(y=10, line_dash="dot", line_color="orange", row=2, col=1)

fig.update_yaxes(title_text="Elevación (°)", row=1, col=1)
fig.update_yaxes(title_text="Elevación (°)", row=2, col=1)
fig.update_xaxes(title_text="Fecha y hora UT", row=2, col=1, rangeslider_visible=True)
fig.update_layout(
    height=700,
    hovermode="x unified",
    legend_title_text="Sitio / objeto",
    margin=dict(l=30, r=30, t=70, b=35),
)
st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 9. INSPECCIÓN DE UN INSTANTE CONCRETO
# =========================================================
st.subheader("🔎 Inspección de un instante")
sitios_disponibles = list(datos_por_sitio.keys())
sitio_inspeccion = st.selectbox(
    "Sitio para inspeccionar:",
    options=sitios_disponibles,
    key="sitio_inspeccion_timeline",
)

df_inspeccion = datos_por_sitio[sitio_inspeccion]
if len(df_inspeccion) >= 2:
    t_min = df_inspeccion["datetime"].min().to_pydatetime()
    t_max = df_inspeccion["datetime"].max().to_pydatetime()
    instante = st.slider(
        "Desplaza el tiempo de misión (UT):",
        min_value=t_min,
        max_value=t_max,
        value=t_min,
        step=timedelta(minutes=intervalo_minutos),
        format="YYYY-MM-DD HH:mm",
    )
    idx_cercano = (df_inspeccion["datetime"] - pd.Timestamp(instante)).abs().idxmin()
    muestra = df_inspeccion.loc[idx_cercano]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Azimut del Sol",
        "n/d" if pd.isna(muestra["sun_azimuth_deg"]) else f"{muestra['sun_azimuth_deg']:.2f}°",
    )
    c2.metric(
        "Elevación del Sol",
        "n/d" if pd.isna(muestra["sun_elevation_deg"]) else f"{muestra['sun_elevation_deg']:.2f}°",
    )
    c3.metric(
        "Azimut de la Tierra",
        "n/d" if pd.isna(muestra["earth_azimuth_deg"]) else f"{muestra['earth_azimuth_deg']:.2f}°",
    )
    c4.metric(
        "Elevación de la Tierra",
        "n/d" if pd.isna(muestra["earth_elevation_deg"]) else f"{muestra['earth_elevation_deg']:.2f}°",
    )
else:
    st.info("No hay suficientes muestras para construir un selector temporal.")


# =========================================================
# 10. TOPOGRAFÍA LOLA DEM (OPCIONAL, CONSERVA LA FUNCIÓN ORIGINAL)
# =========================================================
ruta_dem_analisis = None
archivo_temporal = None

if archivo_dem_subido is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        tmp.write(archivo_dem_subido.getvalue())
        archivo_temporal = tmp.name
    ruta_dem_analisis = archivo_temporal
elif ruta_tif_local and os.path.exists(ruta_tif_local):
    ruta_dem_analisis = ruta_tif_local

if ruta_dem_analisis:
    st.subheader("🏔️ Inspección topográfica LOLA (sitio seleccionado)")
    sitio_dem = st.selectbox(
        "Sitio para consultar el DEM:",
        options=sitios_disponibles,
        key="sitio_dem",
    )
    registro_dem = next(r for r in resultados if r["sitio"] == sitio_dem)
    info_dem = analizar_dem_local(
        ruta_dem_analisis,
        registro_dem["lat"],
        registro_dem["lon"],
    )

    if info_dem:
        d1, d2, d3 = st.columns(3)
        if not np.isnan(info_dem["alt_punto"]):
            d1.metric("Altitud en el punto (DEM)", f"{info_dem['alt_punto']:.2f} m")
            d2.metric("Pendiente en el píxel", f"{info_dem['pend_punto']:.2f}°")
        else:
            d1.metric("Altitud media del raster", f"{info_dem['alt_media']:.2f} m")
            d2.metric("Pendiente media del raster", f"{info_dem['pend_media']:.2f}°")
        d3.metric("Píxeles con pendiente < 5°", f"{info_dem['porcentaje_seguro']:.1f}%")

        fig_dem, axes_dem = plt.subplots(1, 2, figsize=(12, 4))
        im_alt = axes_dem[0].imshow(info_dem["matriz_alt"], cmap="terrain")
        axes_dem[0].set_title("Elevación DEM LOLA")
        fig_dem.colorbar(im_alt, ax=axes_dem[0], label="Elevación (unidades del DEM)")

        im_pend = axes_dem[1].imshow(
            info_dem["matriz_pend"], cmap="hot_r", vmin=0, vmax=25
        )
        axes_dem[1].set_title("Mapa de pendientes")
        fig_dem.colorbar(im_pend, ax=axes_dem[1], label="Pendiente (°)")
        fig_dem.tight_layout()
        st.pyplot(fig_dem, use_container_width=True)
        plt.close(fig_dem)

if archivo_temporal and os.path.exists(archivo_temporal):
    try:
        os.remove(archivo_temporal)
    except OSError:
        pass

st.divider()
st.caption(
    "Nota: las coordenadas Apollo de referencia son promedios de waypoints. "
    "La elevación terrestre > 0° es visibilidad geométrica, no confirmación de "
    "un enlace RF. La banda solar 5°–20° y los umbrales mostrados son criterios "
    "exploratorios del prototipo, no límites certificados de NASA/HLS."
)
