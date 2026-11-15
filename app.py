from datetime import date
import os
import tempfile
import folium
from folium.plugins import MousePosition
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Simulador Integral de Aterrizaje Lunar", layout="wide"
)
st.title("🛰️ Simulador y Analizador Integral de Aterrizaje Lunar")

# ---------------------------------------------------------
# 1. BASES DE DATOS Y CAPAS CARTOGRÁFICAS (NASA MOON TREK)
# ---------------------------------------------------------
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
        "nombre": "Apollo 11 Ultra Res (26cm)",
        "url": "https://trek.nasa.gov/tiles/Moon/EQ/apollo11_26cm_mosaic_byte_geo_1_2_highContrast/1.0.0/default/default028mm/{z}/{y}/{x}.png",
        "max_native_zoom": 16,
    },
    "Apollo 12": {
        "nombre": "Apollo 12 Alta Res (60x60km)",
        "url": "https://trek.nasa.gov/tiles/Moon/EQ/A12_60x60km.0_3.eq/1.0.0/default/default028mm/{z}/{y}/{x}.png",
        "max_native_zoom": 14,
    },
    "Apollo 14": {
        "nombre": "Apollo 14 Alta Res (60x60km)",
        "url": "https://trek.nasa.gov/tiles/Moon/EQ/A14_60x60km.0_3.eq/1.0.0/default/default028mm/{z}/{y}/{x}.png",
        "max_native_zoom": 14,
    },
    "Apollo 15": {
        "nombre": "Apollo 15 Alta Res (60x60km)",
        "url": "https://trek.nasa.gov/tiles/Moon/EQ/A15_60x60km.0_3.eq/1.0.0/default/default028mm/{z}/{y}/{x}.png",
        "max_native_zoom": 14,
    },
    "Apollo 16": {
        "nombre": "Apollo 16 Alta Res (60x60km)",
        "url": "https://trek.nasa.gov/tiles/Moon/EQ/A16_60x60km.0_3.eq/1.0.0/default/default028mm/{z}/{y}/{x}.png",
        "max_native_zoom": 14,
    },
    "Apollo 17": {
        "nombre": "Apollo 17 Alta Res (60x60km)",
        "url": "https://trek.nasa.gov/tiles/Moon/EQ/A17_60x60km.0_3.eq/1.0.0/default/default028mm/{z}/{y}/{x}.png",
        "max_native_zoom": 14,
    },
}


@st.cache_data(ttl=86400)
def cargar_banco_misiones():
  banco = {}
  for mision, url in URLS_APOLLO.items():
    try:
      res = requests.get(url, timeout=10)
      if res.status_code == 200:
        datos = res.json()
        filas = [
            {**f.get("attributes", {}), **f.get("geometry", {})}
            for f in datos.get("features", [])
        ]
        banco[mision] = pd.DataFrame(filas)
    except Exception:
      banco[mision] = pd.DataFrame()
  return banco


banco_misiones = cargar_banco_misiones()

# ---------------------------------------------------------
# 2. CONTROLES INTERACTIVOS EN LA BARRA LATERAL
# ---------------------------------------------------------
st.sidebar.header("Configuración de Misión")
opciones_sitios = list(URLS_APOLLO.keys()) + [
    "Coordenadas Personalizadas (Custom)"
]
sitio_seleccionado = st.sidebar.selectbox("Sitio Lunar:", opciones_sitios, index=0)

if (
    sitio_seleccionado in banco_misiones
    and not banco_misiones[sitio_seleccionado].empty
):
  df_sel = banco_misiones[sitio_seleccionado]
  default_lat = round(float(df_sel["y"].mean()), 5)
  default_lon = round(float(df_sel["x"].mean()), 5)
else:
  default_lat = 23.25
  default_lon = 26.45

if "ultimo_sitio" not in st.session_state:
  st.session_state.ultimo_sitio = sitio_seleccionado
  st.session_state.lat_val = default_lat
  st.session_state.lon_val = default_lon
elif st.session_state.ultimo_sitio != sitio_seleccionado:
  st.session_state.ultimo_sitio = sitio_seleccionado
  st.session_state.lat_val = default_lat
  st.session_state.lon_val = default_lon

lat = st.sidebar.number_input(
    "Latitud (°):",
    min_value=-90.0,
    max_value=90.0,
    value=st.session_state.lat_val,
    step=0.05,
    format="%.5f",
)
lon = st.sidebar.number_input(
    "Longitud (°):",
    min_value=-180.0,
    max_value=180.0,
    value=st.session_state.lon_val,
    step=0.05,
    format="%.5f",
)
zoom_mapa = st.sidebar.slider("Zoom Mapa:", 2, 18, 12)

st.sidebar.divider()
st.sidebar.header("Ventana Temporal (UTC)")
f_ini = st.sidebar.date_input("Inicio:", value=date(2026, 4, 22))
f_fin = st.sidebar.date_input("Fin:", value=date(2026, 4, 23))

st.sidebar.divider()
st.sidebar.header("Topografía LOLA DEM (.tif)")
archivo_dem_subido = st.sidebar.file_uploader(
    "Subir archivo DEM (.tif) opcional", type=["tif", "tiff"]
)
ruta_tif_local = st.sidebar.text_input(
    "O ruta local en repositorio:", value="LRO_LOLA_DEM_Global_128ppd_v04.tif"
)


# ---------------------------------------------------------
# 3. FUNCIONES DE TELEMETRÍA (JPL HORIZONS) Y TOPOGRAFÍA (DEM)
# ---------------------------------------------------------
def extraer_elevacion_y_grados(texto):
  datos = []
  guardar = False
  for linea in texto.split("\n"):
    if "$$EOE" in linea:
      break
    if guardar and linea.strip():
      partes = linea.split()
      if len(partes) >= 4:
        try:
          grados = float(partes[-1])
          datos.append({"grados": grados, "visible": grados > 0})
        except ValueError:
          pass
    if "$$SOE" in linea:
      guardar = True
  return datos


@st.cache_data(ttl=3600)
def evaluar_sitio_lunar(nombre, lat_val, lon_val, fecha_i, fecha_f):
  base_url = "https://ssd.jpl.nasa.gov/api/horizons.api"
  params_comunes = (
      "?format=json&MAKE_EPHEM=YES&EPHEM_TYPE=OBSERVER&CENTER='coord@301'"
      f"&COORD_TYPE=GEODETIC&SITE_COORD='{lon_val},{lat_val},0'"
      f"&START_TIME='{fecha_i}'&STOP_TIME='{fecha_f}'&STEP_SIZE='1%20h'&QUANTITIES='4'"
  )
  try:
    res_sol = requests.get(
        base_url + params_comunes + "&COMMAND='10'", timeout=15
    ).json()
    res_tierra = requests.get(
        base_url + params_comunes + "&COMMAND='399'", timeout=15
    ).json()
    return {
        "sitio": nombre,
        "lat": lat_val,
        "lon": lon_val,
        "luz": extraer_elevacion_y_grados(res_sol.get("result", "")),
        "comms": extraer_elevacion_y_grados(res_tierra.get("result", "")),
    }
  except Exception as e:
    return {"error": str(e)}


def analizar_dem_local(ruta_tif, lat_val, lon_val):
  if not ruta_tif or not os.path.exists(ruta_tif):
    return None
  try:
    with rasterio.open(ruta_tif) as dataset:
      matriz = dataset.read(1).astype(float)
      if dataset.nodata is not None:
        matriz[matriz == dataset.nodata] = np.nan

      res_x, res_y = dataset.res
      factor_m = 30323.0 if res_x < 0.1 else 1.0
      dy, dx = np.gradient(matriz, res_y * factor_m, res_x * factor_m)
      pendiente_mapa = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

      alt_punto, pend_punto = np.nan, np.nan
      try:
        if dataset.crs is not None and not dataset.crs.is_geographic:
          crs_lunar_geo = {"proj": "longlat", "a": 1737400, "b": 1737400}
          xs, ys = transform(crs_lunar_geo, dataset.crs, [lon_val], [lat_val])
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
  except Exception as e:
    st.warning(f"Aviso al leer el archivo DEM: {e}")
    return None


# ---------------------------------------------------------
# 4. MOTOR UNIFICADO DE ANÁLISIS Y VISUALIZACIÓN
# ---------------------------------------------------------
if f_fin <= f_ini:
  st.error(
      "ERROR: La fecha de Fin debe ser posterior a la fecha de Inicio para que"
      " JPL Horizons devuelva datos."
  )
  st.stop()

with st.spinner(
    f"Consultando efemérides en JPL Horizons para {sitio_seleccionado} (Lat:"
    f" {lat}°, Lon: {lon}°)..."
):
  datos_mision = evaluar_sitio_lunar(
      sitio_seleccionado, lat, lon, str(f_ini), str(f_fin)
  )

info_dem = None
if archivo_dem_subido is not None:
  with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_file:
    tmp_file.write(archivo_dem_subido.read())
    tmp_path = tmp_file.name
  info_dem = analizar_dem_local(tmp_path, lat, lon)
  os.remove(tmp_path)
elif os.path.exists(ruta_tif_local):
  info_dem = analizar_dem_local(ruta_tif_local, lat, lon)

if not datos_mision or "error" in datos_mision or len(datos_mision["luz"]) == 0:
  st.error(
      "ERROR: La API de JPL Horizons no devolvió datos. Verifica el rango de"
      " fechas."
  )
  st.stop()

horas_totales = len(datos_mision["luz"])
horas_luz = sum(1 for x in datos_mision["luz"] if x["visible"])
horas_comms = sum(1 for x in datos_mision["comms"] if x["visible"])

horas_ideales_aterrizaje = 0
horas_peligro_termico = 0
perdidas_senal = 0

for i in range(horas_totales):
  sol_grados = datos_mision["luz"][i]["grados"]
  tierra_grados = datos_mision["comms"][i]["grados"]
  if 5 <= sol_grados <= 20:
    horas_ideales_aterrizaje += 1
  if sol_grados > 70:
    horas_peligro_termico += 1
  if tierra_grados < 10:
    perdidas_senal += 1

riesgo_topografico = False
if info_dem and not np.isnan(info_dem["pend_punto"]):
  if info_dem["pend_punto"] > 10.0:
    riesgo_topografico = True

# Dictamen Final de Viabilidad
st.subheader(
    f"📋 Reporte de Viabilidad de Ingeniería: {sitio_seleccionado.upper()}"
)

if perdidas_senal == horas_totales:
  veredicto = (
      "NO VIABLE (RECHAZADO): Zona de sombra de radio. La Tierra no tiene línea"
      " de vista segura (<10°)."
  )
  color_pin = "red"
  st.error(f"🚨 **DICTAMEN:** {veredicto}")
elif riesgo_topografico:
  veredicto = (
      f"NO VIABLE (RECHAZADO): Pendiente local excesiva"
      f" ({info_dem['pend_punto']:.2f}° > 10°). Riesgo de volcadura del módulo."
  )
  color_pin = "red"
  st.error(f"🚨 **DICTAMEN:** {veredicto}")
elif horas_luz == 0:
  veredicto = (
      "NO VIABLE (RECHAZADO): Noche lunar total durante toda la ventana"
      " seleccionada."
  )
  color_pin = "red"
  st.error(f"🚨 **DICTAMEN:** {veredicto}")
elif horas_ideales_aterrizaje == 0:
  veredicto = (
      "VIABILIDAD MARGINAL: Hay visibilidad o luz, pero sin ventana de sombras"
      " largas (5°-20°) para detección de obstáculos en descenso."
  )
  color_pin = "orange"
  st.warning(f"⚠️ **DICTAMEN:** {veredicto}")
elif horas_peligro_termico > (horas_totales / 2):
  veredicto = (
      "VIABLE CON ADVERTENCIA TÉRMICA: Más del 50% de la ventana tiene"
      " elevación solar >70°."
  )
  color_pin = "orange"
  st.warning(f"🔥 **DICTAMEN:** {veredicto}")
else:
  veredicto = (
      f"SITIO VIABLE: Condiciones nominales. Cuenta con"
      f" {horas_ideales_aterrizaje} hrs de iluminación óptima y enlace directo"
      " con Tierra."
  )
  color_pin = "green"
  st.success(f"✅ **DICTAMEN:** {veredicto}")

# Métricas Principales
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ventana Analizada", f"{horas_totales} hrs")
c2.metric(
    "Luz Solar Directa",
    f"{horas_luz} hrs ({horas_luz / horas_totales * 100:.1f}%)",
)
c3.metric(
    "Señal a la Tierra",
    f"{horas_comms} hrs ({horas_comms / horas_totales * 100:.1f}%)",
)
c4.metric("Luz Rasante (5°–20°)", f"{horas_ideales_aterrizaje} hrs")

if info_dem:
  st.markdown("#### 🏔️ Análisis Topográfico (LOLA DEM)")
  d1, d2, d3 = st.columns(3)
  if not np.isnan(info_dem["alt_punto"]):
    d1.metric("Altitud Exacta en Punto", f"{info_dem['alt_punto']:.2f} m")
    d2.metric("Pendiente Local en Punto", f"{info_dem['pend_punto']:.2f}°")
  else:
    d1.metric(
        "Altitud Media DEM (Fuera de recorte)", f"{info_dem['alt_media']:.2f} m"
    )
    d2.metric("Pendiente Media DEM", f"{info_dem['pend_media']:.2f}°")
  d3.metric(
      "Terreno Seguro en Cuadrante (<5°)",
      f"{info_dem['porcentaje_seguro']:.1f}%",
  )

# ---------------------------------------------------------
# 5. MAPA INTERACTIVO FOLIUM (PRIMERO ARRIBA)
# ---------------------------------------------------------
st.subheader("🌍 Mapa Lunar Interactivo (NASA Moon Trek)")
mapa = folium.Map(
    location=[lat, lon],
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

if sitio_seleccionado in CAPAS_ALTA_RES:
  info_capa = CAPAS_ALTA_RES[sitio_seleccionado]
  folium.TileLayer(
      tiles=info_capa["url"],
      attr=f"NASA / LROC / {sitio_seleccionado}",
      name=info_capa["nombre"],
      max_zoom=20,
      max_native_zoom=info_capa["max_native_zoom"],
      overlay=True,
  ).add_to(mapa)

estado_corto = (
    "VIABLE"
    if ("SITIO VIABLE" in veredicto or "ADVERTENCIA" in veredicto)
    else "NO VIABLE / MARGINAL"
)
popup_principal = f"""
<div style="font-family: Arial; min-width: 210px;">
    <b>{sitio_seleccionado}</b><hr style="margin: 4px 0;">
    <b>Lat:</b> {lat:.5f}° | <b>Lon:</b> {lon:.5f}°<br>
    <b>Luz Solar:</b> {horas_luz}/{horas_totales} hrs<br>
    <b>Señal Tierra:</b> {horas_comms}/{horas_totales} hrs<br>
    <b>Luz Ideal (5-20°):</b> {horas_ideales_aterrizaje} hrs<br>
    <b>Estado:</b> {estado_corto}
</div>
"""
folium.Marker(
    location=[lat, lon],
    popup=folium.Popup(popup_principal, max_width=280),
    tooltip=f"Punto Evaluado: {sitio_seleccionado}",
    icon=folium.Icon(color=color_pin, icon="rocket", prefix="fa"),
).add_to(mapa)

if sitio_seleccionado in banco_misiones:
  df_wp = banco_misiones[sitio_seleccionado]
  for _, row in df_wp.iterrows():
    nombre_wp = row.get("name", f"Punto {row.get('FID', '')}")
    link_info = row.get("template", "")
    html_wp = (
        f"<b>{nombre_wp}</b><br>Lat: {row['y']:.5f}°<br>Lon: {row['x']:.5f}°"
    )
    if link_info:
      html_wp += f"<br><a href='{link_info}' target='_blank'>Ficha NASA</a>"
    folium.Marker(
        location=[row["y"], row["x"]],
        popup=folium.Popup(html_wp, max_width=220),
        tooltip=str(nombre_wp),
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(mapa)

folium.LayerControl(collapsed=False).add_to(mapa)
MousePosition(
    position="bottomleft", separator=" | Lon: ", prefix="Lat: "
).add_to(mapa)

st.caption(
    "💡 Haz clic en cualquier punto del mapa para leer sus coordenadas exactas."
)
mapa_data = st_folium(mapa, width=1200, height=500)

if mapa_data and mapa_data.get("last_clicked"):
  c_lat = mapa_data["last_clicked"]["lat"]
  c_lon = mapa_data["last_clicked"]["lng"]
  st.info(
      f"📍 Coordenadas seleccionadas en el mapa: **Latitud = {c_lat:.5f}° ,"
      f" Longitud = {c_lon:.5f}°** (ingrésalas en el panel izquierdo con 'Sitio"
      " Lunar: Coordenadas Personalizadas' para simular ese punto)."
  )

# ---------------------------------------------------------
# 6. GRÁFICAS COMPACTAS DE TELEMETRÍA Y DEM (DEBAJO DEL MAPA)
# ---------------------------------------------------------
st.subheader("📊 Gráficas de Telemetría y Topografía")

horas_eje = range(horas_totales)
grados_sol = [x["grados"] for x in datos_mision["luz"]]
grados_tierra = [x["grados"] for x in datos_mision["comms"]]

if info_dem:
  fig, axes = plt.subplots(1, 3, figsize=(14, 2.8))
  axes[0].plot(
      horas_eje,
      grados_sol,
      label="Elevación Solar (°)",
      color="#f39c12",
      linewidth=1.8,
  )
  axes[0].plot(
      horas_eje,
      grados_tierra,
      label="Elevación Tierra (°)",
      color="#2980b9",
      linewidth=1.8,
  )
  axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
  axes[0].axhspan(5, 20, color="green", alpha=0.15, label="Luz ideal (5°-20°)")
  axes[0].set_title(f"Visibilidad ({sitio_seleccionado})", fontsize=10)
  axes[0].set_xlabel("Horas", fontsize=8)
  axes[0].set_ylabel("Grados (°)", fontsize=8)
  axes[0].tick_params(labelsize=8)
  axes[0].legend(loc="best", fontsize=7)
  axes[0].grid(True, alpha=0.3)

  im1 = axes[1].imshow(info_dem["matriz_alt"], cmap="terrain")
  axes[1].set_title("Elevación DEM LOLA", fontsize=10)
  axes[1].tick_params(labelsize=8)
  fig.colorbar(im1, ax=axes[1], label="Altitud (m)")

  im2 = axes[2].imshow(info_dem["matriz_pend"], cmap="hot_r", vmin=0, vmax=25)
  axes[2].set_title("Mapa de Pendientes", fontsize=10)
  axes[2].tick_params(labelsize=8)
  fig.colorbar(im2, ax=axes[2], label="Inclinación (°)")

  plt.tight_layout()
  st.pyplot(fig, use_container_width=False)
else:
  # Cuando solo hay una gráfica, la centramos en una columna más estrecha para que se vea pequeña y limpia
  col_graf, _ = st.columns([1.3, 1])
  with col_graf:
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    ax.plot(
        horas_eje,
        grados_sol,
        label="Elevación Solar (°)",
        color="#f39c12",
        linewidth=1.8,
    )
    ax.plot(
        horas_eje,
        grados_tierra,
        label="Elevación Tierra (°)",
        color="#2980b9",
        linewidth=1.8,
    )
    ax.axhline(
        0, color="black", linestyle="--", linewidth=0.8, label="Horizonte (0°)"
    )
    ax.axhspan(5, 20, color="green", alpha=0.15, label="Luz ideal (5°-20°)")
    ax.set_title(f"Telemetría de Visibilidad ({sitio_seleccionado})", fontsize=10)
    ax.set_xlabel("Horas desde el inicio", fontsize=8)
    ax.set_ylabel("Grados (°)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
