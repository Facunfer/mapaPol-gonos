import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
import requests
import io
import folium
from folium import FeatureGroup
from shapely.geometry import Polygon
from sklearn.cluster import KMeans
import streamlit as st
from streamlit_folium import st_folium

# --- 2. Configuración de la página ---
st.set_page_config(page_title="Subcomunas CABA", layout="wide")

st.title("📍 División de Comunas de CABA en Subpartes")

# --- 3. Función para cargar los datos ---
@st.cache_data
def cargar_datos():
    # Comunas
    barrios = gpd.read_file('https://cdn.buenosaires.gob.ar/datosabiertos/datasets/barrios/barrios.geojson')
    barrios.columns = barrios.columns.str.lower()
    comunas = barrios.dissolve(by='comuna', as_index=False)
    comunas = comunas.to_crs(22185)

    # Manzanas
    manzanas = gpd.read_file('https://cdn.buenosaires.gob.ar/datosabiertos/datasets/secretaria-de-desarrollo-urbano/manzanas/mapa_manzanas.geojson')
    manzanas = manzanas.to_crs(22185)

    return comunas, manzanas

# --- 4. Función para dividir en subpartes ---
def dividir_manzanas(manzanas_comuna, n_partes=6):
    manzanas_comuna['centroid_x'] = manzanas_comuna.geometry.centroid.x
    manzanas_comuna['centroid_y'] = manzanas_comuna.geometry.centroid.y
    coords = manzanas_comuna[['centroid_x', 'centroid_y']].to_numpy()

    if len(coords) < n_partes:
        labels = np.zeros(len(coords))
    else:
        kmeans = KMeans(n_clusters=n_partes, random_state=0, n_init='auto')
        labels = kmeans.fit_predict(coords)

    manzanas_comuna['grupo'] = labels
    return manzanas_comuna

# --- 5. Crear las subcomunas ---
@st.cache_data
def generar_subcomunas(_comunas, _manzanas):
    subcomunas = []
    for idx, row in _comunas.iterrows():
        comuna_id = row['comuna']
        comuna_geom = row.geometry
        manzanas_en_comuna = _manzanas[_manzanas.intersects(comuna_geom)].copy()
        manzanas_en_comuna = manzanas_en_comuna.clip(comuna_geom)

        if manzanas_en_comuna.empty:
            continue

        manzanas_divididas = dividir_manzanas(manzanas_en_comuna)

        for grupo_id, grupo_manzanas in manzanas_divididas.groupby('grupo'):
            union_geom = grupo_manzanas.unary_union
            subcomunas.append({
                'comuna': comuna_id,
                'subparte': f'Comuna {comuna_id} - Parte {grupo_id + 1}',
                'geometry': union_geom
            })

    gdf_subcomunas = gpd.GeoDataFrame(subcomunas, crs='EPSG:22185')
    return gdf_subcomunas
# --- 6. Cargar datos ---
comunas, manzanas = cargar_datos()
gdf_subcomunas = generar_subcomunas(comunas, manzanas)
gdf_subcomunas_wgs84 = gdf_subcomunas.to_crs(4326)

# --- 7. Barra lateral con filtros ---
st.sidebar.header("Filtros")

comunas_disponibles = sorted(gdf_subcomunas_wgs84['comuna'].unique())
comuna_seleccionada = st.sidebar.selectbox("Seleccioná una Comuna", comunas_disponibles)

subcomunas_disponibles = gdf_subcomunas_wgs84[gdf_subcomunas_wgs84['comuna'] == comuna_seleccionada]['subparte'].tolist()
subparte_seleccionada = st.sidebar.selectbox("Seleccioná una Subparte", ["Todas"] + subcomunas_disponibles)

# --- 8. Crear el mapa ---
m = folium.Map(location=[-34.61, -58.42], zoom_start=13, tiles='cartodb positron')

# Filtrar datos
if subparte_seleccionada == "Todas":
    subcomunas_filtradas = gdf_subcomunas_wgs84[gdf_subcomunas_wgs84['comuna'] == comuna_seleccionada]
else:
    subcomunas_filtradas = gdf_subcomunas_wgs84[gdf_subcomunas_wgs84['subparte'] == subparte_seleccionada]

# Agregar polígonos
for idx, row in subcomunas_filtradas.iterrows():
    parte = row['subparte']
    folium.GeoJson(
        row['geometry'],
        tooltip=parte,
        style_function=lambda feature: {
            'color': 'black',
            'weight': 2,
            'fillColor': '#3388ff',
            'fillOpacity': 0.3,
        }
    ).add_to(m)

# --- 9. Mostrar mapa ---
st_data = st_folium(m, width=1000, height=700)