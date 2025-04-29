import geopandas as gpd
import numpy as np
import shapely
import os
import folium
import streamlit as st
from shapely.geometry import Polygon
from sklearn.cluster import KMeans
from streamlit_folium import st_folium

# --- Configurar página ---
st.set_page_config(page_title="Subcomunas CABA", layout="wide")
st.title("📍 División de Comunas")

# --- Función para cargar comunas y manzanas ---
@st.cache_data

def cargar_datos():
    # Comunas
    barrios = gpd.read_file('https://cdn.buenosaires.gob.ar/datosabiertos/datasets/barrios/barrios.geojson')
    barrios.columns = barrios.columns.str.lower()
    comunas = barrios.dissolve(by='comuna', as_index=False)
    comunas = comunas.to_crs(22185)

    # Manzanas simplificadas
    manzanas = gpd.read_file('https://cdn.buenosaires.gob.ar/datosabiertos/datasets/secretaria-de-desarrollo-urbano/manzanas/mapa_manzanas.geojson')
    manzanas = manzanas.to_crs(22185)
    manzanas = manzanas[['geometry']].copy()
    manzanas['geometry'] = manzanas.simplify(tolerance=5, preserve_topology=True)

    return comunas, manzanas

# --- Dividir en grupos por centroides ---
def dividir_manzanas(manzanas_comuna, n_partes=6):
    manzanas_comuna = manzanas_comuna.copy()
    centroids = manzanas_comuna.geometry.centroid
    coords = np.array([[pt.x, pt.y] for pt in centroids])

    if len(coords) < n_partes:
        labels = np.zeros(len(coords))
    else:
        kmeans = KMeans(n_clusters=n_partes, random_state=0, n_init='auto')
        labels = kmeans.fit_predict(coords)

    manzanas_comuna['grupo'] = labels
    return manzanas_comuna

# --- Generar subcomunas y cachear por comuna ---
def cargar_o_generar_subcomunas(comuna_id, comunas, manzanas, n_partes=6):
    path = f"cache/subcomuna_{comuna_id}.geojson"
    os.makedirs("cache", exist_ok=True)

    if os.path.exists(path):
        return gpd.read_file(path)

    comuna_geom = comunas[comunas['comuna'] == comuna_id].geometry.values[0]
    manzanas_comuna = manzanas[manzanas.intersects(comuna_geom)].copy()
    manzanas_comuna = manzanas_comuna.clip(comuna_geom)

    if manzanas_comuna.empty:
        return gpd.GeoDataFrame(columns=['comuna', 'subparte', 'geometry'], crs=manzanas.crs)

    manzanas_divididas = dividir_manzanas(manzanas_comuna, n_partes=n_partes)

    subcomunas = []
    for grupo_id, grupo_manzanas in manzanas_divididas.groupby('grupo'):
        union_geom = grupo_manzanas.unary_union
        subcomunas.append({
            'comuna': comuna_id,
            'subparte': f'Comuna {comuna_id} - Parte {grupo_id + 1}',
            'geometry': union_geom
        })

    gdf_subcomunas = gpd.GeoDataFrame(subcomunas, crs=manzanas.crs)
    gdf_subcomunas.to_file(path, driver='GeoJSON')
    return gdf_subcomunas

# --- Cargar datos ---
comunas, manzanas = cargar_datos()

# --- Filtros UI ---
st.sidebar.header("Filtros")
comunas_disponibles = sorted(comunas['comuna'].unique())
comuna_seleccionada = st.sidebar.selectbox("Seleccioná una Comuna", comunas_disponibles)

# --- Cargar subcomunas según comuna seleccionada ---
gdf_subcomunas = cargar_o_generar_subcomunas(comuna_seleccionada, comunas, manzanas, n_partes=6)
gdf_subcomunas = gdf_subcomunas.to_crs(4326)

# --- Filtro adicional ---
subcomunas_disponibles = gdf_subcomunas['subparte'].tolist()
subparte_seleccionada = st.sidebar.selectbox("Seleccioná una Subparte", ["Todas"] + subcomunas_disponibles)

if subparte_seleccionada == "Todas":
    subcomunas_filtradas = gdf_subcomunas
else:
    subcomunas_filtradas = gdf_subcomunas[gdf_subcomunas['subparte'] == subparte_seleccionada]

# --- Crear mapa ---
m = folium.Map(location=[-34.61, -58.42], zoom_start=13, tiles='cartodb positron')

for _, row in subcomunas_filtradas.iterrows():
    folium.GeoJson(
        row['geometry'],
        tooltip=row['subparte'],
        style_function=lambda feature: {
            'color': 'black',
            'weight': 2,
            'fillColor': '#3388ff',
            'fillOpacity': 0.3,
        }
    ).add_to(m)

st_data = st_folium(m, width=1000, height=700)