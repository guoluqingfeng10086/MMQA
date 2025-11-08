import shapely.geometry as geometry
import requests
from lxml import etree
import rasterio
from rasterio.transform import rowcol
import math
from geopy.distance import geodesic
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import shape, Point
import fiona
from functools import lru_cache

# Geological Age
@lru_cache(maxsize=1)
def load_geologic_dataset(filepath: str):
    """ Load the geological dataset and create a spatial index """
    data = gpd.read_file(filepath)
    data.sindex
    return data

def get_geologic_epoch(lon: float, lat: float, filepath: str) -> str:
    """
    Retrieves descriptions of Martian geological ages corresponding to specified latitude and longitude locations
    """
    data = load_geologic_dataset(filepath)  # Transmission path
    point = geometry.Point(lon, lat)
    possible_matches_index = list(data.sindex.intersection(point.bounds))
    possible_matches = data.iloc[possible_matches_index]
    for _, row in possible_matches.iterrows():
        if row['geometry'].contains(point):
            return row.get('UnitDesc', None)
    return None

# HIRISE landform
class HiRISESearcher:
    def __init__(self):
        self.base_url = 'https://www.uahirise.org/'
        self.search_url = 'https://www.uahirise.org/results.php'
        self.headers = {
            'User-Agent': 'Mozilla/5.0',
            'Referer': 'https://www.uahirise.org/anazitisi.php'
        }

    def get_response(self, lon, lat, delta):
        params = {
            'lon_beg': str(max(0, lon - delta)),
            'lon_end': str(min(360, lon + delta)),
            'lat_beg': str(max(-90, lat - delta)),
            'lat_end': str(min(90, lat + delta)),
            'solar_all': 'true',
            'image_all': 'true',
            'order': 'WP.release_date'
        }
        response = requests.get(self.search_url, params=params, headers=self.headers, timeout=20)
        return response

    def get_info(self, response):
        html_con = etree.HTML(response.content)
        td_eles = html_con.xpath('//*[@class="catalog-cell-images"]')
        results = []
        for td in td_eles:
            item = {}
            a_href = td.xpath('a/@href')
            if a_href:
                item['id'] = a_href[0]
                item['url'] = self.base_url + a_href[0]
            desc = td.xpath('a/*/@alt')
            if desc:
                item['desc'] = desc[0]
            results.append(item)
        return results

USE_HIRISE = True
@lru_cache(maxsize=128)
def get_hirise_context(lat: float, lon: float):
    if not USE_HIRISE:
        print("🚫 HiRISE search is closed (USE_HIRISE=False)")
        return None, [], []
    if lon < 0:
        lon += 360
    searcher = HiRISESearcher()
    for delta in [0.1, 0.2, 0.3, 0.4, 0.5]:
        try:
            response = searcher.get_response(lon, lat, delta)
            results = searcher.get_info(response)
            if results:
                return delta, results, results[:3]
        except Exception as e:
            print(f"!!!HiRISE request failed：{e}")
            break
    return None, [], []

# albedo
# Mars latitude and longitude → Mars spherical projection coordinates
def mars_lonlat_to_meters(lon_deg, lat_deg, radius=3396000):
    lon_rad = math.radians(lon_deg)
    lat_rad = math.radians(lat_deg)
    x = radius * lon_rad
    y = radius * lat_rad
    return x, y

# Find the nearest valid cell value
def find_nearest_valid(data, mask, row, col, max_radius=10):
    for r in range(1, max_radius + 1):
        row_min = max(0, row - r)
        row_max = min(data.shape[0], row + r + 1)
        col_min = max(0, col - r)
        col_max = min(data.shape[1], col + r + 1)

        window = data[row_min:row_max, col_min:col_max]
        window_mask = mask[row_min:row_max, col_min:col_max]
        valid = window[window_mask > 0]

        if valid.size > 0:
            return valid[0]
    return None

# Albedo lookup function
@lru_cache(maxsize=1)
def load_albedo_src(tif_path):
    return rasterio.open(tif_path)

# The original function was changed to use a cached file object
def get_albedo_value(tif_path, lon_deg, lat_deg):
    scaling_factor = 1.4522365285e-05
    offset = 0.52414565669
    x, y = mars_lonlat_to_meters(lon_deg, lat_deg)
    src = load_albedo_src(tif_path)  # ✅Use cached objects instead of opening each time.
    row, col = rowcol(src.transform, x, y)
    data = src.read(1)
    mask = src.dataset_mask()

    if not (0 <= row < src.height and 0 <= col < src.width):
        return None

    raw_value = data[row, col]
    if mask[row, col] == 0:
        raw_value = find_nearest_valid(data, mask, row, col)
        if raw_value is None:
            return None
    return raw_value * scaling_factor + offset

# Ancient Lakes Inquiry
# Internal cache to avoid duplicate loading
@lru_cache(maxsize=1)
def load_paleolake_csv_cached(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def get_paleolake_context(csv_path, center_lat, center_lon, delta=2):
    """
    Retrieves paleolake data within a specified coordinate range of ±delta°.
    :param csv_path: Path to the paleolake CSV file
    :param center_lat: Latitude
    :param center_lon: Longitude
    :param delta: Search radius (in °)
    :return: A list, each element being a dictionary of paleolake attributes.
    """
    # Field Mapping
    BASIN_TYPE_MAP = {'CBL': 'closed-basin lake', 'OBL': 'open-basin lake'}
    VALLEY_TYPE_MAP = {'II': 'isolated inlet valley', 'VN': 'valley network'}
    df = load_paleolake_csv_cached(csv_path)
    lat_min, lat_max = center_lat - delta, center_lat + delta
    lon_min, lon_max = center_lon - delta, center_lon + delta
    # Filter lakes that meet the criteria
    nearby = df[
        (df["Lat. (N)"] >= lat_min) & (df["Lat. (N)"] <= lat_max) &
        (df["Lon. (E)"] >= lon_min) & (df["Lon. (E)"] <= lon_max)
    ]
    lake_infos = []
    for _, row in nearby.iterrows():
        lake_info = {
            "Basin Type": BASIN_TYPE_MAP.get(row['Basin Type'], row['Basin Type']),
            "Lat": row['Lat. (N)'],
            "Lon": row['Lon. (E)'],
            "Valley Type": VALLEY_TYPE_MAP.get(row['Valley Type'], row['Valley Type']),
            "Degradation": row['Basin Degradation State'],
            "Strahler Order": row['Strahler Order'],
        }
        if pd.notna(row.get('Strahler Order Reference')):
            lake_info["Strahler Reference"] = row['Strahler Order Reference']
        lake_infos.append(lake_info)

    return lake_infos

# Impact crater
@lru_cache(maxsize=1)
def load_crater_csv(crater_csv_path: str) -> pd.DataFrame:
    return pd.read_csv(crater_csv_path, low_memory=False)

def get_crater_context(crater_csv_path: str, lat: float, lon: float, delta=1.0):
    crater_df = load_crater_csv(crater_csv_path)
    db_lon = lon + 360 if lon < 0 else lon

    nearby = crater_df[
        crater_df['LAT_CIRC_IMG'].between(lat - delta, lat + delta) &
        crater_df['LON_CIRC_IMG'].between(db_lon - delta, db_lon + delta)
    ].copy()

    if nearby.empty:
        return []
    center = (lat, db_lon)
    nearby = nearby.dropna(subset=['LAT_CIRC_IMG', 'LON_CIRC_IMG'])
    nearby['distance'] = nearby.apply(
        lambda r: geodesic(center, (r['LAT_CIRC_IMG'], r['LON_CIRC_IMG'])).km,
        axis=1
    )
    nearby = nearby.sort_values(by='distance').head(3)
    result = []
    for _, row in nearby.iterrows():
        info = {
            'crater_id': row.get('CRATER_ID'),
            'lat': row.get('LAT_CIRC_IMG'),
            'lon': row.get('LON_CIRC_IMG'),
            'diameter_km': row.get('DIAM_CIRC_IMG'),
            'int_morph1': row.get('INT_MORPH1'),
            'lay_morph1': row.get('LAY_MORPH1'),
            'DEG_RIM': row.get('DEG_RIM'),
            'DEG_EJC': row.get('DEG_EJC'),
            'DEG_FLR': row.get('DEG_FLR'),
        }
        result.append(info)
    return result

# Mars parameters
R_MARS = 3396190.0
DEG2RAD = np.pi / 180.0
# Display all geological periods in their full names
AGE_MAPPING = {
    'Hesp. Noac.': 'Noachian-Hesperian',
    'Amaz. Hesp.': 'Hesperian-Amazonian',
}
@lru_cache(maxsize=1)
def load_valley_shapefile(shp_path: str) -> gpd.GeoDataFrame:
    """
    Load and parse the valley network shapefile, only once
    """
    records = []
    with fiona.open(shp_path, 'r') as src:
        for feat in src:
            geom = feat.get('geometry')
            try:
                if geom:
                    geom_obj = shape(geom)
                    if geom_obj.geom_type in ['LineString', 'MultiLineString']:
                        pts = list(geom_obj.coords) if geom_obj.geom_type == 'LineString' else [pt for part in geom_obj.geoms for pt in part.coords]
                        if len(pts) > 1:
                            feat['geometry'] = geom_obj
                            records.append(feat)
            except Exception:
                continue
        return gpd.GeoDataFrame(
            [r['properties'] for r in records],
            geometry=[r['geometry'] for r in records],
            crs=src.crs
        )
def get_valley_context(shp_path: str, lat: float, lon: float, bins_km=[0, 20, 100]):
    """
    Returns information on the river valley network (including geological age and type) within a bins_km radius around a given coordinate point.
    """
    df = load_valley_shapefile(shp_path).copy()

    # Coordinate transformation: Latitude and longitude → Mars spherical plane coordinates
    x0 = R_MARS * lon * DEG2RAD
    y0 = R_MARS * lat * DEG2RAD
    pt = Point(x0, y0)

    df['dist_km'] = df.geometry.distance(pt) / 1000.0
    df['lon_centroid'] = df.geometry.centroid.x / R_MARS / DEG2RAD
    df['lat_centroid'] = df.geometry.centroid.y / R_MARS / DEG2RAD

    # Geological Time Field Mapping
    if 'Age' in df.columns:
        df['age_std'] = df['Age'].map(AGE_MAPPING).fillna(df['Age'])

    matched = []
    for low, high in zip(bins_km[:-1], bins_km[1:]):
        subset = df[(df['dist_km'] > low) & (df['dist_km'] <= high)]
        if not subset.empty:
            matched.append((
                f"> {low} km to ≤ {high} km",
                subset[['dist_km', 'lat_centroid', 'lon_centroid', 'Length(km)', 'Age', 'age_std', 'Type']]
            ))

    return matched

# Topographic elevation
@lru_cache(maxsize=1)
def load_elevation_src(tif_path):
    return rasterio.open(tif_path)
# The original function was changed to use a cached object
def get_mars_elevation_direct(tif_path, lon_deg, lat_deg):
    src = load_elevation_src(tif_path)  # ✅ Use cached objects instead of opening each time
    row, col = rowcol(src.transform, lon_deg, lat_deg)
    if not (0 <= row < src.height and 0 <= col < src.width):
        print("❌ Coordinates outside image range")
        return None
    window = rasterio.windows.Window(col, row, 1, 1)
    value = src.read(1, window=window)[0, 0]
    mask_val = src.read_masks(1, window=window)[0, 0]
    if mask_val == 0:
        print("Current pixel: NoData")
        return None
    return value
PIXEL_SIZE = 0.25
LAT_START = -90.0
LON_START = -180.0
NUM_COLS = int(360 / PIXEL_SIZE)
NUM_ROWS = int(180 / PIXEL_SIZE)
def load_all_tifs(minerals):
    """Load all TIFFs at once and cache them"""
    tifs = {}
    for mineral, tif_path in minerals.items():
        with rasterio.open(tif_path) as src:
            band = src.read(1)
            # Reverse the row direction so that row 0 corresponds to lat=-90.
            tifs[mineral] = np.flipud(band)
    return tifs
def get_index_from_latlon(lat, lon):
    """
    Latitude and longitude are mapped to idx, consistent with the original database
    """
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("The input latitude and longitude are out of range: lat [-90,90], lon [-180,180]")
    row = int((lat - LAT_START) // PIXEL_SIZE)
    col = int((lon - LON_START) // PIXEL_SIZE)
    row = min(max(row, 0), NUM_ROWS - 1)
    col = min(max(col, 0), NUM_COLS - 1)
    idx = row * NUM_COLS + col + 1
    return idx
def get_mineral_abundance(lat, lon, minerals):
    """
    Input latitude and longitude, return (idx, dict) Fully compatible with the original function
    The values in dict are Python floats, -1, or NaN converted to None.
    """
    idx = get_index_from_latlon(lat, lon)
    row = (idx - 1) // NUM_COLS
    col = (idx - 1) % NUM_COLS
    tifs = load_all_tifs(minerals)
    mineral_dict = {}
    for mineral, band in tifs.items():
        value = band[row, col]
        if value == -1 or np.isnan(value):
            mineral_dict[mineral] = None
        else:
            mineral_dict[mineral] = float(value)
    if all(v is None for v in mineral_dict.values()):
        return idx, None
    else:
        return idx, mineral_dict

