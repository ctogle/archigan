"""Utility script for boston buildings GIS data

    `parse_GIS_bostonbuildings_2016` parses GIS data from:
        https://www.arcgis.com/home/item.html?id=c423eda7a64b49c98a9ebdf5a6b7e135
        https://data.boston.gov/dataset/parcels-2016-data-full/resource/d53d8e93-034d-4dd0-b59f-8634f4df3a71

    `ParcelInputLayer/ParcelOutputLayer` can represent the mapping from
    parcel outline to building footprint as seen in the above GIS data

"""
import cv2
import os
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from fiona.crs import from_epsg
from tqdm import tqdm_notebook as tqdm
#from tqdm import tqdm
from archigan.datalayer import Layer


def _parse_polygon(py):
    """Parse the Polygon/MultiPolygon objects from shp files"""
    if isinstance(py, Polygon):
        bound = list(zip(*py.exterior.coords.xy))
        holes = [list(zip(*hole.coords.xy)) for hole in py.interiors]
        #return bound, holes
        return [bound]
    elif isinstance(py, MultiPolygon):
        parts = [_parse_polygon(part) for part in py.geoms]
        return [x for y in parts for x in y]
    else:
        print('bad shape type:', type(py))


def _bbox(pts):
    """Find the AABB bounding box for a set of points"""
    x, y = pts[0]
    ax, ay, bx, by = x, y, x, y
    for i in range(1, len(pts)):
        x, y = pts[i]
        ax = x if x < ax else ax
        ay = y if y < ay else ay
        bx = x if x > bx else bx
        by = y if y > by else by
    return ax, ay, bx, by


def _to_first_quadrant(parcel, building):
    """Translate parcel and building to the first quadrant"""
    pts = [p for part in (parcel + building) for p in part]
    ax, ay, bx, by = _bbox(pts)
    height, width = (bx - ax), (by - ay)
    parcel = [[(x - ax, y - ay) for x, y in part] for part in parcel]
    building = [[(x - ax, y - ay) for x, y in part] for part in building]
    return parcel, building, height, width


def parse_GIS_bostonbuildings_2016(path):
    """Parser for 2016 boston buildings GIS data:
        https://www.arcgis.com/home/item.html?id=c423eda7a64b49c98a9ebdf5a6b7e135
        https://data.boston.gov/dataset/parcels-2016-data-full/resource/d53d8e93-034d-4dd0-b59f-8634f4df3a71
    """
    # read in GIS data via geopandas
    parcels = os.path.join(path, 'Parcels_2016_Data_Full.shp')
    print(f'Reading parcel data: {parcels}')
    parcels = gpd.read_file(parcels)
    buildings = os.path.join(path, 'boston_buildings.shp')
    print(f'Reading footprint data: {buildings}')
    buildings = gpd.read_file(buildings)
    # reproject buildings using parcels' CRS
    epsg = parcels.crs.to_epsg()
    print(f'Reprojecting footprint data to EPSG: {epsg}')
    buildings['geometry'] = buildings['geometry'].to_crs(epsg=epsg)
    buildings.crs = from_epsg(epsg)
    # create samples of complete pairs of parcel/building outlines
    pidlookup = {row.PID: j for j, row in parcels.iterrows()}
    samples = []
    for j, row in tqdm(buildings.iterrows(), desc='Preparing parcel/footprint data'):
        if row.PARCEL_ID in pidlookup:
            parcel = _parse_polygon(parcels.iloc[pidlookup[row.PARCEL_ID]].geometry)
            building = _parse_polygon(row.geometry)
            parcel, building, height, width = _to_first_quadrant(parcel, building)
            sample = {
                'byclass': {
                    'parcel': parcel,
                    'building': building,
                },
                'height': height,
                'width': width,
            }
            samples.append(sample)
    return samples


class ParcelInputLayer(Layer):

    labels = dict([
        ('parcel', 5),
    ])

    def parcel_mask(self, byclass, height, width):
        margin = 0.1 * max(height, width)
        parcel = {'parcel': byclass['parcel']}
        mask = self.mask(parcel, height, width, margin=margin)
        # erode so that footprints are better contained by parcels on average...
        kernel = np.ones((3, 3), np.uint8)
        layer = cv2.erode(mask, kernel, iterations=3)
        layer = self.norm_mask(layer)
        return self.norm_mask(layer)

    def __call__(self, byclass, height, width):
        layer = self.parcel_mask(byclass, height, width)
        layer = layer.max() - layer
        return layer


class ParcelOutputLayer(ParcelInputLayer):

    labels = dict([
        ('parcel', 5),
        ('building', 3),
    ])

    def __call__(self, byclass, height, width):
        layer = self.parcel_mask(byclass, height, width)
        layer = layer.max() - layer
        # blend in building footprints
        for cls, label in self.labels.items():
            for py in byclass[cls]:
                ## determine if a dilated mask of py intersects the mask
                ## which implies its a feature on the boundary of the footprint
                margin = 0.1 * max(height, width) # ugh that this is in 2 places...
                pymask = self.mask({cls: [py]}, height, width, margin=margin)
                kernel = np.ones((3, 3), np.uint8)
                pymask = cv2.erode(pymask.copy(), kernel, iterations=3)
                weight = (pymask == 0).astype(int)
                layer = weight * label + (1 - weight) * layer
        return layer
