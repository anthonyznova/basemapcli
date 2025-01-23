#!/usr/bin/env python3
import os
import sys
import math
import io
import time
import logging
import asyncio
import aiohttp
from PIL import Image
import numpy as np
from osgeo import gdal, osr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MapGenerator:
    def __init__(self):
        self.tile_size = 256
        self.max_tiles = 6
        self.earth_circumference = 40075016.686  # WGS84 ellipsoid

    def lat_lon_to_tile(self, lat, lon, zoom, tile_source):
        if tile_source == 'arc':
            lat_rad = math.radians(lat)
            n = 1 << zoom
            x = (lon + 180.0) / 360.0 * n
            y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
        elif tile_source == 'sentinel':
            lon_merc, lat_merc = self.wgs84_to_web_mercator(lon, lat)
            x = (lon_merc + 20037508.342789244) * (1 << zoom) / 40075016.68557849
            y = (20037508.342789244 - lat_merc) * (1 << zoom) / 40075016.68557849
        return (int(x), int(y))

    def wgs84_to_web_mercator(self, lon, lat):
        x = lon * 20037508.34 / 180
        y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)
        y = y * 20037508.34 / 180
        return x, y

    def web_mercator_to_wgs84(self, x, y):
        lon = x / 20037508.34 * 180
        lat = 180/math.pi * (2 * math.atan(math.exp(y / 20037508.34 * 180 * math.pi / 180)) - math.pi/2)
        return lon, lat

    def tile_bounds(self, x, y, zoom, tile_source):
        if tile_source == 'arc':
            n = 1 << zoom
            lon_west = (x / n) * 360.0 - 180.0
            lat_north = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
            lon_east = ((x + 1) / n) * 360.0 - 180.0
            lat_south = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 1) / n))))
        elif tile_source == 'sentinel':
            resolution = 40075016.68557849 / (1 << zoom)
            west = x * resolution - 20037508.342789244
            north = 20037508.342789244 - y * resolution
            east = (x + 1) * resolution - 20037508.342789244
            south = 20037508.342789244 - (y + 1) * resolution
            return (north, west, south, east)
        return (lat_north, lon_west, lat_south, lon_east)

    def calculate_optimal_zoom(self, bounds, tile_source):
        north, west, south, east = bounds
        low, high, best_zoom = 0, 19, 0
        
        while low <= high:
            mid = (low + high) // 2
            tiles = set()
            
            for lat in [north, south]:
                for lon in [west, east]:
                    x, y = self.lat_lon_to_tile(lat, lon, mid, tile_source)
                    tiles.add((x, y))
            
            if not tiles:
                continue
                
            xs = [t[0] for t in tiles]
            ys = [t[1] for t in tiles]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            tile_count = (max_x - min_x + 1) * (max_y - min_y + 1)
            
            if tile_count <= self.max_tiles**2:
                best_zoom = mid
                low = mid + 1
            else:
                high = mid - 1
                
        return best_zoom

    async def fetch_tile(self, session, x, y, zoom, tile_source):
        try:
            if tile_source == 'arc':
                url = f'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}'
            elif tile_source == 'sentinel':
                url = f'https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/GoogleMapsCompatible/{zoom}/{y}/{x}.jpg'
            else:
                raise ValueError(f'Invalid tile source: {tile_source}')

            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    return (x, y, Image.open(io.BytesIO(content)).convert('RGB'))
                logging.error(f'Tile fetch failed: {url} (HTTP {response.status})')
                return None
        except Exception as e:
            logging.error(f'Error fetching tile {x},{y}: {str(e)}')
            return None

    async def download_tiles(self, bounds, zoom, tile_source):
        north, west, south, east = bounds
        
        nw_x, nw_y = self.lat_lon_to_tile(north, west, zoom, tile_source)
        se_x, se_y = self.lat_lon_to_tile(south, east, zoom, tile_source)
        
        min_x = min(nw_x, se_x)
        max_x = max(nw_x, se_x)
        min_y = min(nw_y, se_y)
        max_y = max(nw_y, se_y)
        
        width = (max_x - min_x + 1) * self.tile_size
        height = (max_y - min_y + 1) * self.tile_size
        canvas = Image.new('RGB', (width, height))
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    tasks.append(self.fetch_tile(session, x, y, zoom, tile_source))
            
            results = await asyncio.gather(*tasks)
            
            for result in results:
                if result:
                    x, y, tile = result
                    px = (x - min_x) * self.tile_size
                    py = (y - min_y) * self.tile_size
                    canvas.paste(tile, (px, py))
        
        return canvas, (min_x, min_y, max_x, max_y)

    def create_geotiff(self, image_path, bounds, output_path, zoom, tile_source, utm=False):
        img = Image.open(image_path)
        arr = np.array(img)
        
        # Create initial GeoTIFF
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(
            output_path,
            arr.shape[1],
            arr.shape[0],
            3,
            gdal.GDT_Byte,
            options=['COMPRESS=LZW', 'PREDICTOR=2']
        )
        
        if tile_source == 'sentinel':
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(3857)
            nw = self.tile_bounds(bounds[0], bounds[1], zoom, tile_source)
            se = self.tile_bounds(bounds[2], bounds[3], zoom, tile_source)
            west, east = nw[1], se[3]
            north, south = nw[0], se[2]
            pixel_width = (east - west) / arr.shape[1]
            pixel_height = (north - south) / arr.shape[0]
            ds.SetGeoTransform([west, pixel_width, 0, north, 0, -pixel_height])
        else:
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)
            west = self.tile_bounds(bounds[0], bounds[1], zoom, tile_source)[1]
            east = self.tile_bounds(bounds[2], bounds[3], zoom, tile_source)[3]
            north = self.tile_bounds(bounds[0], bounds[1], zoom, tile_source)[0]
            south = self.tile_bounds(bounds[2], bounds[3], zoom, tile_source)[2]
            pixel_width = (east - west) / arr.shape[1]
            pixel_height = (north - south) / arr.shape[0]
            ds.SetGeoTransform([west, pixel_width, 0, north, 0, -pixel_height])
        
        ds.SetProjection(srs.ExportToWkt())
        
        for i in range(3):
            band = ds.GetRasterBand(i+1)
            band.WriteArray(arr[:,:,i])
        
        # Clean up initial dataset
        ds.FlushCache()
        ds = None

        if utm:
            try:
                # Open with read-only access
                ds = gdal.Open(output_path, gdal.GA_ReadOnly)
                if ds is None:
                    logging.error("Failed to open GeoTIFF for UTM conversion")
                    return

                # Get center coordinates
                geotransform = ds.GetGeoTransform()
                width = ds.RasterXSize
                height = ds.RasterYSize

                if tile_source == 'sentinel':
                    center_x = geotransform[0] + (width/2) * geotransform[1]
                    center_y = geotransform[3] + (height/2) * geotransform[5]
                    lon, lat = self.web_mercator_to_wgs84(center_x, center_y)
                else:
                    lon = geotransform[0] + (width/2) * geotransform[1]
                    lat = geotransform[3] + (height/2) * geotransform[5]

                # Calculate UTM zone
                zone_number = int((lon + 180) // 6) + 1
                epsg_code = 32600 + zone_number if lat >= 0 else 32700 + zone_number

                # Close dataset before reprojection
                ds = None

                # Create temp path with proper extension
                temp_path = output_path.replace('.tif', '_utm.tmp.tif')
                
                # Reproject with explicit format
                warp_options = gdal.WarpOptions(
                    format='GTiff',
                    dstSRS=f'EPSG:{epsg_code}',
                    resampleAlg='bilinear',
                    options=['COMPRESS=LZW', 'PREDICTOR=2']
                )
                warped_ds = gdal.Warp(temp_path, output_path, options=warp_options)
                warped_ds = None  # Close warped dataset

                # Windows-safe file replacement with retries
                if os.path.exists(temp_path):
                    for _ in range(5):  # Retry up to 5 times
                        try:
                            os.remove(output_path)
                            os.rename(temp_path, output_path)
                            break
                        except PermissionError:
                            time.sleep(0.2)
                    else:
                        logging.error("Failed to replace file after multiple attempts")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                
            except Exception as e:
                logging.error(f"UTM conversion failed: {str(e)}")
            finally:
                # Force cleanup
                if 'ds' in locals(): ds = None
                if 'warped_ds' in locals(): warped_ds = None
                if os.path.exists(temp_path):
                    try: os.remove(temp_path)
                    except: pass

    async def process_input(self, config_path, output_dir):
        with open(config_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        utm = False
        tile_source = 'arc'
        if len(lines) in (3, 4):
            map_name = lines[0]
            lat, lon, size_km = map(float, lines[1].split(','))
            tile_source = lines[2].lower()
            if len(lines) == 4:
                utm = lines[3].lower() == 'utm'
            
            d_lat = size_km / 111.32
            d_lon = size_km / (111.32 * math.cos(math.radians(lat)))
            bounds = (
                lat + d_lat/2,
                lon - d_lon/2,
                lat - d_lat/2,
                lon + d_lon/2
            )
        elif len(lines) in (6, 7):
            map_name = lines[0]
            coords = {}
            for line in lines[1:5]:
                parts = line.split(',')
                if parts[0] == 'N':
                    coords['north'] = float(parts[1])
                    coords['west'] = float(parts[2])
                elif parts[0] == 'E':
                    coords['east'] = float(parts[2])
                elif parts[0] == 'S':
                    coords['south'] = float(parts[1])
                elif parts[0] == 'W':
                    coords['west'] = float(parts[2])
            tile_source = lines[5].lower()
            if len(lines) == 7:
                utm = lines[6].lower() == 'utm'
            bounds = (coords['north'], coords['west'], coords['south'], coords['east'])
        else:
            raise ValueError('Invalid input file format')
        
        zoom = self.calculate_optimal_zoom(bounds, tile_source)
        logging.info(f'Calculated optimal zoom: {zoom}')
        
        canvas, tile_bounds = await self.download_tiles(bounds, zoom, tile_source)
        os.makedirs(output_dir, exist_ok=True)
        
        jpg_path = os.path.join(output_dir, f'{map_name}.jpg')
        tif_path = os.path.join(output_dir, f'{map_name}.tif')
        
        canvas.save(jpg_path, quality=85)
        self.create_geotiff(jpg_path, tile_bounds, tif_path, zoom, tile_source, utm)
        logging.info(f'Successfully generated files in {output_dir}')

async def main():
    generator = MapGenerator()
    
    print("\n=== Map Generator ===")
    print("Please provide the following inputs:")
    
    try:
        config_path = input("\nPath to config.txt: ").strip('"')
        if not os.path.isfile(config_path):
            print(f"Error: File not found - {config_path}")
            return

        output_dir = input("Path to output directory: ").strip('"')
        os.makedirs(output_dir, exist_ok=True)

        print("\nProcessing...")
        await generator.process_input(config_path, output_dir)
        print("\nProcessing complete!")

    except Exception as e:
        logging.error(str(e))
        print(f"\nError: {str(e)}")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nCritical error: {str(e)}")