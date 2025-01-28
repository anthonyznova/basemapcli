#!/usr/bin/env python3
import os
import sys
import math
import io
import time
import logging
import asyncio
import aiohttp
import argparse
from typing import Tuple, Dict, List, Optional, Any
from dataclasses import dataclass
from PIL import Image
import numpy as np
from osgeo import gdal, osr

# Configure logging
def setup_logging():
    """Setup logging configuration with file reset"""
    log_file = 'map_generator.log'
    if os.path.exists(log_file):
        os.remove(log_file)  # Reset log file
    
    logging.basicConfig(
        level=logging.DEBUG,  # Changed from INFO to DEBUG for more verbosity
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # Stream to stdout instead of stderr
            logging.FileHandler(log_file)
        ]
    )
    
    # Add initial log entries
    logging.info("Map Generator Started")
    logging.info("=" * 50)

@dataclass
class MapConfig:
    """Configuration data for map generation."""
    map_name: str
    coords: Tuple[float, float, float]  # x/lon, y/lat, size_km
    tile_source: str
    utm_zone: Optional[int] = None  # None means WGS84, int specifies UTM zone

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Map Generator Tool',
        usage='''%(prog)s - x - y - size - mapname - source - outputdir [-nad83 zone]

Required arguments:
  x           X coordinate (longitude for WGS84 or easting for UTM)
  y           Y coordinate (latitude for WGS84 or northing for UTM)
  size        Size in kilometers (max 100)
  mapname     Name of the output map
  source      Tile source (arc or sentinel)
  outputdir   Path to output directory

Optional arguments:
  -nad83      UTM NAD83 zone number (required only when using UTM coordinates)

Example WGS84:
  %(prog)s - -79.389318 - 43.641703 - 2 - toronto - arc - /path/to/output

Example UTM NAD83:
  %(prog)s - 630084 - 4833438 - 2 - toronto - arc - /path/to/output -nad83 17
''')
    
    # Positional arguments with dash separators
    parser.add_argument('x_coord', nargs=2, 
                       help='Dash followed by X coordinate (longitude for WGS84 or easting for UTM)')
    
    parser.add_argument('y_coord', nargs=2,
                       help='Dash followed by Y coordinate (latitude for WGS84 or northing for UTM)')
    
    parser.add_argument('size_param', nargs=2,
                       help='Dash followed by size in kilometers')
    
    parser.add_argument('map_name', nargs=2,
                       help='Dash followed by name of the output map')
    
    parser.add_argument('source_param', nargs=2,
                       help='Dash followed by tile source (arc or sentinel)')
    
    parser.add_argument('output_dir', nargs=2,
                       help='Dash followed by path to output directory')
    
    # Optional NAD83 zone
    parser.add_argument('-nad83', type=int, required=False,
                       help='Optional: UTM NAD83 zone number')
    
    args = parser.parse_args()
    
    # Extract the actual values (second item in each pair)
    x = float(args.x_coord[1])
    y = float(args.y_coord[1])
    size = float(args.size_param[1])
    mapname = args.map_name[1]
    source = args.source_param[1].lower()
    output = args.output_dir[1]
    
    # Validate source
    if source not in ['arc', 'sentinel']:
        parser.error("Source must be either 'arc' or 'sentinel'")
    
    # Create a simple namespace with the processed values
    return argparse.Namespace(
        x=x,
        y=y,
        size=size,
        mapname=mapname,
        source=source,
        nad83=args.nad83,
        output=output
    )

class MapGenerator:
    def __init__(self, tile_size: int = 256, max_tiles: int = 6, rate_limit: float = 0.1):
        self.tile_size = tile_size
        self.max_tiles = max_tiles
        self.rate_limit = rate_limit
        self.earth_circumference = 40075016.686
        self.last_request_time = 0.0
        
        self.tile_sources = {
            'arc': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile',
            'sentinel': 'https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/GoogleMapsCompatible'
        }

    async def _enforce_rate_limit(self):
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit:
            await asyncio.sleep(self.rate_limit - time_since_last)
        self.last_request_time = time.time()

    def lat_lon_to_tile(self, lat: float, lon: float, zoom: int, tile_source: str) -> Tuple[int, int]:
        """Convert latitude/longitude to tile coordinates."""
        if tile_source == 'arc':
            lat_rad = math.radians(lat)
            n = 1 << zoom
            x = (lon + 180.0) / 360.0 * n
            y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
        elif tile_source == 'sentinel':
            lon_merc, lat_merc = self.wgs84_to_web_mercator(lon, lat)
            x = (lon_merc + 20037508.342789244) * (1 << zoom) / 40075016.68557849
            y = (20037508.342789244 - lat_merc) * (1 << zoom) / 40075016.68557849
        else:
            raise ValueError(f'Invalid tile source: {tile_source}')
        return (int(x), int(y))

    def wgs84_to_web_mercator(self, lon: float, lat: float) -> Tuple[float, float]:
        """Convert WGS84 coordinates to Web Mercator."""
        if not (-180 <= lon <= 180 and -85.051129 <= lat <= 85.051129):
            raise ValueError('Coordinates out of bounds for Web Mercator projection')
        
        x = lon * 20037508.34 / 180
        y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)
        y = y * 20037508.34 / 180
        return x, y

    def web_mercator_to_wgs84(self, x: float, y: float) -> Tuple[float, float]:
        """Convert Web Mercator coordinates to WGS84."""
        lon = x / 20037508.34 * 180
        lat = 180/math.pi * (2 * math.atan(math.exp(y / 20037508.34 * 180 * math.pi / 180)) - math.pi/2)
        return lon, lat
    
    def utm_to_wgs84(self, easting: float, northing: float, zone: int) -> Tuple[float, float]:
        """Convert UTM NAD83 coordinates to WGS84 latitude/longitude."""
        utm_srs = osr.SpatialReference()
        utm_srs.SetWellKnownGeogCS('NAD83')
        utm_srs.SetUTM(zone, True)  # True for Northern hemisphere
        
        if hasattr(utm_srs, 'SetAxisMappingStrategy'):
            utm_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        wgs84_srs = osr.SpatialReference()
        wgs84_srs.SetWellKnownGeogCS('WGS84')
        
        if hasattr(wgs84_srs, 'SetAxisMappingStrategy'):
            wgs84_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        transform = osr.CoordinateTransformation(utm_srs, wgs84_srs)
        lon, lat, _ = transform.TransformPoint(easting, northing)
        return lon, lat
    
    async def download_tiles(self, bounds: Tuple[float, float, float, float], zoom: int, 
                           tile_source: str) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """Download and combine all required tiles for the specified bounds."""
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
        
        total_tiles = (max_x - min_x + 1) * (max_y - min_y + 1)
        completed_tiles = 0
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    tasks.append(self.fetch_tile(session, x, y, zoom, tile_source))
            
            for result in asyncio.as_completed(tasks):
                tile_result = await result
                if tile_result:
                    x, y, tile = tile_result
                    px = (x - min_x) * self.tile_size
                    py = (y - min_y) * self.tile_size
                    canvas.paste(tile, (px, py))
                
                completed_tiles += 1
                progress = (completed_tiles / total_tiles) * 100
                logging.info(f'Download progress: {progress:.1f}% ({completed_tiles}/{total_tiles} tiles)')
        
        return canvas, (min_x, min_y, max_x, max_y)

    def create_geotiff(self, image_path: str, bounds: Tuple[int, int, int, int], 
                      output_path: str, zoom: int, tile_source: str, 
                      utm_zone: Optional[int] = None):
        """Create a georeferenced TIFF file from the downloaded image."""
        img = Image.open(image_path)
        arr = np.array(img)
        
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(
            output_path,
            arr.shape[1],
            arr.shape[0],
            3,
            gdal.GDT_Byte,
            options=['COMPRESS=LZW', 'PREDICTOR=2', 'TILED=YES']
        )
        
        if tile_source == 'sentinel':
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(3857)
            nw = self.tile_bounds(bounds[0], bounds[1], zoom, tile_source)
            se = self.tile_bounds(bounds[2], bounds[3], zoom, tile_source)
            west, east = nw[1], se[3]
            north, south = nw[0], se[2]
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
        
        ds.FlushCache()
        ds = None

        if utm_zone is not None:
            try:
                epsg_code = 26900 + utm_zone  # NAD83 UTM zones start at 26901
                
                temp_path = output_path.replace('.tif', '_utm.tmp.tif')
                
                warp_options = gdal.WarpOptions(
                    format='GTiff',
                    dstSRS=f'EPSG:{epsg_code}',
                    resampleAlg='bilinear',
                    options=['COMPRESS=LZW', 'PREDICTOR=2', 'TILED=YES']
                )
                
                gdal.Warp(temp_path, output_path, options=warp_options)

                if os.path.exists(temp_path):
                    os.replace(temp_path, output_path)
                
            except Exception as e:
                logging.error(f"UTM NAD83 conversion failed: {str(e)}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise

    def validate_config(self, config: MapConfig) -> bool:
        """Validate the configuration parameters."""
        if not config.map_name or not config.map_name.strip():
            logging.error("Invalid map name")
            return False
            
        if config.tile_source not in self.tile_sources:
            logging.error(f"Invalid tile source. Must be one of: {', '.join(self.tile_sources.keys())}")
            return False
            
        x, y, size_km = config.coords
        
        if config.utm_zone is not None:
            # UTM coordinate validation
            if size_km <= 0 or size_km > 100:
                logging.error("Invalid size (must be between 0 and 100 km)")
                return False
            if not (100000 <= x <= 1000000):  # More generous easting range
                logging.error(f"UTM easting out of valid range: {x}")
                return False
            if not (0 <= y <= 10000000):  # Northing range
                logging.error(f"UTM northing out of valid range: {y}")
                return False
        else:
            # WGS84 coordinate validation
            if not (-90 <= y <= 90):
                logging.error("Invalid latitude value")
                return False
            if not (-180 <= x <= 180):
                logging.error("Invalid longitude value")
                return False
            if size_km <= 0 or size_km > 100:
                logging.error("Invalid size (must be between 0 and 100 km)")
                return False
        
        return True
    
    def calculate_bounds(self, config: MapConfig) -> Tuple[float, float, float, float]:
        """
        Calculate map bounds in WGS84 coordinates.
        Returns: Tuple[north, west, south, east]
        """
        x, y, size_km = config.coords
        
        if config.utm_zone is not None:
            # UTM input handling
            half_size = size_km * 500
            corners = [
                (x - half_size, y + half_size),
                (x + half_size, y + half_size),
                (x - half_size, y - half_size),
                (x + half_size, y - half_size)
            ]
            
            wgs84_corners = []
            for corner_x, corner_y in corners:
                lon, lat = self.utm_to_wgs84(corner_x, corner_y, config.utm_zone)
                wgs84_corners.append((lat, lon))
            
            lats = [c[0] for c in wgs84_corners]
            lons = [c[1] for c in wgs84_corners]
            
            return (max(lats), min(lons), min(lats), max(lons))
        else:
            # WGS84 input handling
            lat, lon = y, x  # This is the key fix - proper assignment of lat/lon
            
            # Calculate degree deltas
            d_lat = size_km / 111.32
            d_lon = size_km / (111.32 * math.cos(math.radians(lat)))
            
            # Calculate bounds (north, west, south, east)
            north = lat + d_lat/2
            south = lat - d_lat/2
            west = lon - d_lon/2
            east = lon + d_lon/2
            
            logging.debug(f"WGS84 input - Lat: {lat}, Lon: {lon}, Size: {size_km}km")
            logging.debug(f"Calculated bounds: N={north}, W={west}, S={south}, E={east}")
            
            return (north, west, south, east)

    def tile_bounds(self, x: int, y: int, zoom: int, tile_source: str) -> Tuple[float, float, float, float]:
        """Calculate the bounds of a specific tile."""
        if tile_source == 'arc':
            n = 1 << zoom
            lon_west = (x / n) * 360.0 - 180.0
            lat_north = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
            lon_east = ((x + 1) / n) * 360.0 - 180.0
            lat_south = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 1) / n))))
            return (lat_north, lon_west, lat_south, lon_east)
        elif tile_source == 'sentinel':
            resolution = 40075016.68557849 / (1 << zoom)
            west = x * resolution - 20037508.342789244
            north = 20037508.342789244 - y * resolution
            east = (x + 1) * resolution - 20037508.342789244
            south = 20037508.342789244 - (y + 1) * resolution
            return (north, west, south, east)
        raise ValueError(f'Invalid tile source: {tile_source}')

    def calculate_optimal_zoom(self, bounds: Tuple[float, float, float, float], tile_source: str) -> int:
        """Calculate the optimal zoom level for the given bounds."""
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

    async def fetch_tile(self, session: aiohttp.ClientSession, x: int, y: int, zoom: int, 
                        tile_source: str) -> Optional[Tuple[int, int, Image.Image]]:
        """Fetch a single tile from the specified source."""
        await self._enforce_rate_limit()
        
        try:
            if tile_source == 'arc':
                url = f'{self.tile_sources["arc"]}/{zoom}/{y}/{x}'
            elif tile_source == 'sentinel':
                url = f'{self.tile_sources["sentinel"]}/{zoom}/{y}/{x}.jpg'
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

async def main():
    """Main entry point for the map generation tool."""
    # Setup logging
    setup_logging()
    
    try:
        # Parse command line arguments
        args = parse_arguments()
        logging.info("Command line arguments:")
        logging.info(f"  X coordinate: {args.x}")
        logging.info(f"  Y coordinate: {args.y}")
        logging.info(f"  Size (km): {args.size}")
        logging.info(f"  Map name: {args.mapname}")
        logging.info(f"  Source: {args.source}")
        logging.info(f"  NAD83 zone: {args.nad83 if args.nad83 else 'Not using NAD83'}")
        logging.info(f"  Output directory: {args.output}")
        
        # Create output directory
        os.makedirs(args.output, exist_ok=True)
        logging.info(f"Created/verified output directory: {args.output}")
        
        # Create map configuration
        config = MapConfig(
            map_name=args.mapname,
            coords=(args.x, args.y, args.size),
            tile_source=args.source,
            utm_zone=args.nad83
        )
        logging.info("Created map configuration")
        
        # Initialize and run map generator
        generator = MapGenerator()
        logging.info("Initialized MapGenerator")
        
        if not generator.validate_config(config):
            raise ValueError("Invalid configuration parameters")
        logging.info("Configuration validated successfully")
            
        # Calculate bounds
        bounds = generator.calculate_bounds(config)
        logging.info(f"Calculated map bounds:")
        logging.info(f"  North: {bounds[0]:.6f}")
        logging.info(f"  West:  {bounds[1]:.6f}")
        logging.info(f"  South: {bounds[2]:.6f}")
        logging.info(f"  East:  {bounds[3]:.6f}")
        
        # Calculate zoom level
        zoom = generator.calculate_optimal_zoom(bounds, config.tile_source)
        logging.info(f"Calculated optimal zoom level: {zoom}")
        
        # Download tiles
        logging.info("Starting tile download...")
        canvas, tile_bounds = await generator.download_tiles(bounds, zoom, config.tile_source)
        logging.info("Tile download complete")
        
        # Save outputs
        jpg_path = os.path.join(args.output, f'{config.map_name}.jpg')
        tif_path = os.path.join(args.output, f'{config.map_name}.tif')
        
        logging.info(f"Saving JPEG to: {jpg_path}")
        canvas.save(jpg_path, quality=85)
        logging.info("JPEG saved successfully")
        
        # Create GeoTIFF
        logging.info(f"Creating GeoTIFF: {tif_path}")
        if config.utm_zone is not None:
            logging.info(f"Using UTM NAD83 zone {config.utm_zone}")
            generator.create_geotiff(jpg_path, tile_bounds, tif_path, zoom, config.tile_source, 
                                  utm_zone=config.utm_zone)
        else:
            logging.info("Using WGS84 coordinates")
            generator.create_geotiff(jpg_path, tile_bounds, tif_path, zoom, config.tile_source)
        
        logging.info("Process completed successfully")
        logging.info("Generated files:")
        logging.info(f"  - JPEG: {jpg_path}")
        logging.info(f"  - GeoTIFF: {tif_path}")

    except Exception as e:
        logging.error(f"Error during map generation: {str(e)}")
        logging.exception("Full error trace:")
        sys.exit(1)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(2)
    except Exception as e:
        print(f"\nCritical error: {str(e)}")
        sys.exit(1)