# Map Generator CLI

A command-line tool for downloading and georeferencing satellite imagery from ArcGIS or Sentinel-2.


![flowchart](https://github.com/[anthonyznova]/[basemapcli]/assets/flowchart.png)

## Features

- **Sources**:  
  ✅ ArcGIS (high-res, WGS84)  
  ✅ Sentinel-2 (10m res, Web Mercator)  
- **Input Formats**:  
  1. Bounding box coordinates (NESW)  
  2. Center point + radius (km)  
- **Output**: GeoTIFF + JPEG with embedded spatial data  
- **Auto-Zoom**: Maximizes resolution within 6×6 tile limit  

## Installation

```bash
pip install -r requirements.txt
