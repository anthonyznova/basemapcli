# Map Generator CLI

A command-line tool for downloading and georeferencing satellite imagery from ArcGIS or Sentinel-2.

## Features

- **Sources**:  
  ✅ ArcGIS
  ✅ Sentinel-2 
- **Input Formats**:  
  1. Bounding box coordinates (NESW)  
  2. Center point + radius (km)  
- **Output**: GeoTIFF + JPEG  
- **Auto-Zoom**: Maximizes resolution within 6×6 tile limit  

## Usage

Create Config File (input.txt):
Format 1 (Bounding Box):

map_name
N,48.147655,-80.027840
E,48.147555,-79.995609
S,48.134930,-80.028662
W,48.149551,-80.062613
arc

Format 2 (Center Point):

my_map
48.147655,-80.027840,5  # lat, lon, km_radius
sentinel

last line is the imagery source flag

## Workflow

![flowchart](assets/flowchart.png)

## Installation

```bash
pip install -r requirements.txt
