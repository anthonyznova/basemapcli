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

## Workflow

![flowchart](assets/flowchart.png)

## Installation

```bash
pip install -r requirements.txt
