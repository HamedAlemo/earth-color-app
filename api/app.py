from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from planetary_computer import sign
from pystac_client import Client
import stackstac
import xarray as xr
import numpy as np
import datetime
import uvicorn

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Earth Color API is running."}

@app.get("/color")
def get_color(lat: float = Query(...), lon: float = Query(...), year: int = Query(2024)):
    """Return median RGB color for a given location and year from Sentinel-2."""
    # Define date range
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    # Connect to Planetary Computer STAC API
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    # Search for Sentinel-2 L2A
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": 20}}  # Filter low-cloud scenes
    )

    items = list(search.get_items())
    if not items:
        return {"error": "No imagery found for this point and year."}

    # Sign assets
    signed_items = [sign(item) for item in items]

    # Load RGB bands into xarray
    stack = stackstac.stack(
        signed_items,
        assets=["B02", "B03", "B04"],  # Blue, Green, Red
        resolution=10,
        bounds=(lon, lat, lon, lat),
    )

    # Compute median over time
    ds = stack.median(dim="time").compute()

    # Extract RGB
    rgb = [
        float(ds.sel(band="B04").values),
        float(ds.sel(band="B03").values),
        float(ds.sel(band="B02").values),
    ]
    rgb = np.array(rgb) / 10000.0  # Sentinel-2 scale factor
    rgb = np.clip(rgb, 0, 1)

    # Convert to 0-255
    rgb_255 = (rgb * 255).astype(int).tolist()

    # Convert to HEX
    hex_color = "#{:02x}{:02x}{:02x}".format(*rgb_255)

    return {"lat": lat, "lon": lon, "year": year, "rgb": rgb_255, "hex": hex_color}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)