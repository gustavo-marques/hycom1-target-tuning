import numpy as np
import xarray as xr

def compute_3d_area(lat, lon, depth):
    """
    Computes the 3D area of a global degree grid repeated over depth levels.

    Parameters:
        lat (array-like): Latitude values (midpoints of cells).
        lon (array-like): Longitude values (midpoints of cells).
        depth (array-like): Standard depth levels.

    Returns:
        xarray.DataArray: 3D area array with dimensions (depth, lat, lon).
    """
    # Constants
    R = 6371000  # Earth's radius in meters
    deg_to_rad = np.pi / 180  # Conversion factor from degrees to radians

    # Compute latitude bounds
    lat_edges = np.append(lat - 0.25, lat[-1] + 0.25)  # Add 0.25 degrees to get edges
    lat_north = lat_edges[1:]  # Northern edge of each cell
    lat_south = lat_edges[:-1]  # Southern edge of each cell

    # Convert to radians
    lat_north_rad = lat_north * deg_to_rad
    lat_south_rad = lat_south * deg_to_rad
    delta_lambda = 0.25 * deg_to_rad  # Longitude width in radians (0.25 degree)

    # Compute 2D area (latitude dependent)
    area_2d = R**2 * delta_lambda * (np.sin(lat_north_rad) - np.sin(lat_south_rad))
    area_2d = np.tile(area_2d[:, np.newaxis], (1, len(lon)))  # Extend across longitude

    # Create an xarray.DataArray for the 2D area
    area_2d_da = xr.DataArray(
        area_2d,
        coords={"lat": lat, "lon": lon},
        dims=["lat", "lon"],
        name="cell_area",
        attrs={"units": "m^2", "description": "Area of each 1x1 degree grid cell"}
    )

    # Repeat the 2D area across the depth dimension
    area_3d_da = area_2d_da.expand_dims(depth=depth).transpose("lat", "lon", "depth")

    # Assign depth dimension metadata
    area_3d_da = area_3d_da.assign_coords(
        depth=("depth", depth.data, {"units": "m", "description": "Depth levels"})
    )
    area_3d_da.name = "3D_area"
    area_3d_da.attrs.update({"description": "3D area of grid cells over depth", "units": "m^2"})

    return area_3d_da

