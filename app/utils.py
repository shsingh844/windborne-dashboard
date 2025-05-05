# app/utils.py
import math
from typing import Tuple, Optional

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float, 
                      alt1: Optional[float] = None, alt2: Optional[float] = None) -> float:
    """
    Calculate the distance between two points, optionally in 3D with altitude.
    
    Args:
        lat1, lon1: Coordinates of first point in degrees
        lat2, lon2: Coordinates of second point in degrees
        alt1, alt2: Optional altitude of points in meters
        
    Returns:
        Distance in kilometers
    """
    # Earth radius in kilometers
    earth_radius = 6371.0
    
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula for surface distance
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    surface_distance = earth_radius * c
    
    # If altitude is provided, calculate 3D distance
    if alt1 is not None and alt2 is not None:
        # Convert altitude from meters to kilometers
        alt1_km = alt1 / 1000.0
        alt2_km = alt2 / 1000.0
        
        # Calculate altitude difference
        alt_diff = alt2_km - alt1_km
        
        # Use Pythagorean theorem to find the 3D distance
        return math.sqrt(surface_distance**2 + alt_diff**2)
    
    # Otherwise return surface distance
    return surface_distance

def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the initial bearing (direction) from point 1 to point 2.
    
    Args:
        lat1: Latitude of first point in degrees
        lon1: Longitude of first point in degrees
        lat2: Latitude of second point in degrees
        lon2: Longitude of second point in degrees
        
    Returns:
        Bearing in degrees (0-360)
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Calculate bearing
    dlon = lon2_rad - lon1_rad
    y = math.sin(dlon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    bearing_rad = math.atan2(y, x)
    
    # Convert to degrees
    bearing = (math.degrees(bearing_rad) + 360) % 360
    
    return bearing

def get_direction_from_bearing(bearing: float) -> str:
    """
    Convert a bearing in degrees to a cardinal direction.
    
    Args:
        bearing: Bearing in degrees (0-360)
        
    Returns:
        Cardinal direction as a string
    """
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    index = round(bearing / 45) % 8
    return directions[index]