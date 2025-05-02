import os
import math
import pickle
from datetime import datetime
import requests
import logging
import time
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class WeatherDataService:
    """Service to fetch and process weather data for balloon operations."""
    
    # Open-Meteo API for free weather data
    OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    def __init__(self):
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 10800  # 3 hours instead of 1 hour
        self.cache_file = os.path.join(os.path.dirname(__file__), '../data/weather_cache.pkl')
        self._load_cache()
    
    def _load_cache(self):
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.cache = cache_data.get('cache', {})
                    self.cache_expiry = cache_data.get('expiry', {})
                    logger.debug(f"Loaded {len(self.cache)} cached weather entries")
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            self.cache = {}
            self.cache_expiry = {}

    def _save_cache(self):
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump({
                    'cache': self.cache,
                    'expiry': self.cache_expiry
                }, f)
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")

    def get_weather_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Get weather data for a specific location.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Weather data dictionary
        """
        # Check cache
        cache_key = f"{lat:.2f}_{lon:.2f}"
        current_time = time.time()
        
        if cache_key in self.cache and current_time - self.cache_expiry.get(cache_key, 0) < self.cache_duration:
            logger.info(f"Using cached weather data for {cache_key}")
            return self.cache[cache_key]
        
        # Fetch new data
        try:
            logger.debug(f"Fetching weather data for {lat}, {lon}")
            
            # Build request parameters
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m,winddirection_10m,pressure_msl",
                "windspeed_unit": "kmh",
                "timeformat": "unixtime",
                "forecast_days": 1
            }
            
            # Make request
            response = requests.get(self.OPEN_METEO_BASE_URL, params=params)
            
            if response.status_code != 200:
                logger.error(f"Error fetching weather data: {response.status_code} {response.text}")
                return self._get_fallback_weather_data(lat, lon)
            
            weather_data = response.json()
            
            # Process into a more usable format
            processed_data = self._process_weather_data(weather_data)
            
            # Cache the result
            self.cache[cache_key] = processed_data
            self.cache_expiry[cache_key] = current_time
            self._save_cache()  # Save to file
            
            return processed_data
            
        except Exception as e:
            logger.exception(f"Error fetching weather data: {str(e)}")
            return self._get_fallback_weather_data(lat, lon)
    
    def get_upper_air_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Get upper air data for balloon operations.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Upper air data at different altitudes
        """
        # For upper air data, we'd ideally use a service like NOAA's GFS model
        # Since we're using free APIs, we'll simulate this with reasonable approximations
        # based on surface conditions
        
        # Get surface weather first
        surface_weather = self.get_weather_data(lat, lon)
        
        # Extrapolate to different altitudes
        upper_air_data = self._extrapolate_upper_air_data(surface_weather)
        
        return upper_air_data
    
    def _process_weather_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw weather API data into a more usable format.
        
        Args:
            raw_data: Raw API response
            
        Returns:
            Processed weather data
        """
        processed = {
            "location": {
                "latitude": raw_data.get("latitude"),
                "longitude": raw_data.get("longitude"),
                "elevation": raw_data.get("elevation")
            },
            "current": {},
            "hourly": []
        }
        
        # Process hourly data
        hourly = raw_data.get("hourly", {})
        times = hourly.get("time", [])
        
        for i, timestamp in enumerate(times):
            hour_data = {
                "timestamp": timestamp,
                "datetime": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                "temperature": hourly.get("temperature_2m", [])[i] if i < len(hourly.get("temperature_2m", [])) else None,
                "humidity": hourly.get("relativehumidity_2m", [])[i] if i < len(hourly.get("relativehumidity_2m", [])) else None,
                "wind_speed": hourly.get("windspeed_10m", [])[i] if i < len(hourly.get("windspeed_10m", [])) else None,
                "wind_direction": hourly.get("winddirection_10m", [])[i] if i < len(hourly.get("winddirection_10m", [])) else None,
                "pressure": hourly.get("pressure_msl", [])[i] if i < len(hourly.get("pressure_msl", [])) else None
            }
            
            processed["hourly"].append(hour_data)
        
        # Set current conditions to the latest available hour
        if processed["hourly"]:
            processed["current"] = processed["hourly"][0]
        
        return processed
    
    def _extrapolate_upper_air_data(self, surface_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extrapolate upper air conditions from surface weather data.
        
        Args:
            surface_data: Surface weather data
            
        Returns:
            Estimated conditions at different altitudes
        """
        if not surface_data or "current" not in surface_data:
            return {}
        
        current = surface_data["current"]
        
        # Standard atmospheric models:
        # - Temperature decreases by ~6.5°C per 1000m in troposphere
        # - Wind speed increases with altitude (roughly doubles by 5000m)
        # - Pressure decreases exponentially with altitude
        
        # Define altitude levels (in meters)
        altitude_levels = [
            1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000
        ]
        
        upper_air = {
            "altitude_levels": altitude_levels,
            "data": []
        }
        
        surface_temp = current.get("temperature", 15)  # °C
        surface_pressure = current.get("pressure", 1013.25)  # hPa
        surface_wind_speed = current.get("wind_speed", 10)  # km/h
        surface_wind_dir = current.get("wind_direction", 0)  # degrees
        
        for altitude in altitude_levels:
            # Temperature: Standard lapse rate of 6.5°C/km
            temp = surface_temp - (altitude / 1000 * 6.5)
            
            # Pressure: Barometric formula (simplified)
            # P = P0 * exp(-h/8000)
            pressure = surface_pressure * 100 * math.exp(-altitude / 8000) / 100  # Convert to hPa
            
            # Wind speed: Increases with altitude (simplified model)
            # At 5km, wind speed is typically 2-3x surface
            wind_multiplier = 1 + (altitude / 5000) * 2
            wind_speed = surface_wind_speed * wind_multiplier
            
            # Wind direction: Turns clockwise with height due to Coriolis effect
            # This is a very simplified model
            wind_dir_shift = min(45, altitude / 1000 * 3)  # Max 45 degree shift
            wind_dir = (surface_wind_dir + wind_dir_shift) % 360
            
            upper_air["data"].append({
                "altitude": altitude,
                "temperature": temp,
                "pressure": pressure,
                "wind_speed": wind_speed,
                "wind_direction": wind_dir,
                "density": self._calculate_air_density(temp, pressure)
            })
        
        return upper_air
    
    def _calculate_air_density(self, temperature: float, pressure: float) -> float:
        """
        Calculate air density from temperature and pressure.
        
        Args:
            temperature: Temperature in °C
            pressure: Pressure in hPa
            
        Returns:
            Air density in kg/m³
        """
        # Convert to SI units
        temp_kelvin = temperature + 273.15
        pressure_pa = pressure * 100
        
        # Gas constant for dry air
        R = 287.058  # J/(kg·K)
        
        # Density = P / (R * T)
        density = pressure_pa / (R * temp_kelvin)
        
        return density
    
    def _get_fallback_weather_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Generate fallback weather data when API fails.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Fallback weather data
        """
        # Get current date and time
        now = datetime.now()
        timestamp = int(now.timestamp())
        
        # Generate reasonable defaults based on latitude
        temp = 15  # Default 15°C
        
        # Adjust for latitude (colder toward poles)
        temp -= abs(lat) / 90 * 30
        
        # Adjust for season in appropriate hemisphere
        month = now.month
        if (lat > 0 and 5 <= month <= 9) or (lat < 0 and (month <= 3 or month >= 10)):
            # Summer
            temp += 10
        elif (lat > 0 and (month <= 3 or month >= 10)) or (lat < 0 and 5 <= month <= 9):
            # Winter
            temp -= 10
        
        # Generate fallback data
        return {
            "location": {
                "latitude": lat,
                "longitude": lon,
                "elevation": 0
            },
            "current": {
                "timestamp": timestamp,
                "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
                "temperature": temp,
                "humidity": 50,
                "wind_speed": 15,
                "wind_direction": (45 * (int(lat) % 8)) % 360,  # Deterministic but varied
                "pressure": 1013.25
            },
            "hourly": [
                {
                    "timestamp": timestamp,
                    "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "temperature": temp,
                    "humidity": 50,
                    "wind_speed": 15,
                    "wind_direction": (45 * (int(lat) % 8)) % 360,
                    "pressure": 1013.25
                }
            ],
            "is_fallback": True
        }
    
    def fetch_weather_for_balloon_grid(self, balloons: List[Dict]) -> Dict[str, Any]:
        """
        Fetch weather data for a grid of points around balloon positions.
        
        Args:
            balloons: List of balloon objects with positions
            
        Returns:
            Grid of weather data
        """
        # Extract unique positions (rounded to reduce API calls)
        positions = set()
        for balloon in balloons:
            if "latest" in balloon and "lat" in balloon["latest"] and "lon" in balloon["latest"]:
                lat = round(balloon["latest"]["lat"] / 10) * 10  # Round to nearest 10 degrees
                lon = round(balloon["latest"]["lon"] / 10) * 10
                positions.add((lat, lon))
        
        # Fetch weather for each position
        weather_grid = {}
        for lat, lon in positions:
            key = f"{lat},{lon}"
            weather_grid[key] = self.get_weather_data(lat, lon)
        
        return {
            "grid_points": len(weather_grid),
            "data": weather_grid
        }