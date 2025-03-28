import json
import logging
import math
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataProcessor:
    """Service to process and analyze balloon data."""
    
    def __init__(self):
        self.earth_radius = 6371.0  # Earth radius in kilometers
    
    def process_balloon_history(self, balloon_history: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Process the balloon history to extract useful metrics and insights.
        
        Args:
            balloon_history: Dictionary mapping balloon ID to its history
            
        Returns:
            Dictionary with processed data and insights
        """
        if not balloon_history:
            logger.warning("No balloon history data to process")
            return {
                "balloons": [],
                "stats": {
                    "total_balloons": 0,
                    "active_balloons": 0
                }
            }
            
        result = {
            "balloons": [],
            "trajectories": {},
            "stats": {},
            "weather_correlations": {},
            "clusters": {}
        }
        
        # Process each balloon
        active_balloons = []
        all_altitudes = []
        all_speeds = []
        
        for balloon_id, history in balloon_history.items():
            if not history:
                continue
                
            # Get the most recent data point for this balloon
            latest = history[-1]
            
            # Calculate movement metrics if we have more than one data point
            metrics = self._calculate_balloon_metrics(history)
            
            # Add processed balloon data
            balloon_data = {
                "id": balloon_id,
                "latest": latest,
                "history_points": len(history),
                **metrics
            }
            
            result["balloons"].append(balloon_data)
            
            # Add trajectory data
            result["trajectories"][balloon_id] = [
                {"lat": point.get("lat"), "lon": point.get("lon"), "alt": point.get("alt", 0), 
                 "timestamp": point.get("timestamp")}
                for point in history if "lat" in point and "lon" in point
            ]
            
            # Collect data for statistics
            if self._is_balloon_active(latest):
                active_balloons.append(balloon_data)
            
            # Collect altitude data
            all_altitudes.extend([point.get("alt", 0) for point in history if "alt" in point])
            
            # Collect speed data from our calculations
            if metrics.get("avg_speed") is not None:
                all_speeds.append(metrics["avg_speed"])
        
        # Calculate overall statistics
        result["stats"] = {
            "total_balloons": len(result["balloons"]),
            "active_balloons": len(active_balloons),
            "altitude_stats": self._calculate_statistics(all_altitudes) if all_altitudes else {},
            "speed_stats": self._calculate_statistics(all_speeds) if all_speeds else {}
        }
        
        # Identify clusters of balloons
        result["clusters"] = self._identify_balloon_clusters([b["latest"] for b in result["balloons"] if "latest" in b])
        
        # Find wind patterns based on balloon movement
        result["wind_patterns"] = self._analyze_wind_patterns(balloon_history)
        
        return result
    
    def _calculate_balloon_metrics(self, history: List[Dict]) -> Dict[str, Any]:
        """
        Calculate metrics for a balloon based on its history.
        
        Args:
            history: List of historical data points for a balloon
            
        Returns:
            Dictionary with calculated metrics
        """
        metrics = {
            "total_distance": 0,
            "avg_speed": None,
            "max_speed": 0,
            "altitude_change": None,
            "direction": None
        }
        
        if len(history) < 2:
            return metrics
            
        # Calculate total distance and speeds
        total_distance = 0
        speeds = []
        
        for i in range(1, len(history)):
            prev = history[i-1]
            curr = history[i]
            
            # Skip if missing required position data
            if not all(k in prev for k in ["lat", "lon"]) or not all(k in curr for k in ["lat", "lon"]):
                continue
                
            # Calculate distance between consecutive points
            distance = self._calculate_distance(
                prev.get("lat"), prev.get("lon"),
                curr.get("lat"), curr.get("lon")
            )
            
            # Time difference in hours (default to 1 hour if timestamp missing)
            time_diff_hours = (curr.get("timestamp", 0) - prev.get("timestamp", 0)) / 3600
            
            if time_diff_hours > 0:
                # Speed in km/h
                speed = distance / time_diff_hours
                speeds.append(speed)
                total_distance += distance
        
        metrics["total_distance"] = round(total_distance, 2)
        
        if speeds:
            metrics["avg_speed"] = round(sum(speeds) / len(speeds), 2)
            metrics["max_speed"] = round(max(speeds), 2)
        
        # Calculate altitude change if data available
        first_alt = next((p.get("alt") for p in history if "alt" in p), None)
        last_alt = next((p.get("alt") for p in reversed(history) if "alt" in p), None)
        
        if first_alt is not None and last_alt is not None:
            metrics["altitude_change"] = round(last_alt - first_alt, 2)
        
        # Calculate overall direction
        if len(history) >= 2 and all(k in history[0] for k in ["lat", "lon"]) and all(k in history[-1] for k in ["lat", "lon"]):
            first = history[0]
            last = history[-1]
            
            bearing = self._calculate_bearing(first["lat"], first["lon"], last["lat"], last["lon"])
            metrics["direction"] = self._get_direction_from_bearing(bearing)
        
        return metrics
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great-circle distance between two points using the Haversine formula.
        
        Args:
            lat1: Latitude of first point in degrees
            lon1: Longitude of first point in degrees
            lat2: Latitude of second point in degrees
            lon2: Longitude of second point in degrees
            
        Returns:
            Distance in kilometers
        """
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = self.earth_radius * c
        
        return distance
    
    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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
    
    def _get_direction_from_bearing(self, bearing: float) -> str:
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
    
    def _is_balloon_active(self, data_point: Dict) -> bool:
        """
        Determine if a balloon is still active based on its latest data.
        
        Args:
            data_point: Latest data point for a balloon
            
        Returns:
            True if the balloon appears active, False otherwise
        """
        # Check if timestamp is recent (within last 2 hours)
        current_time = int(datetime.now().timestamp())
        if "timestamp" in data_point and (current_time - data_point["timestamp"]) > 7200:
            return False
            
        # If balloon has a status field, check it
        if "status" in data_point and data_point["status"].lower() not in ["active", "operational"]:
            return False
            
        return True
    
    def _calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """
        Calculate basic statistics for a list of values.
        
        Args:
            values: List of numerical values
            
        Returns:
            Dictionary with statistics
        """
        if not values:
            return {}
            
        return {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "median": sorted(values)[len(values) // 2]
        }
    
    def _identify_balloon_clusters(self, balloon_positions: List[Dict]) -> List[Dict[str, Any]]:
        """
        Identify clusters of balloons that are close to each other.
        
        Args:
            balloon_positions: List of balloon position data points
            
        Returns:
            List of cluster information
        """
        # Filter positions with valid lat/lon
        valid_positions = [p for p in balloon_positions if "lat" in p and "lon" in p]
        
        if len(valid_positions) < 2:
            return []
            
        clusters = []
        processed = set()
        
        # Simple clustering algorithm based on proximity
        for i, pos1 in enumerate(valid_positions):
            if i in processed:
                continue
                
            cluster = [pos1]
            processed.add(i)
            
            # Find all positions close to this one
            for j, pos2 in enumerate(valid_positions):
                if j in processed:
                    continue
                    
                distance = self._calculate_distance(
                    pos1["lat"], pos1["lon"],
                    pos2["lat"], pos2["lon"]
                )
                
                # If within 100km, consider part of the same cluster
                if distance < 100:
                    cluster.append(pos2)
                    processed.add(j)
            
            # Only consider groups of 2 or more as clusters
            if len(cluster) >= 2:
                # Calculate cluster center
                avg_lat = sum(p["lat"] for p in cluster) / len(cluster)
                avg_lon = sum(p["lon"] for p in cluster) / len(cluster)
                
                clusters.append({
                    "center": {"lat": avg_lat, "lon": avg_lon},
                    "balloons": [p.get("id", "unknown") for p in cluster if "id" in p],
                    "size": len(cluster)
                })
        
        return clusters
    
    def _analyze_wind_patterns(self, balloon_history: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Analyze wind patterns based on balloon movements.
        
        Args:
            balloon_history: Dictionary mapping balloon ID to its history
            
        Returns:
            Dictionary with wind pattern analysis
        """
        # Group balloons by altitude ranges
        altitude_ranges = {
            "low": (0, 5000),      # 0-5km
            "medium": (5000, 15000), # 5-15km
            "high": (15000, float('inf'))  # Above 15km
        }
        
        wind_by_altitude = {key: [] for key in altitude_ranges}
        
        for balloon_id, history in balloon_history.items():
            if len(history) < 2:
                continue
                
            for i in range(1, len(history)):
                prev = history[i-1]
                curr = history[i]
                
                # Skip if missing required data
                if not all(k in prev for k in ["lat", "lon"]) or not all(k in curr for k in ["lat", "lon"]):
                    continue
                
                # Use altitude if available, otherwise skip
                if "alt" not in curr:
                    continue
                    
                altitude = curr["alt"]
                
                # Calculate bearing/direction
                bearing = self._calculate_bearing(
                    prev["lat"], prev["lon"],
                    curr["lat"], curr["lon"]
                )
                
                # Calculate speed
                distance = self._calculate_distance(
                    prev["lat"], prev["lon"],
                    curr["lat"], curr["lon"]
                )
                
                time_diff_hours = (curr.get("timestamp", 0) - prev.get("timestamp", 0)) / 3600
                speed = distance / time_diff_hours if time_diff_hours > 0 else 0
                
                # Add to appropriate altitude range
                for range_key, (min_alt, max_alt) in altitude_ranges.items():
                    if min_alt <= altitude < max_alt:
                        wind_by_altitude[range_key].append({
                            "bearing": bearing,
                            "speed": speed,
                            "lat": curr["lat"],
                            "lon": curr["lon"],
                            "alt": altitude
                        })
                        break
        
        # Calculate average wind direction and speed for each altitude range
        wind_patterns = {}
        
        for range_key, wind_data in wind_by_altitude.items():
            if not wind_data:
                continue
                
            # Calculate average direction using vector components to handle circular nature of bearings
            sin_sum = sum(math.sin(math.radians(w["bearing"])) for w in wind_data)
            cos_sum = sum(math.cos(math.radians(w["bearing"])) for w in wind_data)
            
            avg_direction = math.degrees(math.atan2(sin_sum, cos_sum)) % 360
            avg_speed = sum(w["speed"] for w in wind_data) / len(wind_data)
            
            wind_patterns[range_key] = {
                "avg_direction": avg_direction,
                "avg_direction_cardinal": self._get_direction_from_bearing(avg_direction),
                "avg_speed": avg_speed,
                "sample_size": len(wind_data)
            }
        
        return wind_patterns
    
    def correlate_with_weather_data(self, processed_data: Dict[str, Any], weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Correlate balloon movement with weather data.
        
        Args:
            processed_data: Processed balloon data
            weather_data: Weather data
            
        Returns:
            Dictionary with correlation analysis
        """
        # This would integrate with a weather API to get actual weather data
        # For now, we'll return a placeholder
        return {
            "correlations": "Weather correlation analysis would go here"
        }