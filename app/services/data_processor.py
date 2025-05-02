import json
import logging
import math
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import requests

logger = logging.getLogger(__name__)

class DataProcessor:
    """Enhanced service to process, analyze, and predict balloon data."""
    
    def __init__(self):
        self.earth_radius = 6371.0  # Earth radius in kilometers
        # Define standard atmospheric layers
        self.atmosphere_layers = {
            "low": (0, 5000),  # 0-5km
            "medium": (5000, 15000),  # 5-15km
            "high": (15000, 30000)  # 15-30km
        }
        # Historical average wind speeds by altitude (m/s)
        # These are approximate global averages - would ideally be replaced with real historical data
        self.historical_wind_speeds = {
            "low": {
                "winter": 8.5,
                "spring": 7.2,
                "summer": 5.8,
                "fall": 7.0
            },
            "medium": {
                "winter": 20.5,
                "spring": 18.2,
                "summer": 15.8,
                "fall": 17.0
            },
            "high": {
                "winter": 35.5,
                "spring": 30.2,
                "summer": 25.8,
                "fall": 30.0
            }
        }
    
    def process_balloon_history(self, balloon_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the balloon history to extract metrics, insights, and predictions.
        
        Args:
            balloon_data: Dictionary with balloon history and errors
            
        Returns:
            Dictionary with processed data, insights, and predictions
        """
        # Extract data from the input
        balloon_history = balloon_data.get('balloon_history', {})
        error_records = balloon_data.get('errors', [])
        
        # Debug log
        logger.info(f"Processing balloon history with {len(balloon_history)} balloons and {len(error_records) if error_records else 0} errors")
        
        # Count missing hours (404 errors)
        missing_hours = 0
        
        if error_records:
            for error in error_records:
                status = error.get('status', '')
                if status == 'missing':
                    missing_hours += 1
        
        if not balloon_history:
            logger.warning("No balloon history data to process")
            
            # Return minimal structure with error data
            return {
                "balloons": [],
                "stats": {
                    "total_balloons": 0,
                    "active_balloons": 0
                },
                "errors": error_records,
                "data_quality": {
                    "missing_hours": missing_hours,
                    "total_hours": 24,
                    "available_hours": 24 - missing_hours
                }
            }
        
        # Initialize result structure with enhanced sections
        result = {
            "balloons": [],
            "trajectories": {},
            "stats": {},
            "weather_correlations": {},
            "clusters": {},
            "errors": error_records,
            "data_quality": {
                "missing_hours": missing_hours,
                "total_hours": 24,
                "available_hours": 24 - missing_hours
            },
            "predictions": {
                "trajectory_forecasts": {},
                "optimal_launch_sites": [],
                "weather_advisories": []
            },
            "atmospheric_anomalies": [],
            "performance_analytics": {
                "optimal_altitude_bands": [],
                "efficiency_metrics": {}
            }
        }
        
        # Process each balloon
        active_balloons = []
        all_altitudes = []
        all_speeds = []
        
        for balloon_id, history in balloon_history.items():
            if not history:
                logger.warning(f"Balloon {balloon_id} has empty history")
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
        
        # Process global statistics
        result["stats"] = {
            "total_balloons": len(result["balloons"]),
            "active_balloons": len(active_balloons),
            "altitude_stats": self._calculate_statistics(all_altitudes) if all_altitudes else {},
            "speed_stats": self._calculate_statistics(all_speeds) if all_speeds else {}
        }
        
        # Generate wind patterns based on balloon movement
        result["wind_patterns"] = self._analyze_wind_patterns(balloon_history)
        
        # Identify clusters of balloons
        result["clusters"] = self._identify_balloon_clusters([b["latest"] for b in result["balloons"] if "latest" in b])
        
        # Generate trajectory predictions (NEW)
        result["predictions"]["trajectory_forecasts"] = self._predict_balloon_trajectories(result["balloons"], result["wind_patterns"])
        
        # Calculate optimal launch sites (NEW)
        result["predictions"]["optimal_launch_sites"] = self._calculate_optimal_launch_sites(result["wind_patterns"])
        
        # Detect atmospheric anomalies (NEW)
        result["atmospheric_anomalies"] = self._detect_atmospheric_anomalies(result["wind_patterns"])
        
        # Calculate optimal altitude bands (NEW)
        result["performance_analytics"]["optimal_altitude_bands"] = self._determine_optimal_altitude_bands(
            result["balloons"], result["wind_patterns"]
        )
        
        # Calculate efficiency metrics (NEW)
        result["performance_analytics"]["efficiency_metrics"] = self._calculate_efficiency_metrics(
            result["balloons"], result["wind_patterns"]
        )
        
        # Sanitize data to ensure all values are JSON-serializable
        sanitized_result = self.sanitize_json_data(result)
        
        return sanitized_result
    
    def _calculate_balloon_metrics(self, history: List[Dict]) -> Dict[str, Any]:
        """Calculate metrics for a balloon based on its history."""
        metrics = {
            "total_distance": 0,
            "avg_speed": 0.0,  # Default to 0 instead of None
            "max_speed": 0,
            "altitude_change": None,
            "direction": None
        }
        
        if len(history) < 2:
            # Even with just one data point, set default values
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
            time_diff_hours = 1.0  # Default time difference
            if "timestamp" in curr and "timestamp" in prev:
                time_diff_secs = curr.get("timestamp", 0) - prev.get("timestamp", 0)
                time_diff_hours = time_diff_secs / 3600
                if time_diff_hours <= 0:
                    time_diff_hours = 1.0  # Default to 1 hour if invalid time difference
            
            # Speed in km/h
            speed = distance / time_diff_hours
            speeds.append(speed)
            total_distance += distance
        
        metrics["total_distance"] = round(total_distance, 2)
        
        # Always provide speed values, even if estimated
        if speeds:
            metrics["avg_speed"] = round(sum(speeds) / len(speeds), 2)
            metrics["max_speed"] = round(max(speeds), 2)
        else:
            # Estimate speed based on total distance
            if total_distance > 0:
                # Assume the journey took place over hours represented by history points
                estimated_hours = len(history) - 1 or 1  # Avoid division by zero
                metrics["avg_speed"] = round(total_distance / estimated_hours, 2)
                metrics["max_speed"] = round(metrics["avg_speed"] * 1.5, 2)  # Estimate max as 1.5x avg
        
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
            metrics["bearing"] = bearing  # Store the actual bearing value
        
        return metrics
    
    def _is_balloon_active(self, data_point: Dict) -> bool:
        """Determine if a balloon is still active based on its latest data."""
        # Check if timestamp is recent (within last 2 hours)
        current_time = int(datetime.now().timestamp())
        if "timestamp" in data_point and (current_time - data_point["timestamp"]) > 7200:
            return False
            
        # If balloon has a status field, check it
        if "status" in data_point and data_point["status"].lower() not in ["active", "operational"]:
            return False
            
        return True
    
    def _calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate basic statistics for a list of values."""
        if not values:
            return {}
        
        # Filter out NaN and infinity values
        filtered_values = [v for v in values if not (math.isnan(v) or math.isinf(v))]
        
        if not filtered_values:
            return {
                "min": 0.0,
                "max": 0.0,
                "avg": 0.0,
                "median": 0.0
            }
        
        # Calculate statistics, replacing any NaN or infinity values with 0
        stats = {
            "min": min(filtered_values),
            "max": max(filtered_values),
            "avg": sum(filtered_values) / len(filtered_values),
            "median": sorted(filtered_values)[len(filtered_values) // 2]
        }
        
        # Ensure all values are JSON-serializable
        for key, value in stats.items():
            if isinstance(value, (float, np.float32, np.float64)):
                if math.isnan(value) or math.isinf(value):
                    stats[key] = 0.0
            
        return stats
    
    # NEW METHOD: Predict future balloon trajectories
    def _predict_balloon_trajectories(self, balloons: List[Dict], wind_patterns: Dict) -> Dict[str, List[Dict]]:
        """
        Predict balloon trajectories for the next 24 hours based on current positions and wind patterns.
        
        Args:
            balloons: List of balloon objects with current positions and metrics
            wind_patterns: Dictionary of wind patterns by altitude
            
        Returns:
            Dictionary mapping balloon IDs to their predicted future positions
        """
        trajectory_forecasts = {}
        
        # Hours to predict (6, 12, 18, 24)
        forecast_hours = [6, 12, 18, 24]
        
        for balloon in balloons:
            if not balloon.get("latest") or "lat" not in balloon["latest"] or "lon" not in balloon["latest"]:
                continue
                
            balloon_id = balloon["id"]
            current_lat = balloon["latest"]["lat"]
            current_lon = balloon["latest"]["lon"]
            current_alt = balloon["latest"].get("alt", 10000)  # Default to 10km if no altitude
            
            # Determine which altitude layer this balloon is in
            altitude_layer = "medium"  # Default to medium
            if current_alt < 5000:
                altitude_layer = "low"
            elif current_alt >= 15000:
                altitude_layer = "high"
            
            # Get wind direction and speed for this altitude
            wind_data = wind_patterns.get(altitude_layer, {})
            if not wind_data or "avg_direction" not in wind_data or "avg_speed" not in wind_data:
                continue
                
            wind_direction = wind_data["avg_direction"]
            wind_speed = wind_data["avg_speed"]
            
            # Calculate predicted positions
            predicted_positions = []
            
            for hours_ahead in forecast_hours:
                # Calculate distance traveled (wind speed * time)
                distance_km = wind_speed * hours_ahead
                
                # Calculate new position based on wind direction and speed
                new_position = self._calculate_new_position(
                    current_lat, current_lon, wind_direction, distance_km
                )
                
                predicted_positions.append({
                    "hours_ahead": hours_ahead,
                    "lat": new_position[0],
                    "lon": new_position[1],
                    "alt": current_alt,  # Assume altitude stays constant for simple prediction
                    "confidence": self._calculate_prediction_confidence(hours_ahead)
                })
            
            trajectory_forecasts[balloon_id] = predicted_positions
        
        return trajectory_forecasts
    
    def _calculate_new_position(self, lat: float, lon: float, bearing: float, distance_km: float) -> Tuple[float, float]:
        """
        Calculate a new position given a starting point, bearing, and distance.
        
        Args:
            lat: Starting latitude in degrees
            lon: Starting longitude in degrees
            bearing: Direction of travel in degrees (0 = North, 90 = East)
            distance_km: Distance to travel in kilometers
            
        Returns:
            Tuple of (new_latitude, new_longitude) in degrees
        """
        # Convert to radians
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        bearing_rad = math.radians(bearing)
        
        # Angular distance
        angular_distance = distance_km / self.earth_radius
        
        # Calculate new latitude
        new_lat_rad = math.asin(
            math.sin(lat_rad) * math.cos(angular_distance) + 
            math.cos(lat_rad) * math.sin(angular_distance) * math.cos(bearing_rad)
        )
        
        # Calculate new longitude
        new_lon_rad = lon_rad + math.atan2(
            math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(lat_rad),
            math.cos(angular_distance) - math.sin(lat_rad) * math.sin(new_lat_rad)
        )
        
        # Convert back to degrees
        new_lat = math.degrees(new_lat_rad)
        new_lon = math.degrees(new_lon_rad) % 360
        if new_lon > 180:
            new_lon -= 360
        
        return (new_lat, new_lon)
    
    def _calculate_prediction_confidence(self, hours_ahead: int) -> float:
        """
        Calculate confidence level for a prediction based on multiple factors.
        
        Args:
            hours_ahead: Number of hours in the future
            
        Returns:
            Confidence level from 0.0 to 1.0
        """
        # Base confidence calculation with time decay
        base_confidence = max(0.0, min(1.0, 1.0 - (hours_ahead / 30.0)))
        
        # Add some randomization to vary between balloons (±15%)
        # Using a deterministic approach based on hours_ahead
        variation = ((hours_ahead * 17) % 30 - 15) / 100.0
        
        # Ensure confidence stays between 0.05 and 0.95
        adjusted_confidence = max(0.05, min(0.95, base_confidence + variation))
        
        return adjusted_confidence
    
    # NEW METHOD: Calculate optimal launch sites
    def _calculate_optimal_launch_sites(self, wind_patterns: Dict) -> List[Dict]:
        """
        Calculate optimal balloon launch sites based on current wind patterns.
        
        Args:
            wind_patterns: Dictionary of wind patterns by altitude
            
        Returns:
            List of optimal launch site recommendations
        """
        # Define potential launch regions
        potential_regions = [
            {"name": "North America", "lat": 40, "lon": -100},
            {"name": "South America", "lat": -20, "lon": -60},
            {"name": "Europe", "lat": 50, "lon": 10},
            {"name": "Africa", "lat": 0, "lon": 20},
            {"name": "Asia", "lat": 30, "lon": 100},
            {"name": "Australia", "lat": -25, "lon": 135},
            {"name": "Pacific", "lat": 0, "lon": -170}
        ]
        
        optimal_sites = []
        
        # Get current season for winds (simplified)
        current_month = datetime.now().month
        season = "winter"
        if 3 <= current_month <= 5:
            season = "spring"
        elif 6 <= current_month <= 8:
            season = "summer"
        elif 9 <= current_month <= 11:
            season = "fall"
        
        # Calculate launch scores for each region
        for region in potential_regions:
            # Initial score - higher is better
            score = 50
            
            # Adjust score based on wind conditions
            for altitude_layer, wind_data in wind_patterns.items():
                if not wind_data or "avg_speed" not in wind_data:
                    continue
                
                wind_speed = wind_data["avg_speed"]
                
                # Compare with historical averages
                historical_avg = self.historical_wind_speeds.get(altitude_layer, {}).get(season, 0)
                
                # Score based on wind stability (closer to historical average is better)
                wind_stability_score = 100 - min(100, abs(wind_speed - historical_avg) * 5)
                
                # Moderate winds (10-30 km/h) are ideal for balloon launches
                if 10 <= wind_speed <= 30:
                    score += 20
                # Very low or very high winds are problematic
                elif wind_speed < 5 or wind_speed > 50:
                    score -= 20
                
                # Weight by altitude layer importance
                if altitude_layer == "low":
                    score += wind_stability_score * 0.5  # Low altitude most important for launch
                elif altitude_layer == "medium":
                    score += wind_stability_score * 0.3
                else:
                    score += wind_stability_score * 0.2
            
            # Create recommendation if score is good
            if score >= 60:
                recommendation = {
                    "region": region["name"],
                    "coordinates": {"lat": region["lat"], "lon": region["lon"]},
                    "score": min(100, max(0, score)),
                    "rationale": self._generate_launch_site_rationale(region["name"], score, wind_patterns)
                }
                optimal_sites.append(recommendation)
        
        # Sort by score (descending)
        optimal_sites.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top 3 sites or all if fewer
        return optimal_sites[:3]
    
    def _generate_launch_site_rationale(self, region: str, score: float, wind_patterns: Dict) -> str:
        """
        Generate a rationale for why a launch site is recommended.
        
        Args:
            region: Name of the region
            score: Recommendation score
            wind_patterns: Wind pattern data
            
        Returns:
            Explanation string
        """
        if score >= 90:
            return f"Excellent conditions in {region} with stable winds at all altitudes"
        elif score >= 80:
            return f"Very good conditions in {region}, particularly at launch altitudes"
        elif score >= 70:
            return f"Good conditions in {region}, though some altitude ranges have suboptimal winds"
        else:
            return f"Acceptable conditions in {region}, but monitor wind changes before launch"
    
    # NEW METHOD: Detect atmospheric anomalies
    def _detect_atmospheric_anomalies(self, wind_patterns: Dict) -> List[Dict]:
        """
        Detect anomalies in atmospheric conditions compared to historical averages.
        
        Args:
            wind_patterns: Dictionary of wind patterns by altitude
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Get current season
        current_month = datetime.now().month
        season = "winter"
        if 3 <= current_month <= 5:
            season = "spring"
        elif 6 <= current_month <= 8:
            season = "summer"
        elif 9 <= current_month <= 11:
            season = "fall"
        
        # Check each altitude layer for anomalies
        for altitude_layer, wind_data in wind_patterns.items():
            if not wind_data or "avg_speed" not in wind_data or "avg_direction" not in wind_data:
                continue
            
            current_speed = wind_data["avg_speed"]
            current_direction = wind_data["avg_direction"]
            
            # Get historical average speed for this altitude and season
            historical_speed = self.historical_wind_speeds.get(altitude_layer, {}).get(season, 0)
            
            # Calculate percentage difference
            if historical_speed > 0:
                speed_diff_percent = ((current_speed - historical_speed) / historical_speed) * 100
            else:
                speed_diff_percent = 0
            
            # Detect significant anomalies (>30% difference)
            if abs(speed_diff_percent) > 30:
                anomaly = {
                    "type": "wind_speed",
                    "altitude_layer": altitude_layer,
                    "current_value": current_speed,
                    "historical_value": historical_speed,
                    "difference_percent": speed_diff_percent,
                    "severity": "high" if abs(speed_diff_percent) > 50 else "medium",
                    "description": self._generate_anomaly_description(
                        "wind_speed", altitude_layer, current_speed, historical_speed, speed_diff_percent
                    )
                }
                anomalies.append(anomaly)
            
            # In a more complete implementation, we would also check direction anomalies
            # and other atmospheric parameters like temperature, pressure, etc.
        
        return anomalies
    
    def _generate_anomaly_description(self, anomaly_type: str, altitude_layer: str, 
                                     current_value: float, historical_value: float, 
                                     diff_percent: float) -> str:
        """
        Generate a human-readable description of an atmospheric anomaly.
        
        Args:
            anomaly_type: Type of anomaly (wind_speed, direction, etc.)
            altitude_layer: Altitude layer (low, medium, high)
            current_value: Current observed value
            historical_value: Historical average value
            diff_percent: Percentage difference
            
        Returns:
            Description string
        """
        altitude_desc = {
            "low": "low altitudes (0-5km)",
            "medium": "medium altitudes (5-15km)",
            "high": "high altitudes (15km+)"
        }.get(altitude_layer, altitude_layer)
        
        if anomaly_type == "wind_speed":
            if diff_percent > 0:
                return f"Wind speeds at {altitude_desc} are {abs(diff_percent):.1f}% higher than seasonal average ({current_value:.1f} vs {historical_value:.1f} km/h)"
            else:
                return f"Wind speeds at {altitude_desc} are {abs(diff_percent):.1f}% lower than seasonal average ({current_value:.1f} vs {historical_value:.1f} km/h)"
        
        # Default case
        return f"Anomaly detected at {altitude_desc}"
    
    # NEW METHOD: Determine optimal altitude bands
    def _determine_optimal_altitude_bands(self, balloons: List[Dict], wind_patterns: Dict) -> List[Dict]:
        """
        Determine optimal altitude bands for balloon operation based on performance data.
        
        Args:
            balloons: List of balloon objects with metrics
            wind_patterns: Dictionary of wind patterns by altitude
            
        Returns:
            List of optimal altitude bands with explanations
        """
        # Group balloons by altitude bands
        altitude_bands = {
            "0-3000m": [],
            "3000-6000m": [],
            "6000-10000m": [],
            "10000-15000m": [],
            "15000-20000m": [],
            "above-20000m": []
        }
        
        # Group performance metrics by altitude band
        for balloon in balloons:
            if not balloon.get("latest") or "alt" not in balloon["latest"]:
                continue
                
            altitude = balloon["latest"]["alt"]
            
            # Determine which band this balloon belongs to
            band_key = "above-20000m"
            if altitude < 3000:
                band_key = "0-3000m"
            elif altitude < 6000:
                band_key = "3000-6000m"
            elif altitude < 10000:
                band_key = "6000-10000m"
            elif altitude < 15000:
                band_key = "10000-15000m"
            elif altitude < 20000:
                band_key = "15000-20000m"
            
            # Add to the band with speed data if available
            if "avg_speed" in balloon:
                altitude_bands[band_key].append({
                    "id": balloon["id"],
                    "alt": altitude,
                    "avg_speed": balloon["avg_speed"],
                    "total_distance": balloon.get("total_distance", 0)
                })
        
        # Calculate metrics for each band
        band_metrics = {}
        for band, balloons in altitude_bands.items():
            if not balloons:
                continue
                
            avg_speed = sum(b["avg_speed"] for b in balloons) / len(balloons)
            avg_distance = sum(b["total_distance"] for b in balloons) / len(balloons)
            
            # Calculate efficiency score (speed and distance both matter)
            efficiency_score = (avg_speed / 20) * 50 + (avg_distance / 100) * 50
            
            band_metrics[band] = {
                "count": len(balloons),
                "avg_speed": avg_speed,
                "avg_distance": avg_distance,
                "efficiency_score": efficiency_score
            }
        
        # Find bands with highest efficiency scores
        sorted_bands = sorted(
            [{"band": band, **metrics} for band, metrics in band_metrics.items()],
            key=lambda x: x["efficiency_score"],
            reverse=True
        )
        
        # Generate recommendations
        optimal_bands = []
        for band_data in sorted_bands[:3]:  # Top 3 bands
            if band_data["count"] < 2:  # Need at least 2 balloons for statistical relevance
                continue
                
            band = band_data["band"]
            recommendation = {
                "altitude_band": band,
                "efficiency_score": band_data["efficiency_score"],
                "avg_speed": band_data["avg_speed"],
                "avg_distance": band_data["avg_distance"],
                "balloon_count": band_data["count"],
                "recommendation": self._generate_altitude_recommendation(band, band_data)
            }
            optimal_bands.append(recommendation)
        
        return optimal_bands
    
    def _generate_altitude_recommendation(self, band: str, metrics: Dict) -> str:
        """
        Generate a recommendation for an altitude band.
        
        Args:
            band: Altitude band name
            metrics: Performance metrics for the band
            
        Returns:
            Recommendation string
        """
        if metrics["efficiency_score"] > 80:
            return f"Excellent performance in the {band} range with high speeds and optimal trajectory stability"
        elif metrics["efficiency_score"] > 60:
            return f"Good performance in the {band} range, suitable for most mission profiles"
        else:
            return f"Moderate performance in the {band} range, may be suitable for specific use cases"
    
    # NEW METHOD: Calculate efficiency metrics
    def _calculate_efficiency_metrics(self, balloons: List[Dict], wind_patterns: Dict) -> Dict[str, Any]:
        """
        Calculate efficiency metrics for the balloon constellation.
        
        Args:
            balloons: List of balloon objects with metrics
            wind_patterns: Dictionary of wind patterns by altitude
            
        Returns:
            Dictionary with efficiency metrics
        """
        if not balloons:
            return {}
            
        # Calculate distance traveled per hour metrics
        speeds = [b.get("avg_speed", 0) for b in balloons if "avg_speed" in b]
        if not speeds:
            speeds = [0]
        
        # Calculate energy efficiency proxies
        # We don't have actual energy data, so we're using altitude and speed as proxies
        altitude_efficiency = []
        for balloon in balloons:
            if not balloon.get("latest") or "alt" not in balloon["latest"]:
                continue
                
            altitude = balloon["latest"]["alt"]
            avg_speed = balloon.get("avg_speed", 0)
            
            if avg_speed > 0:
                # Higher ratio of speed to altitude indicates more efficient use of altitude
                # (getting more horizontal movement from less vertical investment)
                altitude_efficiency.append(avg_speed / max(1000, altitude))
        
        if not altitude_efficiency:
            altitude_efficiency = [0]
        
        # Calculate coverage metrics (area covered per balloon)
        coverage_area = 0
        if len(balloons) > 1:
            # Simple convex hull approximation - in a real implementation, 
            # you'd calculate an actual convex hull
            lat_min = min(b["latest"]["lat"] for b in balloons if "latest" in b and "lat" in b["latest"])
            lat_max = max(b["latest"]["lat"] for b in balloons if "latest" in b and "lat" in b["latest"])
            lon_min = min(b["latest"]["lon"] for b in balloons if "latest" in b and "lon" in b["latest"])
            lon_max = max(b["latest"]["lon"] for b in balloons if "latest" in b and "lon" in b["latest"])
            
            # Very rough area calculation in square km
            width = self._calculate_distance(lat_min, lon_min, lat_min, lon_max)
            height = self._calculate_distance(lat_min, lon_min, lat_max, lon_min)
            coverage_area = width * height
        
        # Build the metrics dictionary
        return {
            "avg_speed": sum(speeds) / len(speeds),
            "speed_efficiency": {
                "min": min(speeds),
                "max": max(speeds),
                "avg": sum(speeds) / len(speeds)
            },
            "altitude_efficiency": {
                "min": min(altitude_efficiency),
                "max": max(altitude_efficiency),
                "avg": sum(altitude_efficiency) / len(altitude_efficiency)
            },
            "coverage": {
                "total_area_km2": coverage_area,
                "area_per_balloon": coverage_area / max(1, len(balloons))
            },
            "operational_metrics": {
                "active_percentage": (sum(1 for b in balloons if self._is_balloon_active(b.get("latest", {}))) / max(1, len(balloons))) * 100,
                "avg_altitude": sum(b["latest"].get("alt", 0) for b in balloons if "latest" in b and "alt" in b["latest"]) / 
                               max(1, sum(1 for b in balloons if "latest" in b and "alt" in b["latest"]))
            }
        }
        
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
        # Earth radius in kilometers
        earth_radius = 6371.0
        
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
        distance = earth_radius * c
        
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
        
        # Count balloons by altitude range for fallback
        balloons_by_altitude = {key: [] for key in altitude_ranges}
        
        # First, collect all balloons by altitude range
        for balloon_id, history in balloon_history.items():
            if not history:
                continue
                
            # Use the latest point for classification
            latest = history[-1]
            if "alt" not in latest:
                continue
                
            altitude = latest["alt"]
            
            # Add to appropriate altitude range
            for range_key, (min_alt, max_alt) in altitude_ranges.items():
                if min_alt <= altitude < max_alt:
                    balloons_by_altitude[range_key].append(balloon_id)
                    break
        
        # Now process movements
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
                if time_diff_hours <= 0:
                    time_diff_hours = 1  # Default to 1 hour if invalid time difference
                    
                speed = distance / time_diff_hours
                
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
                # Fallback: If we have balloons in this range but no movement data,
                # generate a placeholder pattern based on averages
                if balloons_by_altitude[range_key]:
                    # Create placeholder wind pattern with reasonable defaults
                    sample_size = len(balloons_by_altitude[range_key])
                    avg_direction = 45.0 * (hash(range_key) % 8)  # Deterministic but varied direction
                    wind_patterns[range_key] = {
                        "avg_direction": avg_direction,
                        "avg_direction_cardinal": self._get_direction_from_bearing(avg_direction),
                        "avg_speed": 20.0 + 10.0 * (hash(range_key) % 5),  # Random-ish speed between 20-70
                        "sample_size": sample_size,
                        "estimated": True  # Mark as estimated data
                    }
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
                "sample_size": len(wind_data),
                "estimated": False
            }
        
        # Ensure we have at least one pattern
        if not wind_patterns:
            # Generate some placeholder data for visualization
            wind_patterns["medium"] = {
                "avg_direction": 45.0,
                "avg_direction_cardinal": "NE",
                "avg_speed": 35.0,
                "sample_size": 1,
                "estimated": True
            }
        
        return wind_patterns
    
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
    
    def sanitize_json_data(self, data):
        """
        Recursively sanitize a data structure to ensure all values are JSON-serializable.
        Replaces NaN, infinity, and other problematic values.
        
        Args:
            data: Any data structure (dict, list, etc.)
            
        Returns:
            Sanitized data structure
        """
        if isinstance(data, dict):
            return {k: self.sanitize_json_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_json_data(item) for item in data]
        elif isinstance(data, (float, np.float32, np.float64)):
            # Replace NaN and infinity with 0.0
            if math.isnan(data) or math.isinf(data):
                return 0.0
            return float(data)  # Convert numpy types to standard float
        elif isinstance(data, (np.int32, np.int64)):
            return int(data)  # Convert numpy types to standard int
        else:
            return data