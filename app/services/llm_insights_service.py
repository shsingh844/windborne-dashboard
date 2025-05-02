import anthropic
import json
import logging
import time
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class LLMInsightsService:
    """Service to generate insights using Anthropic Claude API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = None
        
        # Cache for insights to avoid excessive API calls
        self.insights_cache = {}
        self.cache_duration = 3600  # 1 hour cache
        
        if api_key:
            self.initialize_client(api_key)
    
    def initialize_client(self, api_key: str) -> bool:
        """
        Initialize the Anthropic client with the provided API key.
        
        Args:
            api_key: Anthropic API key
            
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            self.api_key = api_key
            self.client = anthropic.Anthropic(api_key=api_key)
            # Make a simple call to verify the key works
            self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return True
        except Exception as e:
            logger.error(f"Error initializing Anthropic client: {str(e)}")
            self.client = None
            return False
    
    def has_valid_api_key(self) -> bool:
        """Check if the service has a valid API key."""
        return self.client is not None
    
    def generate_balloon_insights(self, balloon_data: Dict[str, Any], 
                                 weather_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate insights by analyzing balloon and weather data using Claude.
        
        Args:
            balloon_data: Processed balloon data
            weather_data: Optional weather data
            
        Returns:
            Dictionary with generated insights
        """
        if not self.has_valid_api_key():
            return {
                "error": "No valid API key",
                "message": "Please provide a valid Anthropic API key to generate insights"
            }
        
        # Check cache
        cache_key = f"insights_{hash(json.dumps(balloon_data, sort_keys=True))}"
        current_time = time.time()
        
        if cache_key in self.insights_cache and current_time - self.insights_cache[cache_key].get("timestamp", 0) < self.cache_duration:
            logger.info(f"Using cached insights for {cache_key}")
            return self.insights_cache[cache_key]["data"]
        
        # Prepare context with key metrics
        context = self._prepare_context(balloon_data, weather_data)
        
        # Generate insights
        try:
            insights = self._call_llm_with_context(context)
            
            # Cache the results
            self.insights_cache[cache_key] = {
                "timestamp": current_time,
                "data": insights
            }
            
            return insights
            
        except Exception as e:
            logger.exception(f"Error generating insights: {str(e)}")
            return {
                "error": "Failed to generate insights",
                "message": str(e)
            }
    
    def _prepare_context(self, balloon_data: Dict[str, Any], 
                         weather_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Prepare a context dictionary with key metrics for the LLM.
        
        Args:
            balloon_data: Processed balloon data
            weather_data: Optional weather data
            
        Returns:
            Context dictionary
        """
        # Extract key metrics from balloon data
        context = {
            "balloon_count": len(balloon_data.get("balloons", [])),
            "active_balloons": balloon_data.get("stats", {}).get("active_balloons", 0),
            "altitude_data": {}
        }
        
        # Extract altitude statistics
        altitude_stats = balloon_data.get("stats", {}).get("altitude_stats", {})
        context["altitude_data"] = {
            "min": altitude_stats.get("min", 0),
            "max": altitude_stats.get("max", 0),
            "avg": altitude_stats.get("avg", 0)
        }
        
        # Extract wind patterns
        context["wind_patterns"] = {}
        for alt_layer, data in balloon_data.get("wind_patterns", {}).items():
            if data:
                context["wind_patterns"][alt_layer] = {
                    "direction": data.get("avg_direction_cardinal", "N/A"),
                    "speed": data.get("avg_speed", 0)
                }
        
        # Extract prediction data
        trajectory_forecasts = balloon_data.get("predictions", {}).get("trajectory_forecasts", {})
        context["trajectory_forecast_count"] = len(trajectory_forecasts)
        
        # Extract anomalies
        anomalies = balloon_data.get("atmospheric_anomalies", [])
        context["anomaly_count"] = len(anomalies)
        context["anomalies"] = []
        
        for anomaly in anomalies:
            context["anomalies"].append({
                "type": anomaly.get("type", ""),
                "altitude_layer": anomaly.get("altitude_layer", ""),
                "severity": anomaly.get("severity", ""),
                "description": anomaly.get("description", "")
            })
        
        # Extract optimal altitude bands
        optimal_bands = balloon_data.get("performance_analytics", {}).get("optimal_altitude_bands", [])
        context["optimal_altitude_bands"] = []
        
        for band in optimal_bands:
            context["optimal_altitude_bands"].append({
                "band": band.get("altitude_band", ""),
                "score": band.get("efficiency_score", 0),
                "recommendation": band.get("recommendation", "")
            })
        
        # Extract weather data if available
        if weather_data:
            context["weather"] = {
                "current": weather_data.get("current", {}),
                "upper_air": {}
            }
            
            # Extract upper air data if available
            upper_air = weather_data.get("upper_air", {})
            if upper_air and "data" in upper_air:
                for altitude_data in upper_air.get("data", []):
                    if "altitude" in altitude_data:
                        alt_key = f"{altitude_data['altitude']}m"
                        context["weather"]["upper_air"][alt_key] = {
                            "temperature": altitude_data.get("temperature", 0),
                            "wind_speed": altitude_data.get("wind_speed", 0),
                            "wind_direction": altitude_data.get("wind_direction", 0)
                        }
        
        return context
    
    # Update the _call_llm_with_context method to add more robust error handling
    def _call_llm_with_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Call the Claude API with the prepared context to generate insights."""
        # Format prompt with context data
        prompt = self._format_insight_prompt(context)
        
        try:
            # Check if client is initialized
            if not self.client:
                return {
                    "operational_recommendations": ["API key is invalid or not provided."],
                    "wind_analysis": "Please provide a valid Anthropic API key to generate detailed insights.",
                    "safety_alerts": [],
                    "altitude_recommendations": [],
                    "performance_analysis": "API connection unavailable."
                }
                
            response = self.client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
)
            
            # Parse the response
            return self._parse_llm_response(response.content[0].text)
            
        except Exception as e:
            logger.exception(f"Error calling Claude API: {str(e)}")
            # Return a usable fallback response instead of raising an exception
            return {
                "operational_recommendations": [
                    f"Error generating insights: {str(e)}",
                    "Consider using basic analytics data for decision making.",
                    "Check API key and connection status."
                ],
                "wind_analysis": "Unable to analyze wind conditions due to API error.",
                "safety_alerts": [],
                "altitude_recommendations": [],
                "performance_analysis": "API connection error."
            }
    
    def _format_insight_prompt(self, context: Dict[str, Any]) -> str:
        """
        Format a prompt for Claude based on the context data.
        
        Args:
            context: Context dictionary
            
        Returns:
            Formatted prompt string
        """
        # Extract key variables for the prompt
        balloon_count = context.get("balloon_count", 0)
        active_balloons = context.get("active_balloons", 0)
        
        altitude_data = context.get("altitude_data", {})
        min_alt = altitude_data.get("min", 0)
        max_alt = altitude_data.get("max", 0)
        avg_alt = altitude_data.get("avg", 0)
        
        wind_patterns = context.get("wind_patterns", {})
        wind_pattern_str = "\n"
        for layer, data in wind_patterns.items():
            wind_pattern_str += f"- {layer.capitalize()} altitude: {data.get('direction', 'N/A')} direction at {data.get('speed', 0):.1f} km/h\n"
        
        anomalies = context.get("anomalies", [])
        anomaly_str = "\n"
        for anomaly in anomalies:
            anomaly_str += f"- {anomaly.get('description', 'Unknown anomaly')}\n"
        
        optimal_bands = context.get("optimal_altitude_bands", [])
        band_str = "\n"
        for band in optimal_bands:
            band_str += f"- {band.get('band', 'Unknown')}: {band.get('recommendation', '')}\n"
        
        # Weather data formatting
        weather_str = ""
        if "weather" in context:
            current = context["weather"].get("current", {})
            weather_str = f"""
Current surface weather:
- Temperature: {current.get('temperature', 'N/A')}°C
- Wind: {current.get('wind_speed', 'N/A')} km/h from {current.get('wind_direction', 'N/A')}°
- Pressure: {current.get('pressure', 'N/A')} hPa
- Humidity: {current.get('humidity', 'N/A')}%

Upper air conditions:
"""
            for alt, data in context["weather"].get("upper_air", {}).items():
                weather_str += f"- {alt}: {data.get('temperature', 'N/A')}°C, {data.get('wind_speed', 'N/A')} km/h\n"
        
        # Build the final prompt
        prompt = f"""You are an atmospheric science expert analyzing high-altitude balloon data. 
Generate insights based on the following balloon constellation data:

CONSTELLATION OVERVIEW:
- Total balloons: {balloon_count}
- Active balloons: {active_balloons}
- Altitude range: {min_alt:.0f}m to {max_alt:.0f}m (avg: {avg_alt:.0f}m)

WIND PATTERNS:{wind_pattern_str}

ATMOSPHERIC ANOMALIES:{anomaly_str if anomalies else ' None detected'}
  
OPTIMAL ALTITUDE BANDS:{band_str if optimal_bands else ' Data not available'}

{weather_str if weather_str else ''}

Please provide the following insights in JSON format:
1. Three key operational recommendations for balloon management today
2. Analysis of current wind conditions and their impact on balloon trajectories
3. Safety alerts or warnings based on current conditions
4. Optimal altitude recommendations with rationale
5. Performance analysis of the current constellation configuration

For each insight category, provide clear, specific guidance that would be actionable for balloon operators.
Format your response as valid JSON that can be parsed by a computer."""
        return prompt
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the LLM response into a structured format.
        
        Args:
            response_text: Raw text response from Claude
            
        Returns:
            Parsed insights dictionary
        """
        try:
            # Extract JSON from the response
            # Look for JSON-like structure in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx+1]
                insights = json.loads(json_str)
                return insights
            
            # If no JSON found, try to structure the response
            return {
                "error": "Could not parse JSON from response",
                "raw_response": response_text,
                "structured_insights": self._extract_insights_from_text(response_text)
            }
            
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract insights manually
            return {
                "error": "Invalid JSON in response",
                "raw_response": response_text,
                "structured_insights": self._extract_insights_from_text(response_text)
            }
    
    def _extract_insights_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract insights from unstructured text response.
        
        Args:
            text: Unstructured text response
            
        Returns:
            Dictionary with extracted insights
        """
        # Simple extractor for when JSON parsing fails
        insights = {
            "operational_recommendations": [],
            "wind_analysis": "",
            "safety_alerts": [],
            "altitude_recommendations": [],
            "performance_analysis": ""
        }
        
        # Look for recommendation patterns
        sections = text.split('\n\n')
        current_section = None
        
        for section in sections:
            if "recommendation" in section.lower() or "recommend" in section.lower():
                insights["operational_recommendations"].append(section.strip())
            elif "wind" in section.lower() or "trajectory" in section.lower():
                insights["wind_analysis"] += section.strip() + " "
            elif "safety" in section.lower() or "alert" in section.lower() or "warning" in section.lower():
                insights["safety_alerts"].append(section.strip())
            elif "altitude" in section.lower():
                insights["altitude_recommendations"].append(section.strip())
            elif "performance" in section.lower() or "constellation" in section.lower():
                insights["performance_analysis"] += section.strip() + " "
        
        return insights
    
    def generate_anomaly_explanation(self, anomaly: Dict[str, Any]) -> str:
        """
        Generate a detailed explanation for an atmospheric anomaly.
        
        Args:
            anomaly: Anomaly data dictionary
            
        Returns:
            Detailed explanation string
        """
        if not self.has_valid_api_key():
            return "Detailed explanations require an Anthropic API key."
        
        try:
            anomaly_type = anomaly.get("type", "unknown")
            altitude_layer = anomaly.get("altitude_layer", "unknown")
            current_value = anomaly.get("current_value", 0)
            historical_value = anomaly.get("historical_value", 0)
            diff_percent = anomaly.get("difference_percent", 0)
            
            prompt = f"""You are an atmospheric science expert explaining a weather anomaly.
Please explain the following atmospheric anomaly in detail:

- Type: {anomaly_type}
- Altitude layer: {altitude_layer}
- Current value: {current_value}
- Historical average: {historical_value}
- Difference: {diff_percent:.1f}%

Provide a detailed explanation of:
1. What might be causing this anomaly
2. How it could affect balloon operations
3. What historical precedents exist for this type of anomaly
4. What monitoring procedures should be implemented

Keep your response under 250 words and focus on practical implications for high-altitude balloon operations."""

            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.exception(f"Error generating anomaly explanation: {str(e)}")
            return f"Error generating explanation: {str(e)}"
            
    def generate_trajectory_analysis(self, balloon_id: str, 
                                    trajectory_data: List[Dict],
                                    wind_data: Optional[Dict] = None) -> str:
        """Generate a detailed analysis of a balloon's predicted trajectory."""
        if not self.has_valid_api_key():
            return "Trajectory analysis requires an Anthropic API key."
        
        if not trajectory_data:
            return "Insufficient trajectory data for analysis."
        
        try:
            # Format trajectory data
            trajectory_points = "\n"
            for point in trajectory_data:
                hours = point.get("hours_ahead", 0)
                lat = point.get("lat", 0)
                lon = point.get("lon", 0)
                alt = point.get("alt", 0)
                confidence = point.get("confidence", 0) * 100
                
                trajectory_points += f"- {hours} hours: {lat:.4f}, {lon:.4f}, altitude {alt:.0f}m (confidence: {confidence:.0f}%)\n"
            
            # Format wind data if available - FIX HERE
            wind_info = ""
            if wind_data and isinstance(wind_data, dict):  # Check that wind_data is a dictionary
                wind_info = "\nWind conditions:\n"
                for layer, data in wind_data.items():
                    if not isinstance(data, dict):  # Skip if data is not a dictionary
                        continue
                    wind_info += f"- {layer} altitude: {data.get('avg_direction_cardinal', 'N/A')} direction at {data.get('avg_speed', 0):.1f} km/h\n"
            
            prompt = f"""You are an atmospheric science expert analyzing a balloon trajectory forecast.
    Please analyze the following predicted trajectory for balloon {balloon_id}:

    {trajectory_points}

    {wind_info}

    Provide a detailed analysis including:
    1. Key geographic features or regions the balloon will encounter
    2. Potential hazards or challenges along this path
    3. Recommendations for optimal altitude adjustments to improve the trajectory
    4. Confidence assessment of this prediction and factors that could alter the trajectory

    Keep your response under 250 words and focus on practical implications for balloon operations."""

            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.exception(f"Error generating trajectory analysis: {str(e)}")
            return f"Error generating analysis: {str(e)}"