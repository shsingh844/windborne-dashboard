import aiohttp
import asyncio
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
import time
import math

logger = logging.getLogger(__name__)

class DataFetcher:
    """Service to fetch and clean data from the Windborne Systems API."""
    
    BASE_URL = "https://a.windbornesystems.com/treasure"
    
    async def fetch_hour_data(self, hour: int) -> Optional[Dict]:
        """
        Fetch data for a specific hour.
        
        Args:
            hour: Hours ago (0-23)
            
        Returns:
            Parsed JSON data or None if request failed
        """
        if not 0 <= hour <= 23:
            raise ValueError(f"Hour must be between 0 and 23, got {hour}")
            
        url = f"{self.BASE_URL}/{hour:02d}.json"
        logger.info(f"Fetching data from {url}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch data from {url}: HTTP {response.status}")
                        # Return a special marker for 404 errors to track missing data
                        if response.status == 404:
                            return {
                                "error": "404",
                                "hour": hour,
                                "status": "missing",
                                "url": url
                            }
                        return None
                    
                    # Read text content for cleaning before JSON parsing
                    text_content = await response.text()
                    
                    # Debug: Log the raw content (first 100 chars)
                    logger.info(f"Raw content for hour {hour} (first 100 chars): {text_content[:100]}")
                    
                    # Parse the raw content, extracting data even if invalid
                    coordinate_arrays = self._extract_all_coordinates(text_content)
                    
                    if coordinate_arrays and len(coordinate_arrays) > 0:
                        logger.info(f"Extracted {len(coordinate_arrays)} coordinate arrays for hour {hour}")
                        # Convert to balloon format
                        converted_data = self._convert_array_to_balloons(coordinate_arrays, hour)
                        return converted_data
                    
                    # If extraction failed, attempt regular parsing
                    try:
                        # Try standard JSON parsing with cleaning
                        cleaned_data = self._clean_and_parse_json(text_content, hour)
                        
                        # Convert array format to our expected object format with balloons
                        if isinstance(cleaned_data, list):
                            # The data appears to be a list of position arrays
                            converted_data = self._convert_array_to_balloons(cleaned_data, hour)
                            logger.info(f"Converted list data for hour {hour} into {len(converted_data.get('balloons', []))} balloons")
                            return converted_data
                        
                        # Check if balloons field exists
                        if not isinstance(cleaned_data, dict) or 'balloons' not in cleaned_data:
                            logger.warning(f"Parsed data for hour {hour} has unexpected format")
                            # Convert to expected format with empty balloons
                            return {"balloons": [], "hour": hour}
                        
                        # Success with proper format
                        logger.info(f"Successfully parsed data for hour {hour} with {len(cleaned_data.get('balloons', []))} balloons")
                        return cleaned_data
                        
                    except Exception as parse_error:
                        logger.error(f"Error parsing data for hour {hour}: {str(parse_error)}")
                        # Create placeholder with empty balloons
                        return {"balloons": [], "hour": hour}
                    
        except Exception as e:
            logger.exception(f"Unexpected error fetching data from {url}: {str(e)}")
            # Return empty data instead of error marker
            return {"balloons": [], "hour": hour}
    
    def _extract_all_coordinates(self, content: str) -> List[List[float]]:
        """
        Extract all possible coordinate arrays from content, handling NaN values.
        
        Args:
            content: String to extract coordinates from
            
        Returns:
            List of coordinate arrays
        """
        coordinates = []
        
        # Use regex to find patterns that look like coordinate arrays
        # This pattern handles NaN and negative numbers
        pattern = r'\[\s*([-\d\.]+|NaN)\s*,\s*([-\d\.]+|NaN)\s*,\s*([-\d\.]+|NaN)\s*\]'
        
        matches = re.findall(pattern, content)
        for match in matches:
            try:
                # Convert values, replacing NaN with 0
                lat = float(match[0]) if match[0] != "NaN" else 0.0
                lon = float(match[1]) if match[1] != "NaN" else 0.0
                alt = float(match[2]) if match[2] != "NaN" else 0.0
                
                # Skip coordinates that are all zeros or NaN
                if (lat == 0 and lon == 0 and alt == 0) or any(math.isnan(x) for x in [lat, lon, alt]):
                    continue
                    
                coordinates.append([lat, lon, alt])
            except (ValueError, IndexError):
                pass
        
        return coordinates
    
    def _clean_and_parse_json(self, content: str, hour: int) -> Any:
        """
        Clean potentially corrupted JSON and parse it.
        
        Args:
            content: Raw JSON string
            hour: The hour for logging purposes
            
        Returns:
            Parsed JSON data
        """
        try:
            # First try direct parsing
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error for hour {hour}, attempting to clean: {str(e)}")
            
            # Try to fix common issues
            
            # Missing opening bracket
            if not content.lstrip().startswith('[') and ('[' in content or content.lstrip().startswith('    [')):
                fixed_content = '[\n' + content
                try:
                    return json.loads(fixed_content)
                except json.JSONDecodeError:
                    pass
            
            # Remove problematic characters
            content = content.strip()
            content = content.replace(',]', ']')
            content = content.replace(',}', '}')
            content = content.replace('NaN', '0.0')
            content = content.replace('null', '0.0')
            content = content.replace('undefined', '0.0')
            
            # Remove non-ASCII characters
            content = ''.join(char for char in content if ord(char) < 128)
            
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # If all cleaning fails, extract coordinates using regex
                coordinates = self._extract_all_coordinates(content)
                if coordinates:
                    return coordinates
                
                # If all else fails, return an empty array
                logger.error(f"Failed to parse JSON for hour {hour} after all attempts")
                return []
    
    def _convert_array_to_balloons(self, array_data: List, hour: int) -> Dict[str, Any]:
        """
        Convert array data format to our expected balloon format.
        
        The raw data appears to be a list of arrays, where each array is [lat, lon, alt]
        We'll convert this to our expected format with balloons objects.
        
        Args:
            array_data: List of arrays with position data
            hour: The hour for metadata
            
        Returns:
            Dictionary with balloons field
        """
        balloons = []
        
        for i, coords in enumerate(array_data):
            # Check if the array has the expected structure
            if not isinstance(coords, list) or len(coords) < 3:
                continue
                
            # Extract coordinates, handling possible errors
            try:
                lat = float(coords[0]) if coords[0] is not None else 0
                lon = float(coords[1]) if coords[1] is not None else 0
                alt = float(coords[2]) if coords[2] is not None else 0
                
                # Skip if all values are zero (likely placeholder)
                if lat == 0 and lon == 0 and alt == 0:
                    continue
                
                # Skip if any value is NaN
                if any(math.isnan(x) for x in [lat, lon, alt]):
                    continue
                
                # Create a balloon object
                balloon = {
                    "id": f"balloon_{hour:02d}_{i:04d}",  # Generate a unique ID with leading zeros
                    "lat": lat,
                    "lon": lon,
                    "alt": alt
                }
                
                balloons.append(balloon)
            except (ValueError, TypeError, IndexError) as e:
                logger.warning(f"Error converting coordinates for hour {hour}, index {i}: {str(e)}")
        
        logger.info(f"Created {len(balloons)} balloon objects for hour {hour}")
        
        return {
            "balloons": balloons,
            "hour": hour,
            "source": "array_conversion"
        }
    
    async def fetch_all_hours(self) -> Dict[int, Dict]:
        """
        Fetch data for all 24 hours (0-23).
        
        Returns:
            Dictionary mapping hour to data
        """
        tasks = [self.fetch_hour_data(hour) for hour in range(24)]
        results = await asyncio.gather(*tasks)
        
        # Create a dictionary of non-None results
        return {hour: result for hour, result in enumerate(results) if result is not None}
    
    async def fetch_latest_data(self) -> Optional[Dict]:
        """
        Fetch the most recent data (hour 0).
        
        Returns:
            Latest data or None if request failed
        """
        return await self.fetch_hour_data(0)
        
    def get_balloon_history(self, all_hours_data: Dict[int, Dict]) -> Dict[str, Any]:
        """
        Extract the history of each balloon across all hours.
        
        Args:
            all_hours_data: Dictionary mapping hour to hour data
            
        Returns:
            Dictionary mapping balloon ID to its history
        """
        balloon_history = {}
        error_records = []
        valid_hours_count = 0
        
        # Debug: Log overall data received
        logger.info(f"Processing data from {len(all_hours_data)} hours")
        
        # Process each hour of data
        for hour, hour_data in sorted(all_hours_data.items()):
            # Check if this is an error record
            if 'error' in hour_data:
                error_record = {
                    'hour': hour,
                    'error_type': hour_data.get('error'),
                    'status': hour_data.get('status'),
                    'url': hour_data.get('url')
                }
                error_records.append(error_record)
                continue
                
            # Skip if no balloons data
            if not hour_data or 'balloons' not in hour_data:
                logger.warning(f"Hour {hour} has no 'balloons' field. Keys: {list(hour_data.keys()) if hour_data else 'None'}")
                continue
                
            valid_hours_count += 1
            balloons_count = len(hour_data['balloons'])
            logger.info(f"Processing hour {hour} with {balloons_count} balloons")
            
            # Processing balloons from this hour
            for balloon in hour_data['balloons']:
                # Skip if no required position data
                if 'lat' not in balloon or 'lon' not in balloon:
                    logger.warning(f"Skipping balloon in hour {hour} due to missing required fields. Fields: {list(balloon.keys())}")
                    continue
                
                # Get or generate balloon ID
                if 'id' not in balloon:
                    # For balloons without IDs, generate a consistent ID based on coordinates
                    lat = balloon['lat']
                    lon = balloon['lon']
                    alt = balloon.get('alt', 0)
                    balloon_id = f"balloon_{hour:02d}_{lat:.5f}_{lon:.5f}_{alt:.5f}"
                else:
                    balloon_id = str(balloon['id'])
                
                # Add timestamp field based on hour
                balloon_with_time = balloon.copy()
                # Current time minus hours in seconds
                balloon_with_time['timestamp'] = int(time.time()) - (hour * 3600)
                
                # Initialize if this is the first time seeing this balloon
                if balloon_id not in balloon_history:
                    balloon_history[balloon_id] = []
                    
                balloon_history[balloon_id].append(balloon_with_time)
        
        # Debug info
        if valid_hours_count == 0:
            logger.warning("No valid hours found with balloon data!")
        else:
            logger.info(f"Found {valid_hours_count} valid hours with balloon data")
        
        logger.info(f"Created history for {len(balloon_history)} balloons")
        
        # Add error records to return value if there are any
        result = {
            'balloon_history': balloon_history,
            'errors': error_records if error_records else None
        }
                
        # Sort each balloon's history by timestamp
        for balloon_id in balloon_history:
            balloon_history[balloon_id].sort(key=lambda x: x.get('timestamp', 0))
            
        return result