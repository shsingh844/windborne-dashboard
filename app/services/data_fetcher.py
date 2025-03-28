import aiohttp
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
import time

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
                        return None
                    
                    # Read text content for cleaning before JSON parsing
                    text_content = await response.text()
                    return self._clean_and_parse_json(text_content, hour)
                    
        except asyncio.TimeoutError:
            logger.warning(f"Timeout while fetching data from {url}")
        except aiohttp.ClientError as e:
            logger.warning(f"Error fetching data from {url}: {str(e)}")
        except Exception as e:
            logger.exception(f"Unexpected error fetching data from {url}: {str(e)}")
            
        return None
    
    def _clean_and_parse_json(self, content: str, hour: int) -> Optional[Dict]:
        """
        Clean potentially corrupted JSON and parse it.
        
        Args:
            content: Raw JSON string
            hour: The hour for logging purposes
            
        Returns:
            Parsed JSON data or None if parsing failed
        """
        try:
            # First try direct parsing
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error for hour {hour}, attempting to clean: {str(e)}")
            
            # Try some common JSON corruption fixes
            try:
                # Remove trailing commas in arrays/objects
                content = self._fix_trailing_commas(content)
                
                # Handle unquoted keys
                content = self._fix_unquoted_keys(content)
                
                # Fix missing quotes around string values
                content = self._fix_unquoted_values(content)
                
                # Try parsing again after cleanup
                return json.loads(content)
            except json.JSONDecodeError as e2:
                logger.error(f"Failed to parse JSON after cleaning for hour {hour}: {str(e2)}")
                
                # Last resort: try to extract any valid JSON objects
                return self._extract_valid_json_parts(content, hour)
    
    def _fix_trailing_commas(self, content: str) -> str:
        """Fix trailing commas in arrays and objects."""
        # Replace ",]" with "]"
        content = content.replace(",]", "]")
        # Replace ",}" with "}"
        content = content.replace(",}", "}")
        return content
    
    def _fix_unquoted_keys(self, content: str) -> str:
        """Add quotes around unquoted object keys."""
        import re
        # This is a simplified version - a full implementation would be more complex
        return re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', content)
    
    def _fix_unquoted_values(self, content: str) -> str:
        """Add quotes around unquoted string values."""
        import re
        # This is a simplified approach - real-world cases need more robust handling
        return re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*([,}])', r':"\1"\2', content)
    
    def _extract_valid_json_parts(self, content: str, hour: int) -> Optional[Dict]:
        """
        Attempt to extract valid parts from corrupted JSON.
        This is a fallback method when regular cleaning fails.
        
        Returns:
            Partially recovered data or None if recovery failed
        """
        try:
            # Try to extract the balloons array if present
            import re
            balloons_match = re.search(r'"balloons"\s*:\s*(\[.*?\])', content, re.DOTALL)
            
            if balloons_match:
                balloons_str = balloons_match.group(1)
                # Try to parse just the balloons array
                try:
                    balloons = json.loads(balloons_str)
                    logger.info(f"Partially recovered balloons data for hour {hour}")
                    return {"balloons": balloons, "partially_recovered": True}
                except json.JSONDecodeError:
                    pass
            
            # Last resort: Extract any JSON objects or arrays that look valid
            valid_objects = []
            
            # Look for objects
            for obj_match in re.finditer(r'({[^{}]*})', content):
                try:
                    obj = json.loads(obj_match.group(0))
                    valid_objects.append(obj)
                except json.JSONDecodeError:
                    pass
            
            if valid_objects:
                logger.info(f"Recovered {len(valid_objects)} objects for hour {hour}")
                return {"recovered_objects": valid_objects, "partially_recovered": True}
                
            logger.error(f"Could not recover any valid JSON for hour {hour}")
            return None
            
        except Exception as e:
            logger.exception(f"Error during JSON recovery attempt for hour {hour}: {str(e)}")
            return None
            
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
        
    def get_balloon_history(self, all_hours_data: Dict[int, Dict]) -> Dict[str, List[Dict]]:
        """
        Extract the history of each balloon across all hours.
        
        Args:
            all_hours_data: Dictionary mapping hour to hour data
            
        Returns:
            Dictionary mapping balloon ID to its history
        """
        balloon_history = {}
        
        # Process each hour of data
        for hour, hour_data in sorted(all_hours_data.items()):
            # Skip if no balloons data
            if not hour_data or 'balloons' not in hour_data:
                continue
                
            for balloon in hour_data['balloons']:
                # Skip if no ID or required position data
                if 'id' not in balloon or 'lat' not in balloon or 'lon' not in balloon:
                    continue
                    
                balloon_id = str(balloon['id'])
                
                # Add timestamp field based on hour
                balloon_with_time = balloon.copy()
                # Current time minus hours in seconds
                balloon_with_time['timestamp'] = int(time.time()) - (hour * 3600)
                
                # Initialize if this is the first time seeing this balloon
                if balloon_id not in balloon_history:
                    balloon_history[balloon_id] = []
                    
                balloon_history[balloon_id].append(balloon_with_time)
                
        # Sort each balloon's history by timestamp
        for balloon_id in balloon_history:
            balloon_history[balloon_id].sort(key=lambda x: x.get('timestamp', 0))
            
        return balloon_history