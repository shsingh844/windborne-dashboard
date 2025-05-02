from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, Depends
from fastapi.responses import JSONResponse
import asyncio
import logging
from typing import Dict, Any, Optional
import time
import json

from app.services.data_fetcher import DataFetcher
from app.services.data_processor import DataProcessor
from app.services.weather_data_service import WeatherDataService
from app.services.llm_insights_service import LLMInsightsService

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory cache for API responses
cache = {
    "balloon_data": None,
    "enhanced_balloon_data": None,
    "weather_data": None,
    "last_updated": 0,
    "cache_duration": 60 * 5,  # 5 minutes in seconds
    "is_updating": False
}

# Initialize services
data_fetcher = DataFetcher()
data_processor = DataProcessor()
weather_service = WeatherDataService()
llm_service = LLMInsightsService()

@router.get("/api/balloons")
async def get_balloons():
    """Get all balloon data with their history."""
    # Check if cache is valid
    current_time = time.time()
    if (cache["balloon_data"] is not None and 
        current_time - cache["last_updated"] < cache["cache_duration"]):
        logger.info("Returning cached balloon data")
        return cache["balloon_data"]
    
    # If not already updating, trigger an update
    if not cache["is_updating"]:
        cache["is_updating"] = True
        try:
            # Fetch and process data
            
            # Fetch all hours of data
            all_hours_data = await data_fetcher.fetch_all_hours()
            
            if not all_hours_data:
                raise HTTPException(status_code=500, detail="Failed to fetch data from Windborne API")
            
            # Extract balloon history
            balloon_data = data_fetcher.get_balloon_history(all_hours_data)
            
            # Process the data
            processed_data = data_processor.process_balloon_history(balloon_data)
            
            # Update cache
            cache["balloon_data"] = processed_data
            cache["last_updated"] = current_time
            
            logger.info("Balloon data updated successfully")
            return processed_data
            
        except Exception as e:
            logger.exception(f"Error fetching balloon data: {str(e)}")
            # If we have cached data, return it even if expired
            if cache["balloon_data"] is not None:
                return cache["balloon_data"]
            raise HTTPException(status_code=500, detail=f"Error processing balloon data: {str(e)}")
        finally:
            cache["is_updating"] = False
    else:
        # If update is in progress, return cached data or wait briefly for update
        if cache["balloon_data"] is not None:
            return cache["balloon_data"]
        
        # Wait a bit for the update to complete
        await asyncio.sleep(2)
        
        if cache["balloon_data"] is not None:
            return cache["balloon_data"]
        
        raise HTTPException(status_code=503, detail="Data is being updated, please try again shortly")

@router.get("/api/enhanced-balloon-data")
async def get_enhanced_balloon_data():
    """Get enhanced balloon data with predictions and analytics."""
    # Check if cache is valid
    current_time = time.time()
    if (cache["enhanced_balloon_data"] is not None and 
        current_time - cache["last_updated"] < cache["cache_duration"]):
        logger.info("Returning cached enhanced balloon data")
        return cache["enhanced_balloon_data"]
    
    # If not already updating, trigger an update
    if not cache["is_updating"]:
        cache["is_updating"] = True
        try:
            # Fetch and process data
            
            # Fetch all hours of data
            all_hours_data = await data_fetcher.fetch_all_hours()
            
            if not all_hours_data:
                raise HTTPException(status_code=500, detail="Failed to fetch data from Windborne API")
            
            # Extract balloon history
            balloon_data = data_fetcher.get_balloon_history(all_hours_data)
            
            # Process the data with enhanced analytics
            processed_data = data_processor.process_balloon_history(balloon_data)
            
            # Update cache
            cache["enhanced_balloon_data"] = processed_data
            cache["balloon_data"] = processed_data  # Also update regular cache
            cache["last_updated"] = current_time
            
            logger.info("Enhanced balloon data updated successfully")
            return processed_data
            
        except Exception as e:
            logger.exception(f"Error fetching enhanced balloon data: {str(e)}")
            # If we have cached data, return it even if expired
            if cache["enhanced_balloon_data"] is not None:
                return cache["enhanced_balloon_data"]
            raise HTTPException(status_code=500, detail=f"Error processing enhanced balloon data: {str(e)}")
        finally:
            cache["is_updating"] = False
    else:
        # If update is in progress, return cached data or wait briefly for update
        if cache["enhanced_balloon_data"] is not None:
            return cache["enhanced_balloon_data"]
        
        # Wait a bit for the update to complete
        await asyncio.sleep(2)
        
        if cache["enhanced_balloon_data"] is not None:
            return cache["enhanced_balloon_data"]
        
        raise HTTPException(status_code=503, detail="Data is being updated, please try again shortly")

@router.get("/api/weather-data")
async def get_weather_data():
    """Get weather data for balloon locations."""
    # Check if cache is valid
    current_time = time.time()
    if (cache["weather_data"] is not None and 
        current_time - cache["last_updated"] < cache["cache_duration"]):
        logger.info("Returning cached weather data")
        return cache["weather_data"]
    
    # If balloon data not available, fetch it first
    if cache["balloon_data"] is None:
        await get_balloons()
    
    # Get a representative balloon for each cluster
    try:
        balloons = cache["balloon_data"]["balloons"]
        
        # Fetch weather data for a grid of balloon positions
        weather_grid = weather_service.fetch_weather_for_balloon_grid(balloons)
        
        # Get upper air data for a central location
        upper_air = None
        if balloons and len(balloons) > 0:
            # Use the first balloon as a reference point
            ref_balloon = balloons[0]
            if "latest" in ref_balloon and "lat" in ref_balloon["latest"] and "lon" in ref_balloon["latest"]:
                upper_air = weather_service.get_upper_air_data(
                    ref_balloon["latest"]["lat"], 
                    ref_balloon["latest"]["lon"]
                )
        
        # Combine data
        weather_data = {
            "grid": weather_grid,
            "upper_air": upper_air
        }
        
        # Cache the result
        cache["weather_data"] = weather_data
        
        return weather_data
        
    except Exception as e:
        logger.exception(f"Error fetching weather data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching weather data: {str(e)}")

@router.post("/api/validate-anthropic-key")
async def validate_anthropic_key(request: Request):
    """Validate an Anthropic API key."""
    try:
        data = await request.json()
        api_key = data.get("key", "")
        
        if not api_key:
            return JSONResponse(
                status_code=400,
                content={"valid": False, "message": "No API key provided"}
            )
        
        # Simple validation (just check the format for now)
        if api_key.startswith("sk-ant-"):
            return JSONResponse(
                status_code=200,
                content={"valid": True, "message": "API key validated successfully"}
            )
        else:
            return JSONResponse(
                status_code=200,
                content={"valid": False, "message": "Invalid API key format"}
            )
            
    except Exception as e:
        logger.exception(f"Error validating API key: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"valid": False, "message": f"Error validating API key: {str(e)}"}
        )

@router.post("/api/generate-insights")
async def generate_insights(request: Request):
    """Generate insights from balloon and weather data using Claude."""
    try:
        data = await request.json()
        api_key = data.get("api_key", "")
        balloon_data = data.get("balloon_data", {})
        weather_data = data.get("weather_data", None)
        
        if not api_key:
            return JSONResponse(
                status_code=400,
                content={"error": "No API key provided"}
            )
        
        if not balloon_data:
            return JSONResponse(
                status_code=400,
                content={"error": "No balloon data provided"}
            )
        
        # Create insights service with the provided key
        insights_service = LLMInsightsService(api_key)
        
        # Generate insights
        insights = insights_service.generate_balloon_insights(balloon_data, weather_data)
        
        return JSONResponse(
            status_code=200,
            content=insights
        )
            
    except Exception as e:
        logger.exception(f"Error generating insights: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error generating insights: {str(e)}"}
        )

@router.post("/api/explain-anomaly")
async def explain_anomaly(request: Request):
    """Generate an explanation for an atmospheric anomaly using Claude."""
    try:
        data = await request.json()
        api_key = data.get("api_key", "")
        anomaly = data.get("anomaly", {})
        
        if not api_key:
            return JSONResponse(
                status_code=400,
                content={"error": "No API key provided"}
            )
        
        if not anomaly:
            return JSONResponse(
                status_code=400,
                content={"error": "No anomaly data provided"}
            )
        
        # Create insights service with the provided key
        insights_service = LLMInsightsService(api_key)
        
        # Generate explanation
        explanation = insights_service.generate_anomaly_explanation(anomaly)
        
        return JSONResponse(
            status_code=200,
            content={"explanation": explanation}
        )
            
    except Exception as e:
        logger.exception(f"Error explaining anomaly: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error explaining anomaly: {str(e)}"}
        )

@router.post("/api/trajectory-analysis")
async def analyze_trajectory(request: Request):
    """Generate an analysis of a balloon's trajectory using Claude."""
    try:
        data = await request.json()
        api_key = data.get("api_key", "")
        balloon_id = data.get("balloon_id", "")
        trajectory_data = data.get("trajectory_data", [])
        wind_data = data.get("wind_data", None)
        
        if not api_key:
            return JSONResponse(
                status_code=400,
                content={"error": "No API key provided"}
            )
        
        if not balloon_id or not trajectory_data:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing trajectory data"}
            )
        
        # Create insights service with the provided key
        insights_service = LLMInsightsService(api_key)
        
        # Generate analysis
        analysis = insights_service.generate_trajectory_analysis(balloon_id, trajectory_data, wind_data)
        
        return JSONResponse(
            status_code=200,
            content={"analysis": analysis}
        )
            
    except Exception as e:
        logger.exception(f"Error analyzing trajectory: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error analyzing trajectory: {str(e)}"}
        )

@router.get("/api/refresh")
async def refresh_data(background_tasks: BackgroundTasks):
    """Force a refresh of all data."""
    # Schedule a background task to refresh the data
    background_tasks.add_task(refresh_data_task)
    
    return {"message": "Data refresh scheduled"}

@router.get("/api/status")
async def get_status():
    """Get the current status of the data."""
    if cache["balloon_data"] is None:
        return {
            "status": "No data available",
            "last_updated": None,
            "data_quality": {
                "missing_hours": 0,
                "invalid_format_hours": 0,
                "total_errors": 0,
                "total_hours": 24,
                "available_hours": 0
            }
        }
    
    data_quality = cache["balloon_data"].get("data_quality", {})
    return {
        "status": "Data available",
        "last_updated": cache["last_updated"],
        "data_quality": data_quality
    }

async def refresh_data_task():
    """Background task to refresh all data."""
    if cache["is_updating"]:
        logger.info("Data refresh already in progress")
        return
    
    cache["is_updating"] = True
    try:
        # Fetch all hours of data
        all_hours_data = await data_fetcher.fetch_all_hours()
        
        if not all_hours_data:
            logger.error("Failed to fetch data from Windborne API")
            return
        
        # Extract balloon history
        balloon_data = data_fetcher.get_balloon_history(all_hours_data)
        
        # Process the data
        processed_data = data_processor.process_balloon_history(balloon_data)
        
        # Update cache
        cache["balloon_data"] = processed_data
        cache["enhanced_balloon_data"] = processed_data
        cache["last_updated"] = time.time()
        
        # Clear weather data cache to force refresh
        cache["weather_data"] = None
        
        logger.info("All data refreshed successfully")
    except Exception as e:
        logger.exception(f"Error refreshing data: {str(e)}")
    finally:
        cache["is_updating"] = False