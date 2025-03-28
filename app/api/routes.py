from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import asyncio
import logging
from typing import Dict, Any, Optional
import time

from app.services.data_fetcher import DataFetcher
from app.services.data_processor import DataProcessor

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory cache for API responses
cache = {
    "balloon_data": None,
    "last_updated": 0,
    "cache_duration": 60 * 5,  # 5 minutes in seconds
    "is_updating": False
}

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
            data_fetcher = DataFetcher()
            data_processor = DataProcessor()
            
            # Fetch all hours of data
            all_hours_data = await data_fetcher.fetch_all_hours()
            
            if not all_hours_data:
                raise HTTPException(status_code=500, detail="Failed to fetch data from Windborne API")
            
            # Extract balloon history
            balloon_history = data_fetcher.get_balloon_history(all_hours_data)
            
            # Process the data
            processed_data = data_processor.process_balloon_history(balloon_history)
            
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

@router.get("/api/refresh")
async def refresh_data(background_tasks: BackgroundTasks):
    """Force a refresh of the balloon data."""
    # Schedule a background task to refresh the data
    background_tasks.add_task(refresh_data_task)
    
    return {"message": "Data refresh scheduled"}

async def refresh_data_task():
    """Background task to refresh all data."""
    if cache["is_updating"]:
        logger.info("Data refresh already in progress")
        return
    
    cache["is_updating"] = True
    try:
        data_fetcher = DataFetcher()
        data_processor = DataProcessor()
        
        # Fetch all hours of data
        all_hours_data = await data_fetcher.fetch_all_hours()
        
        if not all_hours_data:
            logger.error("Failed to fetch data from Windborne API")
            return
        
        # Extract balloon history
        balloon_history = data_fetcher.get_balloon_history(all_hours_data)
        
        # Process the data
        processed_data = data_processor.process_balloon_history(balloon_history)
        
        # Update cache
        cache["balloon_data"] = processed_data
        cache["last_updated"] = time.time()
        
        logger.info("Balloon data refreshed successfully")
    except Exception as e:
        logger.exception(f"Error refreshing balloon data: {str(e)}")
    finally:
        cache["is_updating"] = False