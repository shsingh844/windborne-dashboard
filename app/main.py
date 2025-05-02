import logging
import os
from fastapi import FastAPI, Request, Depends, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from app.api.routes import router as api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Create FastAPI app
app = FastAPI(title="Windborne Constellation Tracker")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=Path("app/static")), name="static")

# Set up templates
templates = Jinja2Templates(directory="app/templates")

# Include API routes
app.include_router(api_router, prefix="")

@app.get("/")
async def home(request: Request):
    """Serve the home page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/map")
async def map_view(request: Request):
    """Serve the map view page."""
    return templates.TemplateResponse("map_view.html", {"request": request})

@app.get("/analytics")
async def analytics(request: Request):
    """Serve the analytics page."""
    return templates.TemplateResponse("analytics.html", {"request": request})

@app.get("/insights")
async def insights(request: Request):
    """Serve the insights page."""
    return templates.TemplateResponse("insights.html", {"request": request})

@app.get("/enhanced-insights")
async def enhanced_insights(request: Request):
    """Serve the enhanced insights page."""
    return templates.TemplateResponse("enhanced-insights.html", {"request": request})

@app.on_event("startup")
async def startup_event():
    """Initialize data on startup."""
    from app.api.routes import refresh_data_task
    
    # Start a background task to prefetch data on startup
    background_tasks = BackgroundTasks()
    background_tasks.add_task(refresh_data_task)
    await refresh_data_task()
    
    logging.info("Application started, initial data fetch scheduled")

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get the port from the environment or use a default
    port = int(os.environ.get("PORT", 8000))
    
    # Run the application
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
    # Use the --reload option for development